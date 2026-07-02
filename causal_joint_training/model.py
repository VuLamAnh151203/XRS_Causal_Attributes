"""Causal recommendation model and frozen-LLM soft-prefix generation branch."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .config import GenerationConfig, ModelConfig
from .data import CausalLabel


class ModelError(ValueError):
    """Raised when model inputs or optional generation dependencies are invalid."""


class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


def signed_target_distribution(
    labels: Sequence[CausalLabel], vocabulary_size: int, device: torch.device | str | None = None
) -> torch.Tensor:
    """Build normalized positive/negative-channel KL targets from signed OMP coefficients."""

    target = torch.zeros((len(labels), 2 * vocabulary_size), dtype=torch.float32, device=device)
    for row, label in enumerate(labels):
        total = 0.0
        for attribute_index, coefficient in zip(label.attribute_indices, label.coefficients):
            if not 0 <= attribute_index < vocabulary_size:
                raise ModelError(f"Causal label attribute index {attribute_index} is out of range.")
            magnitude = abs(float(coefficient))
            if magnitude == 0:
                continue
            channel_index = attribute_index if coefficient > 0 else vocabulary_size + attribute_index
            target[row, channel_index] += magnitude
            total += magnitude
        if total <= 0:
            raise ModelError(f"Causal label pair_index {label.pair_index} has no nonzero coefficients.")
        target[row] /= total
    return target


class CausalJointModel(nn.Module):
    """Trainable causal modules over frozen propagated LightGCN embeddings."""

    def __init__(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        vocabulary_size: int,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        if user_embeddings.ndim != 2 or item_embeddings.ndim != 2:
            raise ModelError("User and item embeddings must be rank-2 tensors.")
        if user_embeddings.shape[1] != config.embedding_dim or item_embeddings.shape[1] != config.embedding_dim:
            raise ModelError("Frozen embedding width does not match model.embedding_dim.")
        self.config = config
        self.vocabulary_size = vocabulary_size
        self.user_embeddings = nn.Embedding.from_pretrained(user_embeddings.float(), freeze=True)
        self.item_embeddings = nn.Embedding.from_pretrained(item_embeddings.float(), freeze=True)
        self.extractor = TwoLayerMLP(2 * config.embedding_dim, config.extractor_hidden_dim, 2 * vocabulary_size)
        self.attribute_embeddings = nn.Parameter(torch.empty(vocabulary_size, config.embedding_dim))
        nn.init.xavier_uniform_(self.attribute_embeddings)
        self.preference_encoder = TwoLayerMLP(
            3 * config.embedding_dim, config.preference_hidden_dim, config.preference_dim
        )
        self.causal_signal = nn.Linear(config.preference_dim, 1)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def pair_embeddings(self, users: torch.Tensor, items: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_embeddings(users), self.item_embeddings(items)

    def causal_logits_from_embeddings(
        self, user_embeddings: torch.Tensor, item_embeddings: torch.Tensor
    ) -> torch.Tensor:
        return self.extractor(torch.cat((user_embeddings, item_embeddings), dim=-1))

    def causal_outputs(
        self, users: torch.Tensor, items: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_embeddings, item_embeddings = self.pair_embeddings(users, items)
        logits = self.causal_logits_from_embeddings(user_embeddings, item_embeddings)
        probabilities = F.softmax(logits, dim=-1)
        positive, negative = probabilities.split(self.vocabulary_size, dim=-1)
        dense_attributes = (positive - negative) @ self.attribute_embeddings
        return logits, probabilities, dense_attributes

    def preference(
        self, users: torch.Tensor, items: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        user_embeddings, item_embeddings = self.pair_embeddings(users, items)
        logits = self.causal_logits_from_embeddings(user_embeddings, item_embeddings)
        probabilities = F.softmax(logits, dim=-1)
        positive, negative = probabilities.split(self.vocabulary_size, dim=-1)
        dense_attributes = (positive - negative) @ self.attribute_embeddings
        preference = self.preference_encoder(
            torch.cat((user_embeddings, item_embeddings, dense_attributes), dim=-1)
        )
        return preference, probabilities, user_embeddings, item_embeddings

    def score_pairs(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        preference, _, user_embeddings, item_embeddings = self.preference(users, items)
        cf_score = (user_embeddings * item_embeddings).sum(dim=-1)
        causal_score = self.causal_signal(preference).squeeze(-1)
        return cf_score + F.softplus(self.alpha) * causal_score

    def bpr_loss(
        self, users: torch.Tensor, positive_items: torch.Tensor, negative_items: torch.Tensor
    ) -> torch.Tensor:
        positive_scores = self.score_pairs(users, positive_items)
        negative_scores = self.score_pairs(users, negative_items)
        return F.softplus(negative_scores - positive_scores).mean()

    def kl_loss(self, users: torch.Tensor, items: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.causal_outputs(users, items)
        if target.shape != logits.shape:
            raise ModelError(f"KL target shape {tuple(target.shape)} does not match logits {tuple(logits.shape)}.")
        return F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")

    @torch.no_grad()
    def top_attribute_indices(self, users: torch.Tensor, items: torch.Tensor, count: int) -> torch.Tensor:
        _, probabilities, _ = self.causal_outputs(users, items)
        positive, negative = probabilities.split(self.vocabulary_size, dim=-1)
        return (positive + negative).topk(min(count, self.vocabulary_size), dim=-1).indices

    def trainable_state_dict(self) -> dict[str, Any]:
        return {
            "extractor": self.extractor.state_dict(),
            "attribute_embeddings": self.attribute_embeddings.detach().cpu(),
            "preference_encoder": self.preference_encoder.state_dict(),
            "causal_signal": self.causal_signal.state_dict(),
            "alpha": self.alpha.detach().cpu(),
        }

    def load_trainable_state_dict(self, state: Mapping[str, Any]) -> None:
        self.extractor.load_state_dict(state["extractor"])
        self.attribute_embeddings.data.copy_(state["attribute_embeddings"].to(self.attribute_embeddings.device))
        self.preference_encoder.load_state_dict(state["preference_encoder"])
        self.causal_signal.load_state_dict(state["causal_signal"])
        self.alpha.data.copy_(state["alpha"].to(self.alpha.device))


class SoftPromptGenerator(TwoLayerMLP):
    def __init__(self, input_dim: int, hidden_dim: int, llm_hidden_dim: int, token_count: int) -> None:
        super().__init__(input_dim, hidden_dim, llm_hidden_dim * token_count)
        self.llm_hidden_dim = llm_hidden_dim
        self.token_count = token_count

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return super().forward(values).reshape(values.shape[0], self.token_count, self.llm_hidden_dim)


@dataclass(frozen=True)
class LanguageBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor | None


def hard_prompt(instruction: str, title: str, user_profile: str, item_profile: str) -> str:
    return (
        f"Instruction: {instruction}\n"
        f"Book title: {title}\n"
        f"User profile: {user_profile}\n"
        f"Item profile: {item_profile}\n"
        "Explanation:"
    )


def _hidden_size(llm: nn.Module) -> int:
    config = getattr(llm, "config", None)
    for key in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, key, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ModelError("Could not resolve frozen LLM hidden size.")


class FrozenSoftPromptLM(nn.Module):
    """Inject trainable soft-prefix embeddings into an otherwise frozen causal LM."""

    def __init__(
        self,
        llm: nn.Module,
        tokenizer: Any,
        recommender_embedding_dim: int,
        preference_dim: int,
        config: GenerationConfig,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.config = config
        for parameter in self.llm.parameters():
            parameter.requires_grad = False
        hidden = _hidden_size(llm)
        self.causal_prompt = SoftPromptGenerator(
            preference_dim, config.prompt_hidden_dim, hidden, config.causal_prompt_tokens
        )
        self.user_prompt = SoftPromptGenerator(
            recommender_embedding_dim, config.prompt_hidden_dim, hidden, config.user_prompt_tokens
        )
        self.item_prompt = SoftPromptGenerator(
            recommender_embedding_dim, config.prompt_hidden_dim, hidden, config.item_prompt_tokens
        )

    @property
    def prefix_token_count(self) -> int:
        return (
            self.config.causal_prompt_tokens
            + self.config.user_prompt_tokens
            + self.config.item_prompt_tokens
        )

    @classmethod
    def from_pretrained(
        cls,
        config: GenerationConfig,
        recommender_embedding_dim: int,
        preference_dim: int,
        device: torch.device | str | None = None,
    ) -> "FrozenSoftPromptLM":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for the generation branch.") from exc
        kwargs: dict[str, Any] = {}
        if config.load_in_4bit:
            if importlib.util.find_spec("bitsandbytes") is None:
                raise RuntimeError(
                    "generation.load_in_4bit is true but bitsandbytes is not installed. "
                    "Install bitsandbytes or set generation.load_in_4bit: false."
                )
            from transformers import BitsAndBytesConfig

            dtype = getattr(torch, config.compute_dtype, None)
            if dtype is None:
                raise ModelError(f"Unsupported generation.compute_dtype {config.compute_dtype!r}.")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
            )
            target_device = torch.device(device) if device is not None else None
            kwargs["device_map"] = (
                {"": str(target_device)}
                if target_device is not None and target_device.type == "cuda"
                else "auto"
            )
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            local_files_only=config.local_files_only,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        llm = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            local_files_only=config.local_files_only,
            **kwargs,
        )
        return cls(llm, tokenizer, recommender_embedding_dim, preference_dim, config)

    def _encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        values = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return [int(value) for value in values]

    def language_batch(
        self,
        prompts: Sequence[str],
        targets: Sequence[str] | None,
        device: torch.device,
    ) -> LanguageBatch:
        if targets is not None and len(prompts) != len(targets):
            raise ModelError("Prompt and target batch sizes differ.")
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ModelError("Tokenizer requires a pad token.")
        sequences: list[list[int]] = []
        labels: list[list[int]] | None = [] if targets is not None else None
        for index, prompt in enumerate(prompts):
            prompt_ids = self._encode(prompt, add_special_tokens=True)
            target_ids: list[int] = []
            if targets is not None:
                target_ids = self._encode(targets[index], add_special_tokens=False)
                eos_id = self.tokenizer.eos_token_id
                if eos_id is not None:
                    target_ids.append(int(eos_id))
                target_ids = target_ids[: self.config.max_text_tokens]
                prompt_ids = prompt_ids[: self.config.max_text_tokens - len(target_ids)]
            else:
                prompt_ids = prompt_ids[: self.config.max_text_tokens]
            sequences.append(prompt_ids + target_ids)
            if labels is not None:
                labels.append([-100] * len(prompt_ids) + target_ids)
        max_length = max(len(values) for values in sequences)
        input_rows, attention_rows, label_rows = [], [], []
        for index, values in enumerate(sequences):
            values = values[:max_length]
            padding = max_length - len(values)
            input_rows.append(values + [pad_id] * padding)
            attention_rows.append([1] * len(values) + [0] * padding)
            if labels is not None:
                label_values = labels[index][:max_length]
                label_rows.append(label_values + [-100] * (max_length - len(label_values)))
        return LanguageBatch(
            input_ids=torch.tensor(input_rows, dtype=torch.long, device=device),
            attention_mask=torch.tensor(attention_rows, dtype=torch.long, device=device),
            labels=torch.tensor(label_rows, dtype=torch.long, device=device) if labels is not None else None,
        )

    def prefix_embeddings(
        self,
        preference: torch.Tensor,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        target_device = self.llm_input_device()
        return torch.cat(
            (
                self.causal_prompt(preference.to(self.module_device(self.causal_prompt))).to(target_device),
                self.user_prompt(user_embeddings.to(self.module_device(self.user_prompt))).to(target_device),
                self.item_prompt(item_embeddings.to(self.module_device(self.item_prompt))).to(target_device),
            ),
            dim=1,
        )

    @staticmethod
    def module_device(module: nn.Module) -> torch.device:
        return next(module.parameters()).device

    def llm_input_device(self) -> torch.device:
        return next(self.llm.get_input_embeddings().parameters()).device

    def move_prompt_generators(self, device: torch.device) -> None:
        self.causal_prompt.to(device)
        self.user_prompt.to(device)
        self.item_prompt.to(device)

    def forward(
        self,
        preference: torch.Tensor,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        prompts: Sequence[str],
        targets: Sequence[str],
    ) -> Any:
        device = self.llm_input_device()
        batch = self.language_batch(prompts, targets, device)
        prefix = self.prefix_embeddings(preference, user_embeddings, item_embeddings)
        text_embeddings = self.llm.get_input_embeddings()(batch.input_ids)
        prefix = prefix.to(dtype=text_embeddings.dtype)
        inputs_embeds = torch.cat((prefix, text_embeddings), dim=1)
        prefix_mask = torch.ones((len(prompts), self.prefix_token_count), dtype=batch.attention_mask.dtype, device=device)
        attention_mask = torch.cat((prefix_mask, batch.attention_mask), dim=1)
        prefix_labels = torch.full((len(prompts), self.prefix_token_count), -100, dtype=torch.long, device=device)
        labels = torch.cat((prefix_labels, batch.labels), dim=1)
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(
        self,
        preference: torch.Tensor,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        prompts: Sequence[str],
    ) -> list[str]:
        device = self.llm_input_device()
        batch = self.language_batch(prompts, None, device)
        prefix = self.prefix_embeddings(preference, user_embeddings, item_embeddings)
        text_embeddings = self.llm.get_input_embeddings()(batch.input_ids)
        inputs_embeds = torch.cat((prefix.to(dtype=text_embeddings.dtype), text_embeddings), dim=1)
        prefix_mask = torch.ones((len(prompts), self.prefix_token_count), dtype=batch.attention_mask.dtype, device=device)
        attention_mask = torch.cat((prefix_mask, batch.attention_mask), dim=1)
        output_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens,
        )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def prompt_state_dict(self) -> dict[str, Any]:
        return {
            "causal_prompt": self.causal_prompt.state_dict(),
            "user_prompt": self.user_prompt.state_dict(),
            "item_prompt": self.item_prompt.state_dict(),
        }

    def load_prompt_state_dict(self, state: Mapping[str, Any]) -> None:
        self.causal_prompt.load_state_dict(state["causal_prompt"])
        self.user_prompt.load_state_dict(state["user_prompt"])
        self.item_prompt.load_state_dict(state["item_prompt"])
