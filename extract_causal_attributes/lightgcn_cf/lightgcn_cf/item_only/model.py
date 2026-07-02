"""Item-only LightGCN variant with fixed random users and trainable items."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..core.graph import propagate


class ItemOnlyLightGCN(nn.Module):
    """LightGCN where user ego embeddings are random but not trainable."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        self.user_embedding.weight.requires_grad_(False)

    def ego_embeddings(self) -> torch.Tensor:
        return torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)

    def forward(self, normalized_adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        final_embeddings = propagate(
            self.ego_embeddings(), normalized_adjacency, self.num_layers
        )
        return torch.split(final_embeddings, (self.num_users, self.num_items), dim=0)

    def bpr_loss(
        self,
        normalized_adjacency: torch.Tensor,
        users: torch.Tensor,
        positive_items: torch.Tensor,
        negative_items: torch.Tensor,
        l2_regularization: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_embeddings, item_embeddings = self(normalized_adjacency)
        users_final = user_embeddings[users]
        positives_final = item_embeddings[positive_items]
        negatives_final = item_embeddings[negative_items]
        positive_scores = (users_final * positives_final).sum(dim=1)
        negative_scores = (users_final * negatives_final).sum(dim=1)
        ranking_loss = F.softplus(negative_scores - positive_scores).mean()

        positives_ego = self.item_embedding(positive_items)
        negatives_ego = self.item_embedding(negative_items)
        regularization_loss = (
            positives_ego.square().sum() + negatives_ego.square().sum()
        ) / (2.0 * users.shape[0])
        loss = ranking_loss + l2_regularization * regularization_loss
        return loss, ranking_loss.detach(), regularization_loss.detach()
