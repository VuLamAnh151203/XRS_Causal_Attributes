"""LightGCN training and exact graph edge-drop utilities."""

from .graph import build_normalized_adjacency, propagate
from .metrics import evaluate
from .attribute_perturbation import (
    run_candidate_attribute_drops,
    run_support_dict_attribute_drops,
)
from .perturbation import run_edge_drop
from .pair_metric_perturbation import run_top_m_attribute_perturbation_evaluation

__all__ = [
    "build_normalized_adjacency",
    "evaluate",
    "propagate",
    "run_candidate_attribute_drops",
    "run_edge_drop",
    "run_support_dict_attribute_drops",
    "run_top_m_attribute_perturbation_evaluation",
]
