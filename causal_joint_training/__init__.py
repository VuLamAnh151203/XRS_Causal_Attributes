"""Standalone causal recommendation and explanation training package."""

from .config import JointTrainingConfig, load_config
from .model import CausalJointModel

__all__ = ["CausalJointModel", "JointTrainingConfig", "load_config"]
