"""Model loading and management for LLM radiation experiments."""

from llm_radiation.models.control_models import (
    BertTinyControl,
    CNNControl,
    LogisticRegressionControl,
    load_control_model,
)
from llm_radiation.models.registry import (
    MODEL_REGISTRY,
    convert_to_fp16,
    load_model,
    restore_weights,
    snapshot_weights,
)

__all__ = [
    # Registry
    "MODEL_REGISTRY",
    "load_model",
    "convert_to_fp16",
    "snapshot_weights",
    "restore_weights",
    # Control models
    "LogisticRegressionControl",
    "CNNControl",
    "BertTinyControl",
    "load_control_model",
]
