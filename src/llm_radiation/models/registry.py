"""Model registry and loading utilities for LLM radiation experiments.

This module provides a central registry of supported models with their HuggingFace IDs,
along with utilities for loading models, managing weights, and performing dtype conversions.
"""

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Model registry mapping short names to HuggingFace model IDs
MODEL_REGISTRY: dict[str, str] = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "gemma-2b": "google/gemma-2b",
    "llama3-8b": "meta-llama/Meta-Llama-3.1-8B",
    "mistral-7b": "mistralai/Mistral-7B-v0.3",
    "gemma-7b": "google/gemma-7b",
    "llama3-13b": "meta-llama/Meta-Llama-3.1-13B",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-v0.1",
}


def load_model(
    model_id: str,
    dtype: str = "float16",
    device_map: str = "auto",
    revision: str = "main",
    trust_remote_code: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal language model and tokenizer from HuggingFace.

    Args:
        model_id: HuggingFace model ID or short name from MODEL_REGISTRY
        dtype: Data type for model weights (default: "float16")
        device_map: Device mapping strategy (default: "auto")
        revision: Model revision/branch (default: "main")
        trust_remote_code: Allow execution of remote code (default: False)

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ValueError: If model_id is not found in registry and doesn't appear to be a valid HF ID
        RuntimeError: If model loading fails

    Examples:
        >>> model, tokenizer = load_model("tinyllama")
        >>> model, tokenizer = load_model("meta-llama/Meta-Llama-3.1-8B")
    """
    # Resolve short name from registry
    resolved_id = MODEL_REGISTRY.get(model_id, model_id)

    # Convert dtype string to torch dtype
    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

    try:
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            resolved_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

        return model, tokenizer

    except Exception as e:
        raise RuntimeError(f"Failed to load model '{resolved_id}': {str(e)}") from e


def convert_to_fp16(model: torch.nn.Module) -> None:
    """Convert all floating point parameters to float16 in-place.

    This converts all float32 and float64 parameters to float16, which is useful
    for reducing memory usage and enabling bit-flip experiments on fp16 weights.

    Args:
        model: PyTorch model to convert

    Note:
        This modifies the model in-place and affects all parameters and buffers.

    Examples:
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> convert_to_fp16(model)
        >>> assert next(model.parameters()).dtype == torch.float16
    """
    for param in model.parameters():
        if param.dtype in (torch.float32, torch.float64):
            param.data = param.data.half()

    # Also convert buffers (like running statistics in batchnorm)
    for buffer_name, buffer in model.named_buffers():
        if buffer.dtype in (torch.float32, torch.float64):
            model.register_buffer(buffer_name, buffer.half())


def snapshot_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Create a snapshot of all model parameters.

    Creates deep copies of all parameters moved to CPU for safe storage.
    Useful for saving model state before bit-flip injection.

    Args:
        model: PyTorch model to snapshot

    Returns:
        Dictionary mapping parameter names to cloned CPU tensors

    Examples:
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> snapshot = snapshot_weights(model)
        >>> # Later restore from snapshot
        >>> restore_weights(model, snapshot)
    """
    snapshot = {}
    for name, param in model.named_parameters():
        snapshot[name] = param.data.clone().cpu()
    return snapshot


def restore_weights(model: torch.nn.Module, snapshot: dict[str, torch.Tensor]) -> None:
    """Restore model weights from a snapshot.

    Restores parameters from a previously created snapshot, moving tensors
    to the appropriate device and dtype as needed.

    Args:
        model: PyTorch model to restore
        snapshot: Dictionary of parameter names to tensors (from snapshot_weights)

    Raises:
        ValueError: If snapshot keys don't match model parameters

    Note:
        This modifies the model in-place.

    Examples:
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> snapshot = snapshot_weights(model)
        >>> # ... perform some modifications ...
        >>> restore_weights(model, snapshot)  # Restore original weights
    """
    model_params = dict(model.named_parameters())

    # Verify all snapshot keys exist in model
    missing_keys = set(snapshot.keys()) - set(model_params.keys())
    if missing_keys:
        raise ValueError(f"Snapshot contains keys not in model: {missing_keys}")

    extra_keys = set(model_params.keys()) - set(snapshot.keys())
    if extra_keys:
        raise ValueError(f"Model contains keys not in snapshot: {extra_keys}")

    # Restore each parameter
    for name, param in model.named_parameters():
        # Move snapshot tensor to same device and dtype as model parameter
        restored = snapshot[name].to(device=param.device, dtype=param.dtype)
        param.data.copy_(restored)
