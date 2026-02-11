"""Model registry and loading utilities for LLM radiation experiments.

This module provides a central registry of supported models with their HuggingFace IDs,
along with utilities for loading models, managing weights, and performing dtype conversions.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

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
    # GPTQ 4-bit quantized models
    "tinyllama-gptq": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ",
    "gemma-2b-gptq": "TheBloke/gemma-2b-GPTQ",
    "gemma-7b-gptq": "TheBloke/gemma-7b-GPTQ",
    "mistral-7b-gptq": "TheBloke/Mistral-7B-v0.1-GPTQ",
}


def load_model(
    model_id: str,
    dtype: str = "float16",
    device_map: str = "auto",
    revision: str = "main",
    trust_remote_code: bool = False,
    quantization: str = "none",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal language model and tokenizer from HuggingFace.

    Args:
        model_id: HuggingFace model ID or short name from MODEL_REGISTRY
        dtype: Data type for model weights (default: "float16")
        device_map: Device mapping strategy (default: "auto")
        revision: Model revision/branch (default: "main")
        trust_remote_code: Allow execution of remote code (default: False)
        quantization: Quantization method - "none" or "gptq" (default: "none")

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ValueError: If model_id is not found in registry and doesn't appear to be a valid HF ID
        RuntimeError: If model loading fails
    """
    # Resolve short name from registry
    resolved_id = MODEL_REGISTRY.get(model_id, model_id)

    # Convert dtype string to torch dtype
    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

    try:
        kwargs: dict[str, Any] = {
            "device_map": device_map,
            "revision": revision,
            "trust_remote_code": trust_remote_code,
        }

        if quantization == "gptq":
            # Pre-quantized GPTQ models embed their config; just set torch_dtype
            # for the non-quantized layers (scales, embeddings, lm_head)
            kwargs["torch_dtype"] = torch_dtype
            logger.info(f"Loading GPTQ 4-bit model: {resolved_id}")
        else:
            kwargs["torch_dtype"] = torch_dtype

        model = AutoModelForCausalLM.from_pretrained(resolved_id, **kwargs)

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


def _iter_gptq_tensors(
    model: torch.nn.Module,
    targets: list[str] | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """Iterate over GPTQ quantized tensors (qweight, qzeros, scales, g_idx).

    Args:
        model: Model with GPTQ QuantLinear layers.
        targets: Which tensor attrs to include (default: ["qweight"]).

    Returns:
        List of (fully_qualified_name, tensor) tuples.
    """
    if targets is None:
        targets = ["qweight"]
    result = []
    for module_name, module in model.named_modules():
        for attr_name in targets:
            tensor = getattr(module, attr_name, None)
            if tensor is not None and isinstance(tensor, torch.Tensor):
                fqn = f"{module_name}.{attr_name}" if module_name else attr_name
                result.append((fqn, tensor))
    return result


def snapshot_weights(
    model: torch.nn.Module,
    quantization: str = "none",
    gptq_flip_targets: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Create a snapshot of all model parameters (and GPTQ buffers if applicable).

    Creates deep copies of all parameters moved to CPU for safe storage.
    Useful for saving model state before bit-flip injection.

    Args:
        model: PyTorch model to snapshot
        quantization: "none" or "gptq"
        gptq_flip_targets: Which GPTQ attrs to snapshot (default: ["qweight"])

    Returns:
        Dictionary mapping parameter/buffer names to cloned CPU tensors
    """
    snapshot = {}
    for name, param in model.named_parameters():
        snapshot[name] = param.data.clone().cpu()

    if quantization == "gptq":
        # Also snapshot GPTQ-specific buffers that may not be parameters
        gptq_attrs = gptq_flip_targets or ["qweight"]
        # Include scales/qzeros/g_idx for full restore regardless of flip targets
        all_attrs = list(set(gptq_attrs) | {"qweight", "qzeros", "scales", "g_idx"})
        for fqn, tensor in _iter_gptq_tensors(model, all_attrs):
            if fqn not in snapshot:
                snapshot[fqn] = tensor.data.clone().cpu()

    return snapshot


def restore_weights(
    model: torch.nn.Module,
    snapshot: dict[str, torch.Tensor],
    quantization: str = "none",
) -> None:
    """Restore model weights from a snapshot.

    Restores parameters from a previously created snapshot, moving tensors
    to the appropriate device and dtype as needed.

    Args:
        model: PyTorch model to restore
        snapshot: Dictionary of parameter names to tensors (from snapshot_weights)
        quantization: "none" or "gptq"

    Raises:
        ValueError: If snapshot keys don't match model parameters (FP16 mode only)

    Note:
        This modifies the model in-place.
    """
    if quantization == "gptq":
        # Restore parameters
        for name, param in model.named_parameters():
            if name in snapshot:
                restored = snapshot[name].to(device=param.device, dtype=param.dtype)
                param.data.copy_(restored)

        # Restore GPTQ buffers
        all_attrs = ["qweight", "qzeros", "scales", "g_idx"]
        for fqn, tensor in _iter_gptq_tensors(model, all_attrs):
            if fqn in snapshot:
                restored = snapshot[fqn].to(device=tensor.device, dtype=tensor.dtype)
                tensor.data.copy_(restored)
    else:
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
            restored = snapshot[name].to(device=param.device, dtype=param.dtype)
            param.data.copy_(restored)
