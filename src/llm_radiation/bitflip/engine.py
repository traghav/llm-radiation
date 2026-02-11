"""BitFlipEngine - orchestrates bit flips across all model parameters."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.random import Generator

from .fp16 import flip_bits_fp16, generate_uniform_flips
from .int32_packed import flip_bits_int32
from .manifest import FlipManifest

logger = logging.getLogger(__name__)


@dataclass
class HealthReport:
    """Post-flip health report for a model."""

    nan_count: int = 0
    inf_count: int = 0
    total_params: int = 0
    nan_layers: dict[str, int] = field(default_factory=dict)
    inf_layers: dict[str, int] = field(default_factory=dict)
    logits_nan: bool = False
    logits_inf: bool = False
    logits_min: float = 0.0
    logits_max: float = 0.0

    @property
    def weights_healthy(self) -> bool:
        return self.nan_count == 0 and self.inf_count == 0

    @property
    def logits_healthy(self) -> bool:
        return not self.logits_nan and not self.logits_inf

    @property
    def numerically_broken(self) -> bool:
        return self.logits_nan or self.logits_inf

    def to_dict(self) -> dict:
        return {
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "total_params": self.total_params,
            "nan_layers": self.nan_layers,
            "inf_layers": self.inf_layers,
            "logits_nan": self.logits_nan,
            "logits_inf": self.logits_inf,
            "logits_min": self.logits_min,
            "logits_max": self.logits_max,
            "weights_healthy": self.weights_healthy,
            "logits_healthy": self.logits_healthy,
            "numerically_broken": self.numerically_broken,
        }


class BitFlipEngine:
    """Applies random bit flips to model parameters and tracks them for reproducibility."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def count_total_bits(
        self,
        model: torch.nn.Module,
        quantization: str = "none",
        gptq_flip_targets: list[str] | None = None,
    ) -> int:
        """Count all flippable bits in the model.

        For FP16: counts all float16 parameter bits (16 per element).
        For GPTQ: counts bits in targeted qweight tensors (32 per int32 element).
        """
        if quantization == "gptq":
            return self._count_gptq_bits(model, gptq_flip_targets)
        return self._count_fp16_bits(model)

    def _count_fp16_bits(self, model: torch.nn.Module) -> int:
        total = 0
        for param in model.parameters():
            if param.dtype == torch.float16:
                total += param.numel() * 16
        return total

    def _count_gptq_bits(
        self, model: torch.nn.Module, targets: list[str] | None = None
    ) -> int:
        if targets is None:
            targets = ["qweight"]
        total = 0
        for module_name, module in model.named_modules():
            for attr_name in targets:
                tensor = getattr(module, attr_name, None)
                if tensor is not None and isinstance(tensor, torch.Tensor):
                    if tensor.dtype == torch.int32:
                        total += tensor.numel() * 32
        return total

    @staticmethod
    def check_health(
        model: torch.nn.Module,
        tokenizer: object | None = None,
        quantization: str = "none",
    ) -> HealthReport:
        """Check model health after bit flips: NaN/Inf in weights and logits."""
        report = HealthReport()

        if quantization == "gptq":
            # For GPTQ: check float scales/biases for NaN/Inf, skip int32 qweights
            for name, param in model.named_parameters():
                if param.dtype in (torch.float16, torch.float32):
                    flat = param.data.float()
                    report.total_params += flat.numel()
                    nans = torch.isnan(flat).sum().item()
                    infs = torch.isinf(flat).sum().item()
                    report.nan_count += nans
                    report.inf_count += infs
                    if nans > 0:
                        report.nan_layers[name] = nans
                    if infs > 0:
                        report.inf_layers[name] = infs
            # Also check scales buffers on GPTQ modules
            for module_name, module in model.named_modules():
                scales = getattr(module, "scales", None)
                if scales is not None and isinstance(scales, torch.Tensor):
                    if scales.dtype in (torch.float16, torch.float32):
                        fqn = f"{module_name}.scales"
                        flat = scales.data.float()
                        nans = torch.isnan(flat).sum().item()
                        infs = torch.isinf(flat).sum().item()
                        if nans > 0:
                            report.nan_layers[fqn] = nans
                            report.nan_count += nans
                        if infs > 0:
                            report.inf_layers[fqn] = infs
                            report.inf_count += infs
        else:
            # FP16: check all float16 parameters
            for name, param in model.named_parameters():
                if param.dtype == torch.float16:
                    flat = param.data.float()
                    report.total_params += flat.numel()
                    nans = torch.isnan(flat).sum().item()
                    infs = torch.isinf(flat).sum().item()
                    report.nan_count += nans
                    report.inf_count += infs
                    if nans > 0:
                        report.nan_layers[name] = nans
                    if infs > 0:
                        report.inf_layers[name] = infs

        # Check logits if tokenizer available (always run regardless of quantization)
        if tokenizer is not None:
            try:
                prompt = "The capital of France is"
                inputs = tokenizer(prompt, return_tensors="pt")
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = model(**inputs).logits
                report.logits_nan = torch.isnan(logits).any().item()
                report.logits_inf = torch.isinf(logits).any().item()
                if not report.logits_nan:
                    report.logits_min = logits.min().item()
                    report.logits_max = logits.max().item()
                else:
                    report.logits_min = float("nan")
                    report.logits_max = float("nan")
            except Exception as e:
                logger.warning(f"Health check logit probe failed: {e}")
                report.logits_nan = True

        return report

    def apply_flips(
        self,
        model: torch.nn.Module,
        flip_rate: float,
        model_id: str = "unknown",
        model_hash: str = "",
        seed: int | None = None,
        quantization: str = "none",
        gptq_flip_targets: list[str] | None = None,
    ) -> FlipManifest:
        """Apply random bit flips to model weights at the given rate.

        Args:
            model: The model to corrupt (modified in-place).
            flip_rate: Fraction of total bits to flip (e.g. 1e-6).
            model_id: Identifier for the model.
            model_hash: Hash of original weights for integrity checking.
            seed: Override the engine's default seed.
            quantization: "none" for FP16 params, "gptq" for int32 qweights.
            gptq_flip_targets: Which GPTQ attrs to flip (default: ["qweight"]).

        Returns:
            FlipManifest recording all flips for reproducibility.
        """
        if quantization == "gptq":
            return self._apply_gptq_flips(
                model, flip_rate, model_id, model_hash, seed, gptq_flip_targets
            )
        return self._apply_fp16_flips(model, flip_rate, model_id, model_hash, seed)

    def _apply_fp16_flips(
        self,
        model: torch.nn.Module,
        flip_rate: float,
        model_id: str,
        model_hash: str,
        seed: int | None,
    ) -> FlipManifest:
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        total_bits = self._count_fp16_bits(model)
        num_flips = max(1, int(total_bits * flip_rate))

        manifest = FlipManifest(
            model_id=model_id,
            model_hash=model_hash,
            seed=seed if seed is not None else self.seed,
            flip_rate=flip_rate,
            total_bits=total_bits,
            num_flips=num_flips,
        )

        # Build flat index across all FP16 parameters
        param_info: list[tuple[str, torch.nn.Parameter, int]] = []
        cumulative = 0
        for name, param in model.named_parameters():
            if param.dtype == torch.float16:
                param_info.append((name, param, cumulative))
                cumulative += param.numel() * 16

        # Generate all flip locations globally
        total_elements = cumulative // 16
        global_element_indices, global_bit_positions = generate_uniform_flips(
            total_elements, num_flips, rng
        )

        # Distribute flips to their respective parameters
        cumulative_elements = 0
        for name, param, _ in param_info:
            n = param.numel()
            mask = (global_element_indices >= cumulative_elements) & (
                global_element_indices < cumulative_elements + n
            )
            if mask.any():
                local_elements = global_element_indices[mask] - cumulative_elements
                local_bits = global_bit_positions[mask]

                if not param.data.is_contiguous():
                    param.data = param.data.contiguous()

                flip_bits_fp16(param.data, local_elements, local_bits)
                manifest.add_flips_bulk(name, local_elements, local_bits)

            cumulative_elements += n

        return manifest

    def _apply_gptq_flips(
        self,
        model: torch.nn.Module,
        flip_rate: float,
        model_id: str,
        model_hash: str,
        seed: int | None,
        targets: list[str] | None,
    ) -> FlipManifest:
        if targets is None:
            targets = ["qweight"]

        rng = np.random.default_rng(seed if seed is not None else self.seed)
        total_bits = self._count_gptq_bits(model, targets)
        num_flips = max(1, int(total_bits * flip_rate))

        manifest = FlipManifest(
            model_id=model_id,
            model_hash=model_hash,
            seed=seed if seed is not None else self.seed,
            flip_rate=flip_rate,
            total_bits=total_bits,
            num_flips=num_flips,
        )

        # Build flat index across all targeted GPTQ tensors
        tensor_info: list[tuple[str, torch.Tensor]] = []
        cumulative = 0
        for module_name, module in model.named_modules():
            for attr_name in targets:
                tensor = getattr(module, attr_name, None)
                if tensor is not None and isinstance(tensor, torch.Tensor):
                    if tensor.dtype == torch.int32:
                        fqn = f"{module_name}.{attr_name}" if module_name else attr_name
                        tensor_info.append((fqn, tensor))
                        cumulative += tensor.numel() * 32

        # Generate all flip locations globally (32 bits per int32 element)
        total_elements = cumulative // 32
        global_element_indices, global_bit_positions = generate_uniform_flips(
            total_elements, num_flips, rng, bits_per_element=32
        )

        # Distribute flips to their respective tensors
        cumulative_elements = 0
        for fqn, tensor in tensor_info:
            n = tensor.numel()
            mask = (global_element_indices >= cumulative_elements) & (
                global_element_indices < cumulative_elements + n
            )
            if mask.any():
                local_elements = global_element_indices[mask] - cumulative_elements
                local_bits = global_bit_positions[mask]

                if not tensor.data.is_contiguous():
                    tensor.data = tensor.data.contiguous()

                flip_bits_int32(tensor.data, local_elements, local_bits)
                manifest.add_flips_bulk(fqn, local_elements, local_bits)

            cumulative_elements += n

        return manifest
