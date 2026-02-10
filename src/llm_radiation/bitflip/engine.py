"""BitFlipEngine - orchestrates bit flips across all model parameters."""

from __future__ import annotations

import numpy as np
import torch
from numpy.random import Generator

from .fp16 import flip_bits_fp16, generate_uniform_flips
from .manifest import FlipManifest


class BitFlipEngine:
    """Applies random bit flips to model parameters and tracks them for reproducibility."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def count_total_bits(self, model: torch.nn.Module) -> int:
        """Count all FP16-representable parameter bits in the model."""
        total = 0
        for param in model.parameters():
            if param.dtype == torch.float16:
                total += param.numel() * 16
        return total

    def apply_flips(
        self,
        model: torch.nn.Module,
        flip_rate: float,
        model_id: str = "unknown",
        model_hash: str = "",
        seed: int | None = None,
    ) -> FlipManifest:
        """Apply random bit flips to all FP16 parameters at the given rate.

        Args:
            model: The model to corrupt (modified in-place).
            flip_rate: Fraction of total bits to flip (e.g. 1e-6).
            model_id: Identifier for the model.
            model_hash: Hash of original weights for integrity checking.
            seed: Override the engine's default seed.

        Returns:
            FlipManifest recording all flips for reproducibility.
        """
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        total_bits = self.count_total_bits(model)
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
