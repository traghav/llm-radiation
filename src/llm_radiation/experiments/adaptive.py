"""Adaptive sampling to densify around phase transitions."""

from __future__ import annotations

import math
import numpy as np

from llm_radiation.experiments.config import AdaptiveConfig


class AdaptiveSampler:
    """Generates flip rates and adaptively densifies around phase transitions."""

    def __init__(self, config: AdaptiveConfig):
        self.config = config

    def initial_rates(self) -> list[float]:
        """Generate initial log-spaced flip rates."""
        return np.logspace(
            math.log10(self.config.min_rate),
            math.log10(self.config.max_rate),
            self.config.initial_points,
        ).tolist()

    def densify(
        self,
        results: list[dict],
        recursion_level: int = 0,
    ) -> list[float]:
        """Identify phase transition regions and add midpoints.

        Looks at the normalized gradient of the primary metric across
        consecutive flip rates on a log scale. Where the gradient exceeds
        threshold_multiplier * median, we insert geometric-mean midpoints.
        """
        if recursion_level >= self.config.max_recursion:
            return []

        # Extract (rate, mean_score) pairs, averaging over trials
        rate_scores: dict[float, list[float]] = {}
        for r in results:
            rate = r["flip_rate"]
            if rate == 0.0:
                continue
            # Use the first benchmark's first metric as the primary signal
            for bench_results in r["results"].values():
                score = next(iter(bench_results.values()), 0.0)
                rate_scores.setdefault(rate, []).append(score)
                break

        if len(rate_scores) < 3:
            return []

        sorted_rates = sorted(rate_scores.keys())
        mean_scores = [np.mean(rate_scores[r]) for r in sorted_rates]

        # Compute normalized gradient on log scale
        gradients = []
        for i in range(1, len(sorted_rates)):
            log_dr = math.log10(sorted_rates[i]) - math.log10(sorted_rates[i - 1])
            if log_dr == 0:
                gradients.append(0.0)
                continue
            ds = abs(mean_scores[i] - mean_scores[i - 1])
            baseline = max(abs(mean_scores[i]), abs(mean_scores[i - 1]), 1e-10)
            gradients.append((ds / baseline) / log_dr)

        if not gradients:
            return []

        median_grad = float(np.median(gradients))
        threshold = self.config.gradient_threshold_multiplier * median_grad

        # Insert geometric midpoints where gradient exceeds threshold
        new_rates = []
        for i, g in enumerate(gradients):
            if g > threshold:
                midpoint = math.sqrt(sorted_rates[i] * sorted_rates[i + 1])
                if midpoint not in sorted_rates:
                    new_rates.append(midpoint)

        return new_rates
