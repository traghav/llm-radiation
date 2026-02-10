"""Tests for adaptive sampling."""

import pytest

from llm_radiation.experiments.adaptive import AdaptiveSampler
from llm_radiation.experiments.config import AdaptiveConfig


class TestAdaptiveSampler:
    def test_initial_rates_count(self):
        config = AdaptiveConfig(initial_points=25)
        sampler = AdaptiveSampler(config)
        rates = sampler.initial_rates()
        assert len(rates) == 25

    def test_initial_rates_range(self):
        config = AdaptiveConfig(min_rate=1e-8, max_rate=1e-2, initial_points=10)
        sampler = AdaptiveSampler(config)
        rates = sampler.initial_rates()
        assert rates[0] == pytest.approx(1e-8, rel=0.01)
        assert rates[-1] == pytest.approx(1e-2, rel=0.01)

    def test_initial_rates_log_spaced(self):
        config = AdaptiveConfig(initial_points=5)
        sampler = AdaptiveSampler(config)
        rates = sampler.initial_rates()
        # Ratios between consecutive points should be roughly equal
        import math
        ratios = [math.log10(rates[i+1]/rates[i]) for i in range(len(rates)-1)]
        for r in ratios:
            assert r == pytest.approx(ratios[0], rel=0.01)

    def test_densify_with_sharp_transition(self):
        config = AdaptiveConfig(max_recursion=1, gradient_threshold_multiplier=1.5)
        sampler = AdaptiveSampler(config)
        # Simulate results with a single sharp drop at 1e-4 (flat elsewhere)
        results = []
        rates = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        scores = [0.80, 0.80, 0.80, 0.80, 0.20, 0.19, 0.18]
        for rate, score in zip(rates, scores):
            results.append({
                "flip_rate": rate,
                "trial": 0,
                "results": {"hellaswag": {"acc_norm": score}},
            })
        new_rates = sampler.densify(results)
        # Should add points around the 1e-5 to 1e-4 transition
        assert len(new_rates) > 0

    def test_densify_no_transition(self):
        config = AdaptiveConfig(max_recursion=1)
        sampler = AdaptiveSampler(config)
        # Flat curve - no transition
        results = []
        for rate in [1e-6, 1e-5, 1e-4, 1e-3]:
            results.append({
                "flip_rate": rate,
                "trial": 0,
                "results": {"bench": {"acc": 0.80}},
            })
        new_rates = sampler.densify(results)
        assert len(new_rates) == 0
