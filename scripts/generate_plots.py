#!/usr/bin/env python3
"""Generate publication-quality figures from experiment results."""

import argparse
import json
import logging
import sys
from pathlib import Path

from llm_radiation.analysis.plots import (
    plot_degradation_curves,
    plot_model_size_vs_robustness,
    plot_architecture_comparison,
    plot_control_comparison,
)
from llm_radiation.analysis.phase_detection import detect_phase_transition


MODEL_SIZES = {
    "tinyllama": 1.1e9,
    "gemma_2b": 2e9,
    "phi3_mini": 3.8e9,
    "mistral_7b": 7.3e9,
    "gemma_7b": 8.5e9,
    "llama3_8b": 8e9,
    "llama3_13b": 13e9,
    "mixtral_8x7b": 46.7e9,
}


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("results_dir", type=str, help="Directory containing per-model results")
    parser.add_argument("--output-dir", default="paper/figures", help="Output directory for figures")
    parser.add_argument("--benchmark", default="hellaswag", help="Primary benchmark for plots")
    parser.add_argument("--metric", default="acc_norm", help="Primary metric name")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all results
    results_by_model = {}
    for result_file in sorted(results_dir.glob("*/results.json")):
        model_name = result_file.parent.name.replace("sweep_", "")
        with open(result_file) as f:
            results_by_model[model_name] = json.load(f)
        logger.info(f"Loaded {model_name}: {len(results_by_model[model_name])} records")

    if not results_by_model:
        logger.error(f"No results found in {results_dir}")
        return 1

    # 1. Degradation curves
    fig = plot_degradation_curves(
        results_by_model, args.benchmark, args.metric,
        output_path=str(output_dir / f"degradation_curves.{args.format}"),
    )
    logger.info("Generated degradation curves")

    # 2. Model size vs robustness
    critical_rates = {}
    for model_name, results in results_by_model.items():
        try:
            fit = detect_phase_transition(results, args.benchmark, args.metric)
            critical_rates[model_name] = fit["r_c"]
        except Exception as e:
            logger.warning(f"Could not fit {model_name}: {e}")

    if critical_rates:
        sizes = {k: v for k, v in MODEL_SIZES.items() if k in critical_rates}
        fig = plot_model_size_vs_robustness(
            sizes, critical_rates,
            output_path=str(output_dir / f"size_vs_robustness.{args.format}"),
        )
        logger.info("Generated size vs robustness plot")

    # 3. Architecture comparison (dense vs MoE)
    dense_models = {k: v for k, v in results_by_model.items() if "mixtral" not in k}
    moe_models = {k: v for k, v in results_by_model.items() if "mixtral" in k}
    if dense_models and moe_models:
        fig = plot_architecture_comparison(
            dense_models, moe_models, args.benchmark, args.metric,
            output_path=str(output_dir / f"dense_vs_moe.{args.format}"),
        )
        logger.info("Generated architecture comparison plot")

    logger.info(f"All figures saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
