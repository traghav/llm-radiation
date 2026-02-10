#!/usr/bin/env python3
"""Run experiments across multiple models defined in a sweep config."""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from llm_radiation.experiments.config import ExperimentConfig
from llm_radiation.experiments.runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Run full experiment sweep across models")
    parser.add_argument("sweep_config", type=str, help="Path to sweep YAML config")
    parser.add_argument("--models-dir", type=str, default="configs/models", help="Directory containing model configs")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load sweep config as base template
    base_config = ExperimentConfig.from_yaml(args.sweep_config)

    # Find all model configs
    models_dir = Path(args.models_dir)
    model_configs = sorted(models_dir.glob("*.yaml"))

    logger.info(f"Found {len(model_configs)} model configs in {models_dir}")

    all_results = {}
    for model_path in model_configs:
        with open(model_path) as f:
            model_data = yaml.safe_load(f)

        model_name = model_path.stem
        logger.info(f"\n{'='*60}\nRunning sweep for model: {model_name}\n{'='*60}")

        # Override model config in base
        config = base_config.model_copy(deep=True)
        config.model = config.model.model_copy(update=model_data)
        config.name = f"sweep_{model_name}"
        config.output_dir = args.output_dir

        runner = ExperimentRunner(config)
        results = runner.run()
        all_results[model_name] = results

    logger.info(f"\nSweep complete: {len(all_results)} models evaluated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
