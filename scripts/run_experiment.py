#!/usr/bin/env python3
"""Main CLI entry point for running a single experiment."""

import argparse
import logging
import sys

from llm_radiation.experiments.config import ExperimentConfig
from llm_radiation.experiments.runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Run LLM radiation experiment")
    parser.add_argument("config", type=str, help="Path to experiment YAML config")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = ExperimentConfig.from_yaml(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir

    runner = ExperimentRunner(config)
    results = runner.run()

    print(f"\nExperiment complete: {len(results)} records collected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
