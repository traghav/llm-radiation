"""Benchmark harness for running multiple evaluations.

This module provides a unified interface for running multiple benchmarks
(lm-eval tasks, perplexity, etc.) on language models.
"""

from typing import Any

from llm_radiation.benchmarks.lm_eval_bridge import run_lm_eval
from llm_radiation.benchmarks.perplexity import evaluate_perplexity
from llm_radiation.experiments.config import BenchmarkConfig


class BenchmarkRunner:
    """Orchestrates running multiple benchmarks on a model.

    This class manages the execution of various benchmark types including:
    - Standard lm-eval benchmarks (MMLU, HellaSwag, ARC, etc.)
    - Perplexity evaluation on text corpora
    - Custom evaluation tasks
    """

    def __init__(self, benchmarks: list[BenchmarkConfig]):
        """Initialize benchmark runner with configurations.

        Args:
            benchmarks: List of BenchmarkConfig objects defining benchmarks to run

        Examples:
            >>> from llm_radiation.experiments.config import BenchmarkConfig
            >>> configs = [
            ...     BenchmarkConfig(name="mmlu", tasks=["mmlu"], num_fewshot=5),
            ...     BenchmarkConfig(name="hellaswag", tasks=["hellaswag"], limit=100),
            ... ]
            >>> runner = BenchmarkRunner(configs)
        """
        self.benchmarks = benchmarks

    def run_all(
        self,
        model: Any,
        tokenizer: Any,
    ) -> dict[str, dict[str, float]]:
        """Run all configured benchmarks on a model.

        Args:
            model: HuggingFace model (AutoModelForCausalLM)
            tokenizer: HuggingFace tokenizer

        Returns:
            Dictionary mapping benchmark names to metric dictionaries.
            Example:
                {
                    "mmlu": {"acc": 0.45, "acc_stderr": 0.02},
                    "hellaswag": {"acc": 0.52, "acc_norm": 0.54},
                    "perplexity_wikitext": {"perplexity": 28.5}
                }

        Examples:
            >>> from transformers import AutoModelForCausalLM, AutoTokenizer
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> results = runner.run_all(model, tokenizer)
            >>> print(f"MMLU accuracy: {results['mmlu']['acc']:.3f}")
        """
        all_results = {}

        for benchmark_config in self.benchmarks:
            print(f"\n{'='*60}")
            print(f"Running benchmark: {benchmark_config.name}")
            print(f"{'='*60}\n")

            try:
                # Check if this is a perplexity benchmark
                if benchmark_config.name.startswith("perplexity"):
                    results = self._run_perplexity_benchmark(
                        model, tokenizer, benchmark_config
                    )
                else:
                    # Standard lm-eval benchmark
                    results = self._run_lm_eval_benchmark(
                        model, tokenizer, benchmark_config
                    )

                # Store results
                all_results[benchmark_config.name] = results

                # Print summary
                print(f"\nResults for {benchmark_config.name}:")
                for metric, value in results.items():
                    print(f"  {metric}: {value:.4f}")

            except Exception as e:
                print(f"ERROR running benchmark {benchmark_config.name}: {str(e)}")
                # Store error info
                all_results[benchmark_config.name] = {"error": str(e)}

        return all_results

    def _run_lm_eval_benchmark(
        self,
        model: Any,
        tokenizer: Any,
        config: BenchmarkConfig,
    ) -> dict[str, float]:
        """Run an lm-eval benchmark.

        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            config: Benchmark configuration

        Returns:
            Dictionary of metrics
        """
        results = run_lm_eval(
            model=model,
            tokenizer=tokenizer,
            tasks=config.tasks,
            num_fewshot=config.num_fewshot,
            limit=config.limit,
            batch_size=config.batch_size,
        )

        # Flatten results if multiple tasks
        if len(config.tasks) == 1:
            # Single task - return its metrics directly
            return results[config.tasks[0]]
        else:
            # Multiple tasks - flatten with task prefix
            flattened = {}
            for task_name, metrics in results.items():
                for metric_name, value in metrics.items():
                    flattened[f"{task_name}_{metric_name}"] = value
            return flattened

    def _run_perplexity_benchmark(
        self,
        model: Any,
        tokenizer: Any,
        config: BenchmarkConfig,
    ) -> dict[str, float]:
        """Run a perplexity benchmark.

        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            config: Benchmark configuration

        Returns:
            Dictionary with perplexity score
        """
        # Parse dataset name from benchmark name (e.g., "perplexity_wikitext")
        parts = config.name.split("_")
        if len(parts) > 1:
            dataset_name = "_".join(parts[1:])
        else:
            dataset_name = "wikitext"

        # Map common dataset names to their configs
        dataset_configs = {
            "wikitext": ("wikitext", "wikitext-103-raw-v1"),
            "ptb": ("ptb_text_only", "penn_treebank"),
            "lambada": ("lambada", "default"),
        }

        if dataset_name in dataset_configs:
            ds_name, ds_config = dataset_configs[dataset_name]
        else:
            ds_name = dataset_name
            ds_config = "default"

        # Run perplexity evaluation
        perplexity = evaluate_perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset_name=ds_name,
            dataset_config=ds_config,
            limit=config.limit,
        )

        return {"perplexity": perplexity}

    def get_benchmark_names(self) -> list[str]:
        """Get list of all benchmark names.

        Returns:
            List of benchmark names
        """
        return [b.name for b in self.benchmarks]

    def get_benchmark_count(self) -> int:
        """Get the number of configured benchmarks.

        Returns:
            Number of benchmarks
        """
        return len(self.benchmarks)
