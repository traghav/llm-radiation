"""Bridge to EleutherAI's lm-evaluation-harness for LLM benchmarking.

This module provides a clean interface to run standard LLM benchmarks (MMLU, HellaSwag,
ARC, etc.) using the lm-evaluation-harness library.
"""

from typing import Any

import lm_eval
from lm_eval.models.huggingface import HFLM


def run_lm_eval(
    model: Any,
    tokenizer: Any,
    tasks: list[str],
    num_fewshot: int | None = None,
    limit: int | None = None,
    batch_size: int | str = "auto",
) -> dict[str, dict[str, float]]:
    """Run lm-evaluation-harness benchmarks on a model.

    Args:
        model: HuggingFace model (AutoModelForCausalLM)
        tokenizer: HuggingFace tokenizer
        tasks: List of task names (e.g., ["mmlu", "hellaswag", "arc_easy"])
        num_fewshot: Number of few-shot examples (default: None, uses task default)
        limit: Limit number of samples per task (default: None, uses all)
        batch_size: Batch size for evaluation (default: "auto")

    Returns:
        Dictionary mapping task names to dictionaries of metrics and scores.
        Example:
            {
                "mmlu": {"acc": 0.45, "acc_stderr": 0.02},
                "hellaswag": {"acc": 0.52, "acc_norm": 0.54}
            }

    Examples:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> results = run_lm_eval(model, tokenizer, ["hellaswag"], limit=100)
        >>> print(f"HellaSwag accuracy: {results['hellaswag']['acc']:.3f}")
    """
    # Wrap model in lm-eval's HFLM wrapper
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

    # Build task manager arguments
    task_manager_args = {}
    if num_fewshot is not None:
        task_manager_args["num_fewshot"] = num_fewshot

    # Run evaluation
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        limit=limit,
        **task_manager_args,
    )

    # Parse and flatten results
    parsed_results = {}

    # Results structure: results["results"] contains per-task metrics
    if "results" in results:
        for task_name, task_results in results["results"].items():
            # Extract metrics (skip metadata like alias, task_name)
            metrics = {}
            for key, value in task_results.items():
                if key not in ["alias", "task_name"] and isinstance(value, (int, float)):
                    metrics[key] = value

            parsed_results[task_name] = metrics

    return parsed_results
