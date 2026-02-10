"""Plotting utilities for visualizing bit flip degradation results."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .phase_detection import detect_phase_transition

# Set publication-quality style
sns.set_context("paper", font_scale=1.2)
plt.rcParams.update({
    "figure.figsize": (8, 6),
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 100
})


def plot_degradation_curves(
    results_by_model: dict[str, list[dict[str, Any]]],
    benchmark_name: str,
    metric_name: str,
    output_path: str | None = None
) -> plt.Figure:
    """Plot degradation curves for multiple models.

    Args:
        results_by_model: Dictionary mapping model names to lists of result dicts
        benchmark_name: Name of the benchmark being analyzed
        metric_name: Name of the metric to plot (e.g., "accuracy")
        output_path: Optional path to save the figure

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = sns.color_palette("husl", len(results_by_model))

    for (model_name, results), color in zip(results_by_model.items(), colors):
        try:
            # Detect phase transition and get aggregated data
            transition_data = detect_phase_transition(results, benchmark_name, metric_name)

            flip_rates = np.array(transition_data["flip_rates"])
            mean_scores = np.array(transition_data["mean_scores"])
            std_scores = np.array(transition_data["std_scores"])
            r_c = transition_data["fit_results"]["r_c"]

            # Plot the curve with error bars
            ax.errorbar(
                flip_rates,
                mean_scores,
                yerr=std_scores,
                marker="o",
                label=model_name,
                color=color,
                capsize=3,
                alpha=0.7
            )

            # Mark critical flip rate with vertical dashed line
            ax.axvline(
                r_c,
                color=color,
                linestyle="--",
                alpha=0.5,
                linewidth=1,
                label=f"{model_name} $r_c$={r_c:.2e}"
            )

        except Exception as e:
            print(f"Warning: Could not plot {model_name}: {e}")
            continue

    ax.set_xscale("log")
    ax.set_xlabel("Flip Rate (per parameter)")
    ax.set_ylabel(f"{metric_name.capitalize()}")
    ax.set_title(f"Performance Degradation: {benchmark_name}")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_model_size_vs_robustness(
    model_sizes: dict[str, int],
    critical_rates: dict[str, float],
    output_path: str | None = None
) -> plt.Figure:
    """Plot model size vs robustness (critical flip rate).

    Args:
        model_sizes: Dictionary mapping model names to parameter counts (in billions)
        critical_rates: Dictionary mapping model names to critical flip rates
        output_path: Optional path to save the figure

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Extract common models
    common_models = set(model_sizes.keys()) & set(critical_rates.keys())

    if not common_models:
        raise ValueError("No common models found between sizes and critical rates")

    model_names = sorted(common_models, key=lambda m: model_sizes[m])
    sizes = [model_sizes[m] for m in model_names]
    rates = [critical_rates[m] for m in model_names]

    # Scatter plot
    ax.scatter(sizes, rates, s=100, alpha=0.6, c=range(len(sizes)), cmap="viridis")

    # Label each point with model name
    for name, size, rate in zip(model_names, sizes, rates):
        ax.annotate(
            name,
            (size, rate),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            alpha=0.8
        )

    ax.set_xlabel("Model Size (Billion Parameters)")
    ax.set_ylabel("Critical Flip Rate $r_c$")
    ax.set_title("Model Size vs Robustness to Bit Flips")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_architecture_comparison(
    dense_results: dict[str, list[dict[str, Any]]],
    moe_results: dict[str, list[dict[str, Any]]],
    benchmark_name: str,
    metric_name: str,
    output_path: str | None = None
) -> plt.Figure:
    """Compare dense vs MoE architecture degradation curves.

    Args:
        dense_results: Dictionary mapping dense model names to result lists
        moe_results: Dictionary mapping MoE model names to result lists
        benchmark_name: Name of the benchmark being analyzed
        metric_name: Name of the metric to plot
        output_path: Optional path to save the figure

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    dense_color = sns.color_palette("Set2")[0]
    moe_color = sns.color_palette("Set2")[1]

    # Plot dense models
    for model_name, results in dense_results.items():
        try:
            transition_data = detect_phase_transition(results, benchmark_name, metric_name)

            flip_rates = np.array(transition_data["flip_rates"])
            mean_scores = np.array(transition_data["mean_scores"])
            std_scores = np.array(transition_data["std_scores"])
            r_c = transition_data["fit_results"]["r_c"]

            ax.errorbar(
                flip_rates,
                mean_scores,
                yerr=std_scores,
                marker="o",
                label=f"{model_name} (Dense)",
                color=dense_color,
                capsize=3,
                alpha=0.7
            )

            ax.axvline(
                r_c,
                color=dense_color,
                linestyle="--",
                alpha=0.5,
                linewidth=1
            )

        except Exception as e:
            print(f"Warning: Could not plot dense model {model_name}: {e}")

    # Plot MoE models
    for model_name, results in moe_results.items():
        try:
            transition_data = detect_phase_transition(results, benchmark_name, metric_name)

            flip_rates = np.array(transition_data["flip_rates"])
            mean_scores = np.array(transition_data["mean_scores"])
            std_scores = np.array(transition_data["std_scores"])
            r_c = transition_data["fit_results"]["r_c"]

            ax.errorbar(
                flip_rates,
                mean_scores,
                yerr=std_scores,
                marker="s",
                label=f"{model_name} (MoE)",
                color=moe_color,
                capsize=3,
                alpha=0.7
            )

            ax.axvline(
                r_c,
                color=moe_color,
                linestyle="--",
                alpha=0.5,
                linewidth=1
            )

        except Exception as e:
            print(f"Warning: Could not plot MoE model {model_name}: {e}")

    ax.set_xscale("log")
    ax.set_xlabel("Flip Rate (per parameter)")
    ax.set_ylabel(f"{metric_name.capitalize()}")
    ax.set_title(f"Dense vs MoE Architecture: {benchmark_name}")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_control_comparison(
    llm_results: dict[str, list[dict[str, Any]]],
    control_results: dict[str, list[dict[str, Any]]],
    benchmark_name: str,
    output_path: str | None = None
) -> plt.Figure:
    """Compare LLM vs control model degradation curves.

    Args:
        llm_results: Dictionary mapping LLM names to result lists
        control_results: Dictionary mapping control model names (e.g., "logistic_regression",
            "cnn", "bert_tiny") to result lists
        benchmark_name: Name of the benchmark being analyzed
        output_path: Optional path to save the figure

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use different color palettes for LLMs and controls
    llm_colors = sns.color_palette("Blues_d", len(llm_results))
    control_colors = sns.color_palette("Oranges_d", len(control_results))

    # Infer metric name from first available result
    metric_name = "accuracy"  # Default
    for results in list(llm_results.values()) + list(control_results.values()):
        if results:
            for key in results[0].keys():
                if key not in ["flip_rate", "benchmark", "benchmark_name", "model"]:
                    metric_name = key
                    break
            break

    # Plot LLM models
    for (model_name, results), color in zip(llm_results.items(), llm_colors):
        try:
            transition_data = detect_phase_transition(results, benchmark_name, metric_name)

            flip_rates = np.array(transition_data["flip_rates"])
            mean_scores = np.array(transition_data["mean_scores"])
            std_scores = np.array(transition_data["std_scores"])
            r_c = transition_data["fit_results"]["r_c"]

            ax.errorbar(
                flip_rates,
                mean_scores,
                yerr=std_scores,
                marker="o",
                label=f"{model_name} (LLM)",
                color=color,
                capsize=3,
                alpha=0.7,
                linewidth=2
            )

            ax.axvline(
                r_c,
                color=color,
                linestyle="--",
                alpha=0.5,
                linewidth=1
            )

        except Exception as e:
            print(f"Warning: Could not plot LLM {model_name}: {e}")

    # Plot control models
    for (model_name, results), color in zip(control_results.items(), control_colors):
        try:
            transition_data = detect_phase_transition(results, benchmark_name, metric_name)

            flip_rates = np.array(transition_data["flip_rates"])
            mean_scores = np.array(transition_data["mean_scores"])
            std_scores = np.array(transition_data["std_scores"])
            r_c = transition_data["fit_results"]["r_c"]

            ax.errorbar(
                flip_rates,
                mean_scores,
                yerr=std_scores,
                marker="^",
                label=f"{model_name} (Control)",
                color=color,
                capsize=3,
                alpha=0.7,
                linewidth=2
            )

            ax.axvline(
                r_c,
                color=color,
                linestyle=":",
                alpha=0.5,
                linewidth=1
            )

        except Exception as e:
            print(f"Warning: Could not plot control model {model_name}: {e}")

    ax.set_xscale("log")
    ax.set_xlabel("Flip Rate (per parameter)")
    ax.set_ylabel(f"{metric_name.capitalize()}")
    ax.set_title(f"LLM vs Control Models: {benchmark_name}")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig
