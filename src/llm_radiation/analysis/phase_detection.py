"""Phase transition detection using sigmoid fitting for bit flip degradation analysis."""

from typing import Any

import numpy as np
from scipy.optimize import curve_fit


def sigmoid_model(log_r: float | np.ndarray, baseline: float, k: float, log_rc: float) -> float | np.ndarray:
    """Sigmoid model for performance degradation.

    Models performance as: baseline / (1 + exp(k * (log_r - log_rc)))

    Args:
        log_r: Log10 of flip rate(s)
        baseline: Performance at zero flip rate
        k: Steepness of the transition
        log_rc: Log10 of critical flip rate (inflection point)

    Returns:
        Predicted performance score(s)
    """
    return baseline / (1 + np.exp(k * (log_r - log_rc)))


def fit_sigmoid(flip_rates: list[float], scores: list[float]) -> dict[str, Any]:
    """Fit a sigmoid model to degradation data.

    Args:
        flip_rates: List of bit flip rates
        scores: List of corresponding benchmark scores

    Returns:
        Dictionary containing:
            - baseline: Fitted baseline performance
            - k: Steepness parameter
            - r_c: Critical flip rate (inflection point, in linear scale)
            - log_rc: Log10 of critical flip rate
            - r_squared: Goodness of fit (R² value)
            - params_cov: Covariance matrix of fitted parameters

    Raises:
        ValueError: If fitting fails or data is invalid
    """
    flip_rates_arr = np.array(flip_rates)
    scores_arr = np.array(scores)

    if len(flip_rates_arr) < 3:
        raise ValueError("Need at least 3 data points to fit sigmoid")

    if np.any(flip_rates_arr <= 0):
        raise ValueError("All flip rates must be positive for log transform")

    # Convert to log scale for fitting
    log_rates = np.log10(flip_rates_arr)

    # Initial parameter guesses
    baseline_guess = np.max(scores_arr)
    k_guess = 1.0
    log_rc_guess = np.mean(log_rates)  # Midpoint of log rates

    initial_guess = [baseline_guess, k_guess, log_rc_guess]

    try:
        # Fit the sigmoid model
        params, params_cov = curve_fit(
            sigmoid_model,
            log_rates,
            scores_arr,
            p0=initial_guess,
            maxfev=10000
        )

        baseline, k, log_rc = params

        # Calculate R²
        predictions = sigmoid_model(log_rates, baseline, k, log_rc)
        ss_res = np.sum((scores_arr - predictions) ** 2)
        ss_tot = np.sum((scores_arr - np.mean(scores_arr)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Convert critical rate back to linear scale
        r_c = 10 ** log_rc

        return {
            "baseline": float(baseline),
            "k": float(k),
            "r_c": float(r_c),
            "log_rc": float(log_rc),
            "r_squared": float(r_squared),
            "params_cov": params_cov.tolist()
        }

    except Exception as e:
        raise ValueError(f"Failed to fit sigmoid model: {e}")


def detect_phase_transition(
    results: list[dict[str, Any]],
    benchmark_name: str,
    metric_name: str
) -> dict[str, Any]:
    """Detect phase transition in benchmark results.

    Args:
        results: List of result dictionaries, each containing:
            - flip_rate: Bit flip rate
            - benchmark_name: Name of benchmark
            - metric_name: Performance metric value
        benchmark_name: Name of benchmark to analyze
        metric_name: Name of metric to analyze (e.g., "accuracy")

    Returns:
        Dictionary containing:
            - fit_results: Sigmoid fit parameters
            - critical_region: Dictionary with flip rates where score drops
              from 90% to 10% of baseline
            - flip_rates: Array of flip rates used
            - mean_scores: Array of mean scores at each flip rate
            - std_scores: Array of standard deviations at each flip rate

    Raises:
        ValueError: If no matching results found or fitting fails
    """
    # Filter results for the specified benchmark
    filtered = [
        r for r in results
        if r.get("benchmark") == benchmark_name or r.get("benchmark_name") == benchmark_name
    ]

    if not filtered:
        raise ValueError(f"No results found for benchmark: {benchmark_name}")

    # Group by flip rate and compute statistics
    rate_groups: dict[float, list[float]] = {}

    for result in filtered:
        flip_rate = result.get("flip_rate")
        score = result.get(metric_name)

        if flip_rate is None or score is None:
            continue

        if flip_rate not in rate_groups:
            rate_groups[flip_rate] = []
        rate_groups[flip_rate].append(score)

    if not rate_groups:
        raise ValueError(f"No valid data found for metric: {metric_name}")

    # Sort by flip rate and compute mean/std
    sorted_rates = sorted(rate_groups.keys())
    flip_rates = np.array(sorted_rates)
    mean_scores = np.array([np.mean(rate_groups[r]) for r in sorted_rates])
    std_scores = np.array([np.std(rate_groups[r]) for r in sorted_rates])

    # Fit sigmoid model
    fit_results = fit_sigmoid(flip_rates.tolist(), mean_scores.tolist())

    # Identify critical region (90% to 10% of baseline)
    baseline = fit_results["baseline"]
    k = fit_results["k"]
    log_rc = fit_results["log_rc"]

    # Solve for flip rates at 90% and 10% of baseline
    # baseline / (1 + exp(k * (log_r - log_rc))) = threshold * baseline
    # 1 / (1 + exp(k * (log_r - log_rc))) = threshold
    # 1 + exp(k * (log_r - log_rc)) = 1 / threshold
    # exp(k * (log_r - log_rc)) = (1 / threshold) - 1
    # k * (log_r - log_rc) = ln((1 / threshold) - 1)
    # log_r = log_rc + ln((1 / threshold) - 1) / k

    threshold_90 = 0.9
    threshold_10 = 0.1

    if k != 0:
        log_r_90 = log_rc + np.log((1 / threshold_90) - 1) / k
        log_r_10 = log_rc + np.log((1 / threshold_10) - 1) / k

        r_90 = 10 ** log_r_90
        r_10 = 10 ** log_r_10

        critical_region = {
            "start": float(min(r_90, r_10)),
            "end": float(max(r_90, r_10)),
            "width": float(abs(r_10 - r_90))
        }
    else:
        critical_region = {
            "start": float(fit_results["r_c"]),
            "end": float(fit_results["r_c"]),
            "width": 0.0
        }

    return {
        "fit_results": fit_results,
        "critical_region": critical_region,
        "flip_rates": flip_rates.tolist(),
        "mean_scores": mean_scores.tolist(),
        "std_scores": std_scores.tolist()
    }
