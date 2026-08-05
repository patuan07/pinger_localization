"""
Shared metric computation utilities for pinger localization evaluation.

Functions here are used by both the live eval node and the offline analysis script.
"""

import math
from typing import Dict, List, Tuple

import numpy as np


def position_error(north_est: float, east_est: float,
                   north_gt: float, east_gt: float) -> float:
    """Euclidean distance between estimated and true pinger position (meters)."""
    return math.hypot(north_est - north_gt, east_est - east_gt)


def rmse(errors: List[float]) -> float:
    """Root mean squared error from a list of error values."""
    if not errors:
        return float("nan")
    return math.sqrt(np.mean(np.array(errors) ** 2))


def convergence_time(timestamps: List[float], errors: List[float],
                     threshold: float = 1.0, hold_count: int = 3) -> float:
    """Time (seconds) until error drops below `threshold` and stays below for
    `hold_count` consecutive pings.

    Returns:
        Timestamp of convergence, or float("inf") if never converged.
    """
    consecutive = 0
    for i, (t, err) in enumerate(zip(timestamps, errors)):
        if err < threshold:
            consecutive += 1
            if consecutive >= hold_count:
                return t
        else:
            consecutive = 0
    return float("inf")


def final_error(errors: List[float], window: int = 5) -> float:
    """Mean error over the last `window` measurements (reduces noise)."""
    if not errors:
        return float("nan")
    w = min(window, len(errors))
    return float(np.mean(errors[-w:]))


def solver_summary(timestamps: List[float], errors: List[float],
                   solver_name: str = "") -> Dict:
    """Compute a summary of solver performance.

    Returns a dictionary with:
        - rmse: overall RMSE (m)
        - final_error: mean error in last 5 pings (m)
        - convergence_1m: time to converge to <1m (seconds)
        - convergence_0_5m: time to converge to <0.5m (seconds)
        - n_pings: total number of pings
    """
    return {
        "solver": solver_name,
        "rmse": rmse(errors),
        "final_error": final_error(errors),
        "convergence_1m": convergence_time(timestamps, errors, threshold=1.0),
        "convergence_0_5m": convergence_time(timestamps, errors, threshold=0.5),
        "n_pings": len(errors),
    }
