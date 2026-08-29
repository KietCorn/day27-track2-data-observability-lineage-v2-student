"""Anomaly detection starter.

Z-score is deliberately the default baseline. Students should improve `auto`
mode for seasonality/outliers rather than deleting the simple implementation.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def zscore_detector(current: float, history: Iterable[float], threshold: float = 3.0) -> dict[str, Any]:
    values = np.asarray(list(history), dtype=float)
    if values.size < 3:
        return {"is_anomaly": False, "score": 0.0, "method": "zscore", "reason": "insufficient_history"}
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0:
        score = float("inf") if float(current) != mean else 0.0
    else:
        score = abs(float(current) - mean) / std
    return {
        "is_anomaly": bool(score > threshold),
        "score": float(score),
        "method": "zscore",
        "reason": f"mean={mean:.3f}, std={std:.3f}, threshold={threshold}",
    }


def mad_detector(current: float, history: Iterable[float], threshold: float = 3.5) -> dict[str, Any]:
    """Robust example, intentionally incomplete around zero-MAD edge cases.

    Students may improve this function and/or use it from auto mode.
    """
    values = np.asarray(list(history), dtype=float)
    if values.size < 5:
        return {"is_anomaly": False, "score": 0.0, "method": "mad", "reason": "insufficient_history"}
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0:
        different = float(current) != median
        return {"is_anomaly": different, "score": float("inf") if different else 0.0,
                "method": "mad", "reason": "constant_baseline"}
    modified_z = 0.6745 * abs(float(current) - median) / mad
    return {
        "is_anomaly": bool(modified_z > threshold),
        "score": float(modified_z),
        "method": "mad",
        "reason": f"median={median:.3f}, mad={mad:.3f}, threshold={threshold}",
    }


def detect_anomaly(
    current: float,
    history: Iterable[float],
    *,
    method: str = "auto",
    threshold: float = 3.0,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stable lab API.

    Current starter behavior:
    - `zscore`: basic z-score.
    - `mad`: MAD example.
    - `auto`: still uses naive z-score and ignores context.

    TODO(student): make `auto` context-aware. Useful context keys used by the
    instructor may include `day_of_week`, `same_segment_history`,
    `metric_name`, `known_event`, and `trend`.
    """
    if method == "mad":
        return mad_detector(current, history)
    if method == "zscore":
        return zscore_detector(current, history, threshold=threshold)
    if method == "auto":
        selected_history = history
        baseline_name = "all_history"
        if context and context.get("same_segment_history"):
            selected_history = context["same_segment_history"]
            baseline_name = "same_segment_history"
        values = list(selected_history)
        result = mad_detector(current, values, threshold=max(3.5, threshold)) if len(values) >= 5 else zscore_detector(current, values, threshold=threshold)
        result["method"] = f"auto:{result['method']}"
        result["reason"] += f"; baseline={baseline_name}"
        if context and context.get("known_event"):
            result["reason"] += "; known_event=true"
        return result
    raise ValueError(f"Unsupported method: {method}")
