from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from observability.anomaly import zscore_detector


def approximate_token_lengths(texts: Iterable[str]) -> list[int]:
    # Deliberately simple proxy; no tokenizer/model download needed.
    return [len(str(t).split()) for t in texts]


def detect_text_length_shift(
    current_texts: Iterable[str],
    baseline_batch_means: Iterable[float],
    *,
    threshold: float = 3.0,
) -> dict[str, Any]:
    lengths = approximate_token_lengths(current_texts)
    current_mean = float(np.mean(lengths)) if lengths else 0.0
    result = zscore_detector(current_mean, baseline_batch_means, threshold=threshold)
    result["metric"] = "mean_text_length"
    result["current_mean"] = current_mean
    return result


def detect_embedding_norm_shift(
    current_norms: Iterable[float], baseline_norms: Iterable[float]
) -> dict[str, Any]:
    current = np.asarray(list(current_norms), dtype=float)
    baseline = np.asarray(list(baseline_norms), dtype=float)
    current = current[np.isfinite(current)]
    baseline = baseline[np.isfinite(baseline)]
    if current.size == 0 or baseline.size == 0:
        return {"is_anomaly": False, "score": 0.0, "method": "mad", "reason": "empty_input",
                "metric": "embedding_norm", "current_mean": float(np.mean(current)) if current.size else 0.0}
    result = zscore_detector(float(np.mean(current)), baseline)
    result["metric"] = "embedding_norm"
    result["current_mean"] = float(np.mean(current))
    result["baseline_mean"] = float(np.mean(baseline))
    return result