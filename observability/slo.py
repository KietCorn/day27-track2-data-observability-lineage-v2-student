from __future__ import annotations

from typing import Any
import math


def calculate_slo(target: float, bad_events: int, total_events: int) -> dict[str, Any]:
    if not 0 < target < 1:
        raise ValueError("target must be between 0 and 1 (exclusive)")
    if bad_events < 0 or total_events < 0 or bad_events > total_events:
        raise ValueError("invalid event counts")
    allowed_bad_rate = 1.0 - target
    if total_events == 0:
        return {
            "target": target,
            "actual_bad_rate": 0.0,
            "allowed_bad_rate": allowed_bad_rate,
            "burn_rate": 0.0,
            "remaining_error_budget_fraction": 1.0,
            "breached": False,
        }
    actual_bad_rate = bad_events / total_events
    burn_rate = actual_bad_rate / allowed_bad_rate
    consumed_fraction = min(1.0, actual_bad_rate / allowed_bad_rate)
    return {
        "target": target,
        "actual_bad_rate": actual_bad_rate,
        "allowed_bad_rate": allowed_bad_rate,
        "burn_rate": burn_rate,
        "remaining_error_budget_fraction": max(0.0, 1.0 - consumed_fraction),
        "breached": bool(actual_bad_rate > allowed_bad_rate),
    }


def evaluate_multiwindow_burn(
    *,
    short_window_burn: float,
    long_window_burn: float,
    policy: str = "starter",
) -> dict[str, Any]:
    """Evaluate a two-window fast-burn policy."""
    if not all(math.isfinite(float(value)) and float(value) >= 0 for value in (short_window_burn, long_window_burn)):
        raise ValueError("burn rates must be finite and non-negative")
    fast_threshold, sustained_threshold = 14.0, 2.0
    page = short_window_burn >= fast_threshold and long_window_burn >= sustained_threshold
    if page:
        severity = "critical"
        reason = f"sustained_fast_burn short>={fast_threshold} and long>={sustained_threshold}"
    elif short_window_burn >= fast_threshold:
        severity = "warning"
        reason = "transient_fast_burn_short_window_only"
    else:
        severity = "info"
        reason = "burn_within_policy"
    return {"page": page, "severity": severity, "reason": reason,
            "short_window_burn": short_window_burn, "long_window_burn": long_window_burn}
