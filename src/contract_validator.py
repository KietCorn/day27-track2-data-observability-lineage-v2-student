"""Simple contract validator used as the starter baseline.

The implementation intentionally covers only common deterministic checks.
Students are expected to extend it with:
- stronger type validation/coercion rules,
- freshness checks,
- cross-field/cross-table assertions,
- severity-aware actions (block/quarantine/warn),
- richer observability metadata.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _issue(
    check: str,
    *,
    column: str | None,
    severity: str,
    passed: bool,
    details: str,
) -> dict[str, Any]:
    return {
        "check": check,
        "column": column,
        "severity": severity,
        "passed": bool(passed),
        "details": details,
    }


def load_contract(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_dataframe(df: pd.DataFrame, contract: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    columns = contract.get("columns", {})

    for column, rules in columns.items():
        severity = rules.get("severity", "warning")
        required = bool(rules.get("required", False))

        if column not in df.columns:
            if required:
                issues.append(
                    _issue(
                        "required_column",
                        column=column,
                        severity=severity,
                        passed=False,
                        details=f"Missing required column: {column}",
                    )
                )
            continue

        series = df[column]

        if required:
            null_count = int(series.isna().sum())
            issues.append(
                _issue(
                    "not_null",
                    column=column,
                    severity=severity,
                    passed=(null_count == 0),
                    details=f"null_count={null_count}",
                )
            )

        if rules.get("unique"):
            duplicate_count = int(series.duplicated(keep=False).sum())
            issues.append(
                _issue(
                    "unique",
                    column=column,
                    severity=severity,
                    passed=(duplicate_count == 0),
                    details=f"duplicate_rows={duplicate_count}",
                )
            )

        accepted = rules.get("accepted_values")
        if accepted is not None:
            invalid_mask = series.notna() & ~series.isin(accepted)
            invalid_count = int(invalid_mask.sum())
            issues.append(
                _issue(
                    "accepted_values",
                    column=column,
                    severity=severity,
                    passed=(invalid_count == 0),
                    details=f"invalid_count={invalid_count}; accepted={accepted}",
                )
            )

        declared_type = rules.get("type")
        if declared_type:
            non_null = series.dropna()
            if declared_type == "integer":
                numeric = pd.to_numeric(non_null, errors="coerce")
                valid = numeric.notna() & numeric.map(float).map(pd.api.types.is_number)
                valid &= numeric.map(lambda value: float(value).is_integer() if pd.notna(value) else False)
                valid &= ~non_null.map(lambda value: isinstance(value, (bool, pd.BooleanDtype)))
            elif declared_type == "number":
                numeric = pd.to_numeric(non_null, errors="coerce")
                valid = numeric.notna() & numeric.map(lambda value: pd.notna(value) and bool(pd.api.types.is_number(value)))
            elif declared_type == "string":
                valid = non_null.map(lambda value: isinstance(value, str))
            elif declared_type == "datetime":
                valid = pd.to_datetime(non_null, errors="coerce", utc=True).notna()
            else:
                valid = pd.Series(True, index=non_null.index)
            invalid_count = int((~valid).sum())
            issues.append(_issue("type", column=column, severity=severity,
                                 passed=invalid_count == 0,
                                 details=f"expected={declared_type}; invalid_count={invalid_count}"))

        if "min" in rules or "max" in rules:
            numeric = pd.to_numeric(series, errors="coerce")
            invalid = pd.Series(False, index=series.index)
            if "min" in rules:
                invalid |= numeric < rules["min"]
            if "max" in rules:
                invalid |= numeric > rules["max"]
            invalid_count = int(invalid.fillna(False).sum())
            issues.append(
                _issue(
                    "range",
                    column=column,
                    severity=severity,
                    passed=(invalid_count == 0),
                    details=f"invalid_count={invalid_count}",
                )
            )

    freshness = contract.get("freshness") or {}
    freshness_column = freshness.get("column")
    if freshness_column:
        severity = freshness.get("severity", "warning")
        if freshness_column not in df.columns:
            issues.append(_issue("freshness", column=freshness_column, severity=severity,
                                 passed=False, details="freshness_column_missing"))
        else:
            timestamps = pd.to_datetime(df[freshness_column], errors="coerce", utc=True)
            invalid_count = int(timestamps.isna().sum())
            if invalid_count:
                issues.append(_issue("freshness", column=freshness_column, severity=severity,
                                     passed=False, details=f"invalid_timestamp_count={invalid_count}"))
            else:
                reference = timestamps.max()
                delay = (reference - timestamps).dt.total_seconds().max() / 60
                max_delay = float(freshness.get("max_delay_minutes", 0))
                issues.append(_issue("freshness", column=freshness_column, severity=severity,
                                     passed=bool(delay <= max_delay),
                                     details=f"batch_span_minutes={delay:.3f}; max_delay_minutes={max_delay:g}"))

    return issues


def failed_issues(issues: list[dict[str, Any]], min_severity: str | None = None) -> list[dict[str, Any]]:
    failed = [i for i in issues if not i.get("passed", False)]
    if min_severity is None:
        return failed
    order = {"info": 0, "warning": 1, "critical": 2}
    threshold = order[min_severity]
    return [i for i in failed if order.get(i.get("severity", "warning"), 1) >= threshold]
