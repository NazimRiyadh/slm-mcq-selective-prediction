"""Strict output-only feature whitelist for a future all-model rerun.

This module is intentionally conservative. It excludes any feature whose name
suggests that it was derived from hidden activations or layer-level summaries.
It is supplied to make the paper's proposed six-model output-only rerun
operationally precise; it does not replace any archived result in the paper.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

_INTERNAL_MARKERS = re.compile(
    r"(?:layer|hidden|activation|embedding|residual|mlp|attention|attn|state)",
    flags=re.IGNORECASE,
)

_EXACT_OUTPUT_NAMES = {
    "top_prob",
    "max_prob",
    "p_max",
    "margin",
    "prob_margin",
    "logit_margin",
    "entropy",
    "n_options",
}

_OUTPUT_PREFIXES = (
    "prob_",
    "p_",
    "logit_",
    "score_",
    "option_prob_",
    "option_logit_",
    "option_score_",
)


def is_strict_output_feature(name: str) -> bool:
    """Return True only for explicitly permitted output-derived features."""
    normalized = name.strip().lower()
    if not normalized or _INTERNAL_MARKERS.search(normalized):
        return False
    if normalized in _EXACT_OUTPUT_NAMES:
        return True
    return normalized.startswith(_OUTPUT_PREFIXES)


def select_strict_output_features(columns: Iterable[str]) -> list[str]:
    """Select output-derived columns and reject ambiguous/internal columns."""
    return [column for column in columns if is_strict_output_feature(column)]


def audit_columns(columns: Iterable[str]) -> dict[str, list[str]]:
    """Return selected and rejected columns for a reproducible schema audit."""
    columns = list(columns)
    selected = select_strict_output_features(columns)
    selected_set = set(selected)
    return {
        "selected_output_only": selected,
        "rejected_or_ambiguous": [c for c in columns if c not in selected_set],
    }


if __name__ == "__main__":
    example = [
        "top_prob",
        "margin",
        "entropy",
        "n_options",
        "early_layer8_max",
        "final_hidden_norm",
    ]
    print(audit_columns(example))
