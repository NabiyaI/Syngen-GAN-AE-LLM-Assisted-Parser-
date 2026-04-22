from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParseErrorReport:
    message: str
    conflicting_fields: list[str] = field(default_factory=list)
    offending_clauses: list[str] = field(default_factory=list)
    suggested_prompt: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class PromptSpec:
    n_rows: int = 500
    filters: dict[str, dict[str, Any]] = field(default_factory=dict)
    distribution_hints: dict[str, dict[str, float]] = field(default_factory=dict)
    priority_rules: list[str] = field(default_factory=list)
    strict_mode: bool = False
    target_dataset_profile: str = "general"
    seed: int | None = None
    warnings: list[str] = field(default_factory=list)

