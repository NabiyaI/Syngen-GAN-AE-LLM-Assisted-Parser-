from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import asdict
from difflib import get_close_matches
from typing import Any

from .domains import DOMAIN_KEYWORDS, DOMAIN_REGISTRY, DomainProfile
from .models import ParseErrorReport, PromptSpec

INTENT_WORDS = ("generate", "create", "make", "simulate", "build", "produce", "need", "want")
STOPWORDS = {
    "with",
    "and",
    "for",
    "records",
    "patients",
    "customers",
    "users",
    "dataset",
    "data",
    "mostly",
    "high",
    "low",
    "industry",
    "domain",
    "around",
    "about",
    "approximately",
    "roughly",
    "at",
    "least",
    "most",
    "than",
    "more",
    "less",
    "under",
    "over",
    "above",
    "below",
    "near",
    "roughly",
    "approximately",
}


class PromptParseException(Exception):
    def __init__(self, report: ParseErrorReport):
        super().__init__(report.message)
        self.report = report


def parse_prompt(text: str) -> PromptSpec:
    raw = text.strip()
    if not raw:
        raise PromptParseException(ParseErrorReport(message="Prompt is empty."))

    normalized = _normalize(raw)
    has_intent = _validate_intent(normalized)
    segments = [s.strip() for s in normalized.split(";") if s.strip()]

    kv_pairs: dict[str, str] = {}
    nl_segments: list[str] = []
    for segment in segments:
        if "=" in segment and _looks_like_kv(segment):
            k, v = segment.split("=", 1)
            kv_pairs[k.strip()] = v.strip()
        else:
            nl_segments.append(segment)

    warnings: list[str] = []
    if not has_intent:
        warnings.append(
            "No explicit generation intent found. Parsed with defaults; add generate/create for clearer behavior."
        )
    profile_name, profile = _resolve_profile(normalized, kv_pairs, warnings)

    spec = PromptSpec(target_dataset_profile=profile_name)
    clause_status: list[tuple[str, bool]] = []
    conflicts: list[tuple[str, str, str]] = []

    _extract_rows(normalized, spec)
    _extract_seed_and_strict(kv_pairs, spec, warnings)
    strict_explicit = "strict" in kv_pairs

    nl_text = ", ".join(nl_segments)
    if nl_text:
        _extract_numeric_constraints(nl_text, profile, spec, conflicts, clause_status)
        _extract_categorical_constraints(nl_text, profile, spec, conflicts, clause_status)
        _extract_distribution_hints(nl_text, profile, spec, clause_status, warnings)

    kv_overrides = deepcopy(kv_pairs)
    for system_key in ("n", "rows", "strict", "seed", "industry", "domain", "profile"):
        kv_overrides.pop(system_key, None)
    _apply_kv_overrides(kv_overrides, profile, spec, conflicts, warnings)

    _detect_unparsed_clauses(nl_text, clause_status, warnings)
    _normalize_distribution_hints(spec)

    spec.priority_rules = _build_priority_rules(spec.filters)
    if conflicts:
        raise PromptParseException(_build_conflict_report(conflicts, warnings))

    if strict_explicit and spec.strict_mode and any(
        w.startswith("Unknown key") or w.startswith("Unrecognized clause")
        for w in warnings
    ):
        raise PromptParseException(
            ParseErrorReport(
                message="Strict mode rejected prompt due to unknown or unrecognized tokens.",
                offending_clauses=[w for w in warnings if "Unrecognized clause" in w],
                suggested_prompt="Remove unknown terms or provide explicit supported key=value fields.",
                warnings=warnings,
            )
        )

    spec.warnings = warnings
    return spec


def parse_prompt_safe(text: str) -> dict[str, Any]:
    try:
        return {"ok": True, "spec": asdict(parse_prompt(text))}
    except PromptParseException as exc:
        return {"ok": False, "error": asdict(exc.report)}


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("≥", ">=").replace("≤", "<=").replace("–", "-")
    text = re.sub(r"\s+", " ", text)
    replacements = {
        "non-smoker": "non smoker",
        "nonsmoker": "non smoker",
        "e-commerce": "ecommerce",
        "blood sugar": "glucose",
        "credt score": "credit score",
        "creditscore": "credit score",
        "hba 1c": "hba1c",
        "at least": ">=",
        "at most": "<=",
        "no less than": ">=",
        "no more than": "<=",
        "greater than": ">",
        "less than": "<",
        "more than": ">",
        "under ": "< ",
        "over ": "> ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _validate_intent(text: str) -> None:
    return any(word in text for word in INTENT_WORDS)


def _looks_like_kv(segment: str) -> bool:
    # KV overrides must be compact identifiers like n=500, strict=true, industry=finance.
    # Avoid treating long natural-language clauses containing '=' as KV.
    return re.match(r"^[a-zA-Z_][a-zA-Z0-9_]{0,39}\s*=", segment) is not None


def _resolve_profile(text: str, kv_pairs: dict[str, str], warnings: list[str]) -> tuple[str, DomainProfile]:
    requested = kv_pairs.get("industry") or kv_pairs.get("domain") or kv_pairs.get("profile")
    if requested:
        requested = requested.strip().lower()
        if requested in DOMAIN_REGISTRY:
            return requested, DOMAIN_REGISTRY[requested]
        if requested in DOMAIN_KEYWORDS:
            key = DOMAIN_KEYWORDS[requested]
            return key, DOMAIN_REGISTRY[key]
        raise PromptParseException(
            ParseErrorReport(
                message=f"Unsupported domain '{requested}'.",
                suggested_prompt="Use one of: healthcare, finance, or ecommerce.",
            )
        )

    for keyword, profile_name in DOMAIN_KEYWORDS.items():
        if keyword in text:
            return profile_name, DOMAIN_REGISTRY[profile_name]
    return "healthcare_v1", DOMAIN_REGISTRY["healthcare_v1"]


def _extract_rows(text: str, spec: PromptSpec) -> None:
    match = re.search(r"(?:generate|create|make|simulate)\s+(\d+)", text)
    if match:
        spec.n_rows = int(match.group(1))
        return
    match = re.search(r"\b(\d+)\s*(rows|records|samples|customers|patients|users)\b", text)
    if match:
        spec.n_rows = int(match.group(1))
        return
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*k\s*(rows|records|samples|customers|patients|users)?\b", text)
    if match:
        spec.n_rows = int(float(match.group(1)) * 1000)
        return
    match = re.search(
        r"\b(?:around|about|approximately|roughly)?\s*(\d+)\s*(rows|records|samples|customers|patients|users)\b",
        text,
    )
    if match:
        spec.n_rows = int(match.group(1))
        return
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*m\s*(rows|records|samples|customers|patients|users)?\b", text)
    if match:
        spec.n_rows = int(float(match.group(1)) * 1_000_000)


def _extract_seed_and_strict(kv_pairs: dict[str, str], spec: PromptSpec, warnings: list[str]) -> None:
    if "n" in kv_pairs:
        spec.n_rows = _safe_positive_int(kv_pairs["n"], "n", warnings, default=spec.n_rows)
    if "rows" in kv_pairs:
        spec.n_rows = _safe_positive_int(kv_pairs["rows"], "rows", warnings, default=spec.n_rows)
    if "seed" in kv_pairs:
        try:
            spec.seed = int(kv_pairs["seed"])
        except ValueError:
            warnings.append(f"Invalid seed value '{kv_pairs['seed']}' ignored.")
    if "strict" in kv_pairs:
        spec.strict_mode = kv_pairs["strict"].strip().lower() in {"true", "1", "yes"}


def _extract_numeric_constraints(
    text: str,
    profile: DomainProfile,
    spec: PromptSpec,
    conflicts: list[tuple[str, str, str]],
    clause_status: list[tuple[str, bool]],
) -> None:
    alias_pattern = _column_group_pattern(profile)

    for m in re.finditer(
        rf"\b({alias_pattern})\b\s*(?:between\s+)?(\d+(?:\.\d+)?)\s*(?:-|to|and)\s*(\d+(?:\.\d+)?)",
        text,
    ):
        raw_col = m.group(1).strip()
        col = _resolve_column(raw_col, profile)
        if col:
            _merge_range_filter(spec, col, float(m.group(2)), float(m.group(3)), m.group(0), conflicts)
            clause_status.append((m.group(0), True))

    for m in re.finditer(
        rf"\bbetween\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s+({alias_pattern})\b",
        text,
    ):
        raw_col = m.group(3).strip()
        col = _resolve_column(raw_col, profile)
        if col:
            _merge_range_filter(spec, col, float(m.group(1)), float(m.group(2)), m.group(0), conflicts)
            clause_status.append((m.group(0), True))

    for m in re.finditer(rf"\b({alias_pattern})\b\s*(>=|<=|>|<|=)\s*(\d+(?:\.\d+)?)", text):
        raw_col = m.group(1).strip()
        col = _resolve_column(raw_col, profile)
        if not col:
            continue
        value = float(m.group(3))
        op = m.group(2)
        _merge_operator_filter(spec, col, op, value, m.group(0), conflicts)
        clause_status.append((m.group(0), True))

    for m in re.finditer(rf"\b({alias_pattern})\b\s*(above|below)\s*(\d+(?:\.\d+)?)", text):
        raw_col = m.group(1).strip()
        col = _resolve_column(raw_col, profile)
        if not col:
            continue
        op = ">" if m.group(2) == "above" else "<"
        _merge_operator_filter(spec, col, op, float(m.group(3)), m.group(0), conflicts)
        clause_status.append((m.group(0), True))

    for m in re.finditer(rf"\bhigh\s+({alias_pattern})\b", text):
        col = _resolve_column(m.group(1).strip(), profile)
        if col:
            low, high = profile.numeric_bounds.get(col, (0.0, 1.0))
            cutoff = low + (high - low) * 0.7
            _merge_operator_filter(spec, col, ">", cutoff, m.group(0), conflicts, heuristic_high=True)
            clause_status.append((m.group(0), True))

    for m in re.finditer(rf"\blow\s+({alias_pattern})\b", text):
        col = _resolve_column(m.group(1).strip(), profile)
        if col:
            low, high = profile.numeric_bounds.get(col, (0.0, 1.0))
            cutoff = low + (high - low) * 0.3
            _merge_operator_filter(spec, col, "<", cutoff, m.group(0), conflicts, heuristic_high=True)
            clause_status.append((m.group(0), True))


def _extract_categorical_constraints(
    text: str,
    profile: DomainProfile,
    spec: PromptSpec,
    conflicts: list[tuple[str, str, str]],
    clause_status: list[tuple[str, bool]],
) -> None:
    phrase_items = sorted(profile.phrase_to_value.items(), key=lambda kv: len(kv[0]), reverse=True)
    consumed: list[tuple[int, int]] = []

    for phrase, (column, value) in phrase_items:
        if re.search(rf"\bnot\s+{re.escape(phrase)}\b", text):
            neg = _negate_binary_value(value)
            if neg is not None:
                _merge_exact_filter(spec, column, neg, f"not {phrase}", conflicts)
                clause_status.append((f"not {phrase}", True))

    for phrase, (column, value) in phrase_items:
        for m in re.finditer(rf"\b{re.escape(phrase)}\b", text):
            span = m.span()
            if any(_overlap(span, old) for old in consumed):
                continue
            _merge_exact_filter(spec, column, value, phrase, conflicts)
            clause_status.append((phrase, True))
            consumed.append(span)
            break

    alias_pattern = _column_group_pattern(profile)
    for m in re.finditer(rf"\b({alias_pattern})\b\s*(?:is|=)\s*([a-z_]+)", text):
        raw_col = m.group(1).strip()
        col = _resolve_column(raw_col, profile)
        if not col or col in profile.numeric_columns:
            continue
        value = m.group(2).strip()
        allowed = profile.categorical_values.get(col)
        if allowed and value in allowed:
            _merge_exact_filter(spec, col, value, m.group(0), conflicts)
            clause_status.append((m.group(0), True))

    for m in re.finditer(rf"\b([a-z_]+)\s+({alias_pattern})\b", text):
        value = m.group(1).strip()
        raw_col = m.group(2).strip()
        col = _resolve_column(raw_col, profile)
        if not col or col in profile.numeric_columns:
            continue
        allowed = profile.categorical_values.get(col)
        if allowed and value in allowed:
            _merge_exact_filter(spec, col, value, m.group(0), conflicts)
            clause_status.append((m.group(0), True))

    for m in re.finditer(rf"\b({alias_pattern})\b\s*[: ]\s*([a-z_]+)", text):
        raw_col = m.group(1).strip()
        col = _resolve_column(raw_col, profile)
        if not col or col in profile.numeric_columns:
            continue
        value = m.group(2).strip()
        allowed = profile.categorical_values.get(col)
        if allowed and value in allowed:
            _merge_exact_filter(spec, col, value, m.group(0), conflicts)
            clause_status.append((m.group(0), True))


def _extract_distribution_hints(
    text: str,
    profile: DomainProfile,
    spec: PromptSpec,
    clause_status: list[tuple[str, bool]],
    warnings: list[str],
) -> None:
    for m in re.finditer(r"\bmostly\s+([a-z_ ]+?)(?:,|$)", text):
        phrase = m.group(1).strip()
        mapped = _map_value_phrase(phrase, profile)
        if mapped:
            col, value = mapped
            spec.distribution_hints.setdefault(col, {})
            spec.distribution_hints[col][value] = 0.7
            clause_status.append((m.group(0), True))
        else:
            warnings.append(f"Unknown distribution phrase '{phrase}' ignored.")

    for m in re.finditer(r"(\d{1,3})%\s+([a-z_ ]+?)(?:,|$)", text):
        pct = float(m.group(1)) / 100.0
        phrase = m.group(2).strip()
        mapped = _map_value_phrase(phrase, profile)
        if mapped:
            col, value = mapped
            spec.distribution_hints.setdefault(col, {})
            spec.distribution_hints[col][value] = pct
            clause_status.append((m.group(0), True))

    for m in re.finditer(r"\b(majority|minority|half)\s+([a-z_ ]+?)(?:,|$)", text):
        modifier = m.group(1)
        phrase = m.group(2).strip()
        mapped = _map_value_phrase(phrase, profile)
        if not mapped:
            continue
        col, value = mapped
        spec.distribution_hints.setdefault(col, {})
        if modifier == "majority":
            spec.distribution_hints[col][value] = 0.65
        elif modifier == "minority":
            spec.distribution_hints[col][value] = 0.35
        else:
            spec.distribution_hints[col][value] = 0.5
        clause_status.append((m.group(0), True))

    alias_pattern = _column_group_pattern(profile)
    for m in re.finditer(rf"\b({alias_pattern})\s*:\s*([a-z_]+)\s+([01](?:\.\d+)?)\s+([a-z_]+)\s+([01](?:\.\d+)?)", text):
        col = _resolve_column(m.group(1).strip(), profile)
        if not col:
            continue
        v1, p1, v2, p2 = m.group(2), float(m.group(3)), m.group(4), float(m.group(5))
        spec.distribution_hints[col] = {v1: p1, v2: p2}
        clause_status.append((m.group(0), True))


def _apply_kv_overrides(
    kv_pairs: dict[str, str],
    profile: DomainProfile,
    spec: PromptSpec,
    conflicts: list[tuple[str, str, str]],
    warnings: list[str],
) -> None:
    for raw_key, raw_value in kv_pairs.items():
        key = raw_key.strip().lower()
        col = _resolve_column(key, profile)
        if not col:
            warnings.append(f"Unknown key '{raw_key}' ignored.")
            continue
        value = raw_value.strip().lower()
        if re.match(r"^\d+(\.\d+)?-\d+(\.\d+)?$", value):
            low, high = value.split("-", 1)
            _replace_filter(spec, col, {"type": "range", "min": float(low), "max": float(high), "source": raw_key})
            continue
        m = re.match(r"^(>=|<=|>|<|=)\s*(\d+(\.\d+)?)$", value)
        if m:
            _replace_filter(
                spec,
                col,
                _filter_from_operator(m.group(1), float(m.group(2)), raw_key, heuristic_high=False),
            )
            continue
        if col in profile.numeric_columns:
            try:
                _replace_filter(spec, col, {"type": "exact", "value": float(value), "source": raw_key})
            except ValueError:
                warnings.append(f"Invalid numeric value for '{raw_key}' ignored.")
        else:
            _replace_filter(spec, col, {"type": "exact", "value": value, "source": raw_key})


def _detect_unparsed_clauses(text: str, clause_status: list[tuple[str, bool]], warnings: list[str]) -> None:
    clauses = [c.strip() for c in re.split(r",|\band\b", text) if c.strip()]
    recognized = {c for c, ok in clause_status if ok}
    for clause in clauses:
        clean = clause.strip()
        if any(clean in r or r in clean for r in recognized):
            continue
        tokens = [t for t in re.findall(r"[a-z_]+", clean) if t not in STOPWORDS]
        if tokens and not any(tok in INTENT_WORDS for tok in tokens):
            warnings.append(f"Unrecognized clause: '{clean}'")


def _normalize_distribution_hints(spec: PromptSpec) -> None:
    for col, probs in list(spec.distribution_hints.items()):
        total = sum(max(v, 0.0) for v in probs.values())
        if total <= 0:
            del spec.distribution_hints[col]
            continue
        if abs(total - 1.0) > 1e-6:
            spec.distribution_hints[col] = {k: v / total for k, v in probs.items()}


def _build_priority_rules(filters: dict[str, dict[str, Any]]) -> list[str]:
    hard = []
    soft = []
    for col, data in filters.items():
        if data["type"] == "range":
            hard.append(f"{col}:range")
        elif data["type"] == "exact":
            hard.append(f"{col}:exact")
        else:
            soft.append(f"{col}:{data['type']}")
    return hard + soft


def _build_conflict_report(
    conflicts: list[tuple[str, str, str]], warnings: list[str]
) -> ParseErrorReport:
    fields = sorted({c[0] for c in conflicts})
    offending = [f"{c[0]} => '{c[1]}' conflicts with '{c[2]}'" for c in conflicts]
    return ParseErrorReport(
        message="Conflicting constraints found. Prompt rejected.",
        conflicting_fields=fields,
        offending_clauses=offending,
        suggested_prompt="Use one consistent constraint per field (for example: age 30-50; not age 20-30 and age>50).",
        warnings=warnings,
    )


def _safe_positive_int(value: str, label: str, warnings: list[str], default: int) -> int:
    cleaned = value.strip().lower()
    if cleaned.endswith("m"):
        try:
            parsed = int(float(cleaned[:-1]) * 1_000_000)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    if cleaned.endswith("k"):
        try:
            parsed = int(float(cleaned[:-1]) * 1000)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    try:
        parsed = int(cleaned)
        if parsed <= 0:
            warnings.append(f"{label} must be positive; using default {default}.")
            return default
        return parsed
    except ValueError:
        warnings.append(f"Invalid integer for {label}: '{value}'. Using default {default}.")
        return default


def _resolve_column(raw: str, profile: DomainProfile) -> str | None:
    raw = raw.strip().lower()
    if raw in profile.aliases:
        return profile.aliases[raw]
    if raw in profile.numeric_columns:
        return raw
    if raw in profile.categorical_values:
        return raw
    candidates = set(profile.aliases.keys()) | set(profile.numeric_columns) | set(profile.categorical_values.keys())
    alt = raw.replace("_", " ")
    match = get_close_matches(alt, list(candidates), n=1, cutoff=0.8)
    if match:
        matched = match[0]
        return profile.aliases.get(matched, matched)
    return None


def _column_group_pattern(profile: DomainProfile) -> str:
    terms_set = set(profile.aliases.keys()) | set(profile.numeric_columns) | set(profile.categorical_values.keys())
    terms = sorted(terms_set, key=len, reverse=True)
    return "|".join(re.escape(t) for t in terms)


def _map_value_phrase(phrase: str, profile: DomainProfile) -> tuple[str, str] | None:
    phrase = phrase.strip().lower()
    if phrase in profile.phrase_to_value:
        return profile.phrase_to_value[phrase]
    parts = phrase.split()
    if len(parts) == 1:
        for col, values in profile.categorical_values.items():
            if phrase in values:
                return col, phrase
    return None


def _negate_binary_value(value: str) -> str | None:
    if value == "yes":
        return "no"
    if value == "no":
        return "yes"
    return None


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _merge_range_filter(
    spec: PromptSpec,
    column: str,
    low: float,
    high: float,
    source: str,
    conflicts: list[tuple[str, str, str]],
) -> None:
    if low > high:
        low, high = high, low
    new_filter = {"type": "range", "min": low, "max": high, "source": source}
    _merge_filter(spec, column, new_filter, conflicts)


def _merge_operator_filter(
    spec: PromptSpec,
    column: str,
    op: str,
    value: float,
    source: str,
    conflicts: list[tuple[str, str, str]],
    heuristic_high: bool = False,
) -> None:
    new_filter = _filter_from_operator(op, value, source, heuristic_high=heuristic_high)
    _merge_filter(spec, column, new_filter, conflicts)


def _filter_from_operator(op: str, value: float, source: str, heuristic_high: bool = False) -> dict[str, Any]:
    if op == "=":
        return {"type": "exact", "value": value, "source": source}
    if op == ">":
        return {"type": "range", "min": value, "max": None, "source": source, "heuristic": heuristic_high}
    if op == ">=":
        return {"type": "range", "min": value, "max": None, "source": source}
    if op == "<":
        return {"type": "range", "min": None, "max": value, "source": source, "heuristic": heuristic_high}
    return {"type": "range", "min": None, "max": value, "source": source}


def _merge_exact_filter(
    spec: PromptSpec,
    column: str,
    value: str | float,
    source: str,
    conflicts: list[tuple[str, str, str]],
) -> None:
    _merge_filter(spec, column, {"type": "exact", "value": value, "source": source}, conflicts)


def _replace_filter(spec: PromptSpec, column: str, new_filter: dict[str, Any]) -> None:
    spec.filters[column] = new_filter


def _merge_filter(
    spec: PromptSpec,
    column: str,
    new_filter: dict[str, Any],
    conflicts: list[tuple[str, str, str]],
) -> None:
    old = spec.filters.get(column)
    if not old:
        spec.filters[column] = new_filter
        return

    if old["type"] == "exact" and new_filter["type"] == "exact":
        if old["value"] != new_filter["value"]:
            conflicts.append((column, old["source"], new_filter["source"]))
        return

    if old["type"] == "range" and new_filter["type"] == "range":
        low = old.get("min")
        high = old.get("max")
        nlow = new_filter.get("min")
        nhigh = new_filter.get("max")
        min_val = max(v for v in [low, nlow] if v is not None) if low is not None or nlow is not None else None
        max_val = min(v for v in [high, nhigh] if v is not None) if high is not None or nhigh is not None else None
        if min_val is not None and max_val is not None and min_val > max_val:
            conflicts.append((column, old["source"], new_filter["source"]))
            return
        spec.filters[column] = {"type": "range", "min": min_val, "max": max_val, "source": f"{old['source']} + {new_filter['source']}"}
        return

    if old["type"] == "exact" and new_filter["type"] == "range":
        value = float(old["value"])
        min_ok = new_filter.get("min") is None or value >= new_filter["min"]
        max_ok = new_filter.get("max") is None or value <= new_filter["max"]
        if not (min_ok and max_ok):
            conflicts.append((column, old["source"], new_filter["source"]))
        return

    if old["type"] == "range" and new_filter["type"] == "exact":
        value = float(new_filter["value"])
        min_ok = old.get("min") is None or value >= old["min"]
        max_ok = old.get("max") is None or value <= old["max"]
        if not (min_ok and max_ok):
            conflicts.append((column, old["source"], new_filter["source"]))
            return
        spec.filters[column] = new_filter
