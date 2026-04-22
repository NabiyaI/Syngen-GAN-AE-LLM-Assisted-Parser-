from __future__ import annotations

import pytest

from prompt_parser.parser import PromptParseException, parse_prompt


def test_happy_path_healthcare() -> None:
    spec = parse_prompt("Generate 600 diabetic females age 40-65 with high glucose in healthcare")
    assert spec.n_rows == 600
    assert spec.target_dataset_profile == "healthcare_v1"
    assert spec.filters["diabetes"]["value"] == "yes"
    assert spec.filters["sex"]["value"] == "female"
    assert spec.filters["age"]["type"] == "range"
    assert spec.filters["glucose"]["type"] == "range"


def test_mixed_format_kv_override() -> None:
    spec = parse_prompt("Create patients with hypertension; n=1000; strict=true; glucose>=130; industry=healthcare")
    assert spec.n_rows == 1000
    assert spec.strict_mode is True
    assert spec.filters["hypertension"]["value"] == "yes"
    assert spec.filters["glucose"]["type"] == "range"
    assert spec.filters["glucose"]["min"] == 130.0


def test_distribution_hint_mostly_and_percent() -> None:
    spec = parse_prompt("Generate 400 healthcare records, mostly female, 30% smoker; strict=false")
    assert spec.n_rows == 400
    assert spec.strict_mode is False
    assert "sex" in spec.distribution_hints
    assert abs(sum(spec.distribution_hints["sex"].values()) - 1.0) < 1e-6


def test_conflict_detected() -> None:
    with pytest.raises(PromptParseException) as exc:
        parse_prompt("Generate 500 healthcare patients age 20-30 and age>50")
    assert "Conflicting constraints" in exc.value.report.message
    assert "age" in exc.value.report.conflicting_fields


def test_unknown_column_with_strict_fails() -> None:
    with pytest.raises(PromptParseException) as exc:
        parse_prompt("Generate 500 records; strict=true; vitamin_z=high")
    assert "Strict mode rejected prompt" in exc.value.report.message


def test_freeform_without_intent_is_accepted() -> None:
    spec = parse_prompt("finance customers, mostly defaulted, credit score 600-700")
    assert spec.target_dataset_profile == "finance_v1"
    assert spec.filters["credit_score"]["type"] == "range"
    assert any("No explicit generation intent" in w for w in spec.warnings)


def test_fuzzy_column_match() -> None:
    spec = parse_prompt("Generate 200 finance customers with credt score 650-700")
    assert "credit_score" in spec.filters


def test_rows_about_k_notation() -> None:
    spec = parse_prompt("Need around 1.5k ecommerce customers with churned is yes")
    assert spec.n_rows == 1500


def test_majority_distribution_hint() -> None:
    spec = parse_prompt("Create 500 ecommerce users with majority mobile users")
    assert "device" in spec.distribution_hints
    assert "mobile" in spec.distribution_hints["device"]


def test_at_least_normalization_to_operator() -> None:
    spec = parse_prompt("Generate 400 finance records with credit score at least 700")
    assert spec.filters["credit_score"]["type"] == "range"
    assert spec.filters["credit_score"]["min"] == 700.0


def test_nl_clause_with_equals_is_not_misread_as_kv() -> None:
    spec = parse_prompt("Generate 1500 finance customers with defaulted=yes and credit score >=620")
    assert spec.n_rows == 1500
    assert spec.filters["defaulted"]["value"] == "yes"
    assert spec.filters["credit_score"]["min"] == 620.0


def test_non_defaulted_no_conflict() -> None:
    spec = parse_prompt("Generate 300 finance customers with non defaulted")
    assert spec.filters["defaulted"]["value"] == "no"
