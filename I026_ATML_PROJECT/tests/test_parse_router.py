from __future__ import annotations

import pytest

import prompt_parser.llm_parser as llm_parser
from prompt_parser.parse_router import parse_user_prompt
from prompt_parser.parser import PromptParseException


def test_rules_mode_parses() -> None:
    spec = parse_user_prompt("Generate 500 finance customers with credit score 650-700", mode="rules")
    assert spec.n_rows == 500
    assert "credit_score" in spec.filters


def test_hybrid_mode_falls_back_when_llm_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_root = tmp_path
    monkeypatch.chdir(fake_root)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("PROMPT_PARSER_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("PROMPT_PARSER_GEMINI_MODEL", raising=False)
    monkeypatch.delenv("PROMPT_PARSER_LLM_MOCK", raising=False)
    monkeypatch.setattr(llm_parser, "__file__", str(fake_root / "prompt_parser" / "llm_parser.py"))
    spec = parse_user_prompt("Generate 400 ecommerce users with churned yes", mode="hybrid")
    assert spec.n_rows == 400
    assert any("Hybrid fallback to rules parser" in w for w in spec.warnings)


def test_llm_mode_with_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROMPT_PARSER_LLM_MOCK", "1")
    spec = parse_user_prompt("telecom customers around 900 rows churn is yes", mode="llm")
    assert spec.n_rows == 900
    assert any("Parsed via llm mode." in w for w in spec.warnings)


def test_llm_mode_loads_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "PROMPT_PARSER_LLM_MOCK=1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(llm_parser, "__file__", str(tmp_path / "prompt_parser" / "llm_parser.py"))
    monkeypatch.delenv("PROMPT_PARSER_LLM_MOCK", raising=False)
    monkeypatch.chdir(tmp_path)

    spec = parse_user_prompt("telecom customers around 900 rows churn is yes", mode="llm")
    assert spec.n_rows == 900
    assert any("Parsed via llm mode." in w for w in spec.warnings)


def test_llm_mode_without_key_fails(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    fake_root = tmp_path
    monkeypatch.chdir(fake_root)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("PROMPT_PARSER_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("PROMPT_PARSER_GEMINI_MODEL", raising=False)
    monkeypatch.delenv("PROMPT_PARSER_LLM_MOCK", raising=False)
    monkeypatch.setattr(llm_parser, "__file__", str(fake_root / "prompt_parser" / "llm_parser.py"))
    with pytest.raises(PromptParseException) as exc:
        parse_user_prompt("Generate 300 records with age 20-30", mode="llm")
    assert "LLM parse mode failed" in exc.value.report.message
