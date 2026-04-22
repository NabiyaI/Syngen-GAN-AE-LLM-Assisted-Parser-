from __future__ import annotations

from .llm_parser import LLMParseException, rewrite_prompt_with_llm
from .models import ParseErrorReport, PromptSpec
from .parser import PromptParseException, parse_prompt


def parse_user_prompt(text: str, mode: str = "hybrid") -> PromptSpec:
    parse_mode = (mode or "hybrid").strip().lower()
    if parse_mode not in {"rules", "llm", "hybrid"}:
        raise PromptParseException(
            ParseErrorReport(
                message=f"Unsupported parse mode '{mode}'. Use rules, llm, or hybrid."
            )
        )

    if parse_mode == "rules":
        return parse_prompt(text)

    if parse_mode == "llm":
        canonical = _rewrite_or_raise(text)
        spec = parse_prompt(canonical)
        spec.warnings.append("Parsed via llm mode.")
        return spec

    # hybrid
    try:
        canonical = rewrite_prompt_with_llm(text)
        spec = parse_prompt(canonical)
        spec.warnings.append("Parsed via hybrid mode using llm canonical rewrite.")
        return spec
    except (LLMParseException, PromptParseException) as llm_exc:
        spec = parse_prompt(text)
        spec.warnings.append(f"Hybrid fallback to rules parser: {type(llm_exc).__name__}.")
        return spec


def _rewrite_or_raise(text: str) -> str:
    try:
        return rewrite_prompt_with_llm(text)
    except LLMParseException as exc:
        raise PromptParseException(
            ParseErrorReport(
                message=f"LLM parse mode failed: {exc}",
                suggested_prompt=(
                    "Set GEMINI_API_KEY (or OPENAI_API_KEY with PROMPT_PARSER_LLM_PROVIDER=openai), "
                    "or set PROMPT_PARSER_LLM_MOCK=1, or use --parse-mode rules/hybrid."
                ),
            )
        ) from exc
