from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class LLMParseException(Exception):
    pass


def _load_env_file() -> None:
    for env_path in (Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"):
        if not env_path.is_file():
            continue
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or key in os.environ:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            os.environ[key] = value


SYSTEM_PROMPT = """You convert user requests for synthetic tabular data into a canonical prompt.
Return JSON only:
{"canonical_prompt":"..."}

Rules:
- Keep original intent and constraints.
- Use a concise canonical style the downstream parser can understand.
- Include generation verb at start (Generate/Create/Simulate/Make).
- Preserve row count, domain, numeric filters, categorical filters, percentages, strict/seed hints when present.
- Do not add fields that are not implied by user input.
"""


def rewrite_prompt_with_llm(user_prompt: str, timeout_s: int = 20) -> str:
    _load_env_file()
    if os.getenv("PROMPT_PARSER_LLM_MOCK", "").strip().lower() in {"1", "true", "yes"}:
        return _mock_rewrite(user_prompt)

    provider = os.getenv("PROMPT_PARSER_LLM_PROVIDER", "gemini").strip().lower()
    if provider == "gemini":
        payload = _call_gemini(user_prompt, timeout_s)
    elif provider == "openai":
        payload = _call_openai(user_prompt, timeout_s)
    else:
        raise LLMParseException(
            f"Unsupported PROMPT_PARSER_LLM_PROVIDER '{provider}'. Use gemini or openai."
        )

    canonical = str(payload.get("canonical_prompt", "")).strip()
    if not canonical:
        raise LLMParseException("LLM JSON missing canonical_prompt.")
    return canonical


def _call_gemini(user_prompt: str, timeout_s: int) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMParseException("GEMINI_API_KEY not set for llm parse mode.")

    model = os.getenv("PROMPT_PARSER_GEMINI_MODEL", "gemini-2.5-flash")
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{urllib.parse.quote(model)}:generateContent?key={urllib.parse.quote(api_key)}"
    )
    body = {
        "contents": [
            {
                "parts": [
                    {"text": SYSTEM_PROMPT},
                    {"text": user_prompt},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        msg = exc.read().decode("utf-8", errors="replace")
        raise LLMParseException(f"Gemini API HTTP {exc.code}: {msg}") from exc
    except Exception as exc:
        raise LLMParseException(f"Gemini API request failed: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMParseException("Gemini API returned invalid JSON response.") from exc

    text = _extract_gemini_text(payload)
    if not text:
        raise LLMParseException("Gemini API returned empty text.")

    parsed = _extract_json_object(text)
    if not isinstance(parsed, dict):
        raise LLMParseException("Gemini output is not a JSON object.")
    return parsed


def _extract_gemini_text(payload: dict) -> str:
    candidates = payload.get("candidates", [])
    if not candidates:
        return ""
    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    out: list[str] = []
    for part in parts:
        text = part.get("text")
        if isinstance(text, str):
            out.append(text)
    return "\n".join(out).strip()


def _call_openai(user_prompt: str, timeout_s: int) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMParseException("OPENAI_API_KEY not set for llm parse mode.")

    model = os.getenv("PROMPT_PARSER_LLM_MODEL", "gpt-4.1-mini")
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        raise LLMParseException("openai package not installed. Run: pip install openai") from exc

    client = OpenAI(api_key=api_key, timeout=timeout_s)
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:
        raise LLMParseException(f"OpenAI request failed: {exc}") from exc

    content = response.choices[0].message.content if response.choices else None
    if not content:
        raise LLMParseException("OpenAI returned empty response.")

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMParseException("OpenAI did not return valid JSON.") from exc


def _extract_json_object(text: str) -> dict:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
        if not match:
            raise LLMParseException("LLM output missing JSON object.")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMParseException("Failed to parse JSON object from LLM output.") from exc


def _mock_rewrite(user_prompt: str) -> str:
    text = user_prompt.strip()
    low = text.lower()
    if any(w in low for w in ("generate", "create", "simulate", "make", "produce", "build")):
        return text
    return f"Generate {text}"
