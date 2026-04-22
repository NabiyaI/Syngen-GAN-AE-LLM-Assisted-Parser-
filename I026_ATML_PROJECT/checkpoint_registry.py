from __future__ import annotations

import json
from pathlib import Path

from prompt_parser.domains import DOMAIN_KEYWORDS

DEFAULT_REGISTRY_PATH = "checkpoints/registry.json"
REQUIRED_GAN_AE_FILES = [
    "metadata.json",
    "preprocessor.pkl",
    "encoder.pt",
    "decoder.pt",
    "generator.pt",
]


def load_registry(path: str = DEFAULT_REGISTRY_PATH) -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint registry not found: {path}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint registry must be a JSON object.")
    out: dict[str, str] = {}
    for k, v in payload.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k.strip().lower()] = v.strip()
    return out


def resolve_checkpoint_for_profile(
    profile: str,
    registry: dict[str, str],
    prompt_text: str | None = None,
) -> str:
    profile_key = (profile or "").strip().lower()
    candidates: list[str] = []

    # If profile falls back to general, use prompt keywords first for better domain routing.
    if prompt_text:
        low = prompt_text.lower()
        matched_registry_keys: list[str] = []
        # First, match explicit registry keys directly from prompt text.
        for key in registry.keys():
            k = key.strip().lower()
            if not k or k in {"_default", "default", "general"}:
                continue
            probe = k.replace("_", " ")
            if probe in low or k in low:
                matched_registry_keys.append(k)
        for k in sorted(set(matched_registry_keys), key=len, reverse=True):
            candidates.append(k)
        for keyword, mapped in DOMAIN_KEYWORDS.items():
            if keyword in low:
                candidates.append(keyword)
                candidates.append(mapped)

    if profile_key and profile_key != "general":
        candidates.append(profile_key)
    if profile_key in DOMAIN_KEYWORDS:
        candidates.append(DOMAIN_KEYWORDS[profile_key])
    for keyword, mapped in DOMAIN_KEYWORDS.items():
        if keyword in profile_key:
            candidates.append(mapped)
    if profile_key:
        candidates.append(profile_key)
    candidates.extend(["general", "_default"])

    seen: set[str] = set()
    for key in candidates:
        key = key.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        path = registry.get(key)
        if path:
            return path
    raise FileNotFoundError(
        f"No checkpoint mapping found for profile '{profile}'. Add it to registry.json."
    )


def validate_checkpoint_dir(path: str) -> list[str]:
    base = Path(path)
    if not base.exists():
        return REQUIRED_GAN_AE_FILES[:]
    return [name for name in REQUIRED_GAN_AE_FILES if not (base / name).exists()]
