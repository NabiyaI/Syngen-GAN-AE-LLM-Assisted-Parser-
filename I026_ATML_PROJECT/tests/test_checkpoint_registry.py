from __future__ import annotations

import json
from pathlib import Path

from checkpoint_registry import (
    load_registry,
    resolve_checkpoint_for_profile,
    validate_checkpoint_dir,
)


def test_resolve_checkpoint_direct_mapping(tmp_path: Path) -> None:
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(json.dumps({"finance_v1": "checkpoints/finance"}), encoding="utf-8")
    registry = load_registry(str(reg_path))
    ckpt = resolve_checkpoint_for_profile("finance_v1", registry)
    assert ckpt == "checkpoints/finance"


def test_validate_checkpoint_dir_missing(tmp_path: Path) -> None:
    missing = validate_checkpoint_dir(str(tmp_path / "missing_dir"))
    assert "metadata.json" in missing
