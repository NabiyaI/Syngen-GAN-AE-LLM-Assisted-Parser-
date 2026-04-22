from __future__ import annotations

from pathlib import Path

import pytest

from generator import build_backend, generate_synthetic, write_csv
from prompt_parser.models import PromptSpec


def test_generate_respects_exact_and_range_filters() -> None:
    spec = PromptSpec(
        n_rows=50,
        target_dataset_profile="finance_v1",
        filters={
            "defaulted": {"type": "exact", "value": "yes", "source": "test"},
            "credit_score": {"type": "range", "min": 700.0, "max": 760.0, "source": "test"},
        },
        seed=7,
    )
    rows = generate_synthetic(spec)
    assert len(rows) == 50
    for row in rows:
        assert row["defaulted"] == "yes"
        assert 700.0 <= float(row["credit_score"]) <= 760.0


def test_generate_respects_distribution_hints() -> None:
    spec = PromptSpec(
        n_rows=200,
        target_dataset_profile="ecommerce_v1",
        distribution_hints={"device": {"mobile": 1.0}},
        seed=9,
    )
    rows = generate_synthetic(spec)
    assert all(r["device"] == "mobile" for r in rows)


def test_write_csv(tmp_path: Path) -> None:
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    out = write_csv(rows, str(tmp_path / "out.csv"))
    content = Path(out).read_text(encoding="utf-8")
    assert "a,b" in content
    assert "1,x" in content


def test_build_backend_baseline() -> None:
    backend = build_backend("baseline")
    assert backend.__class__.__name__ == "BaselineSamplerBackend"


def test_build_backend_gan_ae_requires_ckpt_dir() -> None:
    with pytest.raises(ValueError):
        build_backend("gan_ae", None)


def test_build_backend_gan_ae_with_ckpt_dir(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    backend = build_backend("gan_ae", str(ckpt_dir))
    assert backend.__class__.__name__ == "GanAeBackend"
