from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from prompt_parser.domains import get_domain_profile
from prompt_parser.models import PromptSpec


class SyntheticBackend(Protocol):
    def generate(self, spec: PromptSpec) -> list[dict[str, float | str]]:
        ...


@dataclass
class BaselineSamplerBackend:
    round_digits: int = 4

    def generate(self, spec: PromptSpec) -> list[dict[str, float | str]]:
        profile = get_domain_profile(spec.target_dataset_profile)
        rng = random.Random(spec.seed if spec.seed is not None else 42)

        rows: list[dict[str, float | str]] = []
        for _ in range(spec.n_rows):
            row: dict[str, float | str] = {}

            for col in sorted(profile.numeric_columns):
                low, high = profile.numeric_bounds.get(col, (0.0, 1.0))
                row[col] = round(rng.uniform(low, high), self.round_digits)

            for col, allowed in sorted(profile.categorical_values.items()):
                row[col] = rng.choice(sorted(allowed))

            self._apply_distribution_hints(row, spec, profile, rng)
            self._apply_filters(row, spec)
            rows.append(row)
        return rows

    def _apply_distribution_hints(
        self,
        row: dict[str, float | str],
        spec: PromptSpec,
        profile,
        rng: random.Random,
    ) -> None:
        for col, probs in spec.distribution_hints.items():
            if col not in profile.categorical_values or not probs:
                continue
            values = list(probs.keys())
            weights = list(probs.values())
            row[col] = rng.choices(values, weights=weights, k=1)[0]

    def _apply_filters(self, row: dict[str, float | str], spec: PromptSpec) -> None:
        for col, filt in spec.filters.items():
            if filt["type"] == "exact":
                row[col] = filt["value"]
                continue
            if filt["type"] == "range":
                if col not in row:
                    continue
                val = row[col]
                if not isinstance(val, (float, int)):
                    continue
                low = filt.get("min")
                high = filt.get("max")
                if low is not None and val < low:
                    val = low
                if high is not None and val > high:
                    val = high
                row[col] = val


def build_backend(name: str, ckpt_dir: str | None = None) -> SyntheticBackend:
    backend_name = (name or "baseline").strip().lower()
    if backend_name == "baseline":
        return BaselineSamplerBackend()
    if backend_name == "gan_ae":
        if not ckpt_dir:
            raise ValueError("gan_ae backend requires --ckpt-dir")
        from gan_ae_backend import GanAeBackend

        return GanAeBackend(ckpt_dir=ckpt_dir)
    raise ValueError(f"Unsupported backend '{name}'. Choose baseline or gan_ae.")


def generate_synthetic(spec: PromptSpec, backend: SyntheticBackend | None = None) -> list[dict[str, float | str]]:
    impl = backend or BaselineSamplerBackend()
    return impl.generate(spec)


def write_csv(rows: list[dict[str, float | str]], out_path: str) -> str:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return str(path)
    headers = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(path)
