from __future__ import annotations

import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt_parser.domains import get_domain_profile
from prompt_parser.models import PromptSpec


@dataclass
class GanAeBackend:
    ckpt_dir: str
    round_digits: int = 4

    def __post_init__(self) -> None:
        base = Path(self.ckpt_dir)
        if not base.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {self.ckpt_dir}")
        self.ckpt_path = base
        self.full_infer = None
        if self._has_full_artifacts():
            try:
                from gan_ae_full.infer import GanAeInference

                self.full_infer = GanAeInference(str(self.ckpt_path))
            except Exception:
                self.full_infer = None
        self.encoder = self._load_component("encoder")
        self.decoder = self._load_component("decoder")
        self.generator = self._load_component("generator")

    def generate(self, spec: PromptSpec) -> list[dict[str, float | str]]:
        if self.full_infer is not None:
            return self.full_infer.generate(spec)

        rng = random.Random(spec.seed if spec.seed is not None else 42)
        rows: list[dict[str, float | str]] = []
        profile = get_domain_profile(spec.target_dataset_profile)
        cond = self._spec_to_condition_vector(spec)

        for _ in range(spec.n_rows):
            noise = self._sample_noise(rng, dim=16)
            latent = self._generate_latent(noise, cond, rng)
            decoded = self._decode_latent(latent, profile, rng)
            row = self._apply_post_filters(decoded, spec)
            rows.append(row)
        return rows

    def _has_full_artifacts(self) -> bool:
        required = [
            "metadata.json",
            "preprocessor.pkl",
            "encoder.pt",
            "decoder.pt",
            "generator.pt",
        ]
        return all((self.ckpt_path / name).exists() for name in required)

    def _load_component(self, stem: str) -> Any:
        for ext, loader in (
            (".pkl", self._load_pickle),
            (".json", self._load_json),
            (".pt", self._load_torch),
            (".pth", self._load_torch),
        ):
            path = self.ckpt_path / f"{stem}{ext}"
            if path.exists():
                return loader(path)
        return None

    def _load_pickle(self, path: Path) -> Any:
        with path.open("rb") as handle:
            return pickle.load(handle)

    def _load_json(self, path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_torch(self, path: Path) -> Any:
        try:
            import torch  # type: ignore
        except Exception:
            return {"torch_checkpoint": str(path), "load_error": "torch not installed"}
        return torch.load(path, map_location="cpu")

    def _spec_to_condition_vector(self, spec: PromptSpec) -> dict[str, Any]:
        return {
            "profile": spec.target_dataset_profile,
            "filters": spec.filters,
            "distribution_hints": spec.distribution_hints,
        }

    def _sample_noise(self, rng: random.Random, dim: int) -> list[float]:
        return [rng.uniform(-1.0, 1.0) for _ in range(dim)]

    def _generate_latent(
        self, noise: list[float], cond: dict[str, Any], rng: random.Random
    ) -> list[float]:
        # Hook: replace this with actual GAN generator forward pass.
        # Current fallback keeps deterministic pseudo-latent behavior.
        cond_bias = len(cond.get("filters", {})) * 0.01
        return [n + cond_bias + rng.uniform(-0.05, 0.05) for n in noise]

    def _decode_latent(
        self, latent: list[float], profile, rng: random.Random
    ) -> dict[str, float | str]:
        # Hook: replace this with AE decoder forward pass.
        row: dict[str, float | str] = {}
        for idx, col in enumerate(sorted(profile.numeric_columns)):
            low, high = profile.numeric_bounds.get(col, (0.0, 1.0))
            base = (latent[idx % len(latent)] + 1.0) / 2.0
            value = low + (high - low) * max(0.0, min(1.0, base))
            row[col] = round(value, self.round_digits)
        for col, values in sorted(profile.categorical_values.items()):
            ordered = sorted(values)
            row[col] = ordered[rng.randrange(len(ordered))]
        return row

    def _apply_post_filters(
        self, row: dict[str, float | str], spec: PromptSpec
    ) -> dict[str, float | str]:
        for col, filt in spec.filters.items():
            if filt["type"] == "exact":
                row[col] = filt["value"]
                continue
            if filt["type"] == "range" and col in row and isinstance(row[col], (int, float)):
                low = filt.get("min")
                high = filt.get("max")
                val = float(row[col])
                if low is not None and val < low:
                    val = low
                if high is not None and val > high:
                    val = high
                row[col] = round(val, self.round_digits)
        return row
