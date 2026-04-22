from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from prompt_parser.models import PromptSpec

from .models import AutoEncoder, LatentGenerator
from .preprocessing import TabularPreprocessor


class GanAeInference:
    def __init__(self, ckpt_dir: str, device: str = "cpu") -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.device = torch.device(device)
        self._load()

    def _load(self) -> None:
        metadata_path = self.ckpt_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json missing in {self.ckpt_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.metadata = metadata

        self.preprocessor = TabularPreprocessor.load(str(self.ckpt_dir))
        self.ae = AutoEncoder(
            input_dim=int(metadata["input_dim"]),
            latent_dim=int(metadata["latent_dim"]),
            hidden_dim=int(metadata["hidden_dim"]),
        ).to(self.device)
        self.generator = LatentGenerator(
            noise_dim=int(metadata["noise_dim"]),
            cond_dim=int(metadata["cond_dim"]),
            latent_dim=int(metadata["latent_dim"]),
            hidden_dim=int(metadata["hidden_dim"]),
        ).to(self.device)

        self.ae.encoder.load_state_dict(torch.load(self.ckpt_dir / "encoder.pt", map_location=self.device))
        self.ae.decoder.load_state_dict(torch.load(self.ckpt_dir / "decoder.pt", map_location=self.device))
        self.generator.load_state_dict(torch.load(self.ckpt_dir / "generator.pt", map_location=self.device))
        latent_bank_path = self.ckpt_dir / "latent_bank.npy"
        self.latent_bank = np.load(latent_bank_path).astype(np.float32) if latent_bank_path.exists() else None
        self.ae.eval()
        self.generator.eval()

    def generate(self, spec: PromptSpec) -> list[dict[str, float | str]]:
        n = spec.n_rows
        cond_mode = str(self.metadata.get("cond_mode", "prompt")).lower()
        if cond_mode == "zero":
            cond = np.zeros((n, int(self.metadata["cond_dim"])), dtype=np.float32)
        else:
            cond = self.preprocessor.condition_from_spec(spec, n)
        cond_t = torch.tensor(cond, dtype=torch.float32, device=self.device)
        noise = torch.randn((n, int(self.metadata["noise_dim"])), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            z_gan = self.generator(noise, cond_t)
            z = z_gan
            if self.latent_bank is not None and len(self.latent_bank) > 0:
                idx = np.random.randint(0, len(self.latent_bank), size=n)
                z_bank = torch.tensor(self.latent_bank[idx], dtype=torch.float32, device=self.device)
                alpha = float(self.metadata.get("latent_blend_alpha", 0.65))
                alpha = max(0.0, min(1.0, alpha))
                z = alpha * z_gan + (1.0 - alpha) * z_bank
            x_hat = self.ae.decode(z)
        arr = x_hat.detach().cpu().numpy()
        df = self.preprocessor.inverse_transform(arr)
        df = _align_to_training_distribution(df, self.preprocessor, spec)
        df = _apply_post_filters(df, spec)
        return df.to_dict(orient="records")


def _apply_post_filters(df, spec: PromptSpec):
    out = df.copy()
    for col, filt in spec.filters.items():
        if col not in out.columns:
            continue
        if filt["type"] == "exact":
            out[col] = filt["value"]
            continue
        if filt["type"] == "range":
            low = filt.get("min")
            high = filt.get("max")
            series = pd.to_numeric(out[col], errors="coerce")
            # Skip range clamping when the generated column is non-numeric.
            if not series.notna().any():
                continue
            if low is not None:
                series = series.where(series >= low, low)
            if high is not None:
                series = series.where(series <= high, high)
            out[col] = series
    return out


def _align_to_training_distribution(df, pre: TabularPreprocessor, spec: PromptSpec):
    out = df.copy()
    num_mean = getattr(pre, "num_mean", {})
    num_std = getattr(pre, "num_std", {})
    num_min = getattr(pre, "num_min", {})
    num_max = getattr(pre, "num_max", {})
    cat_probs = getattr(pre, "cat_probs", {})

    for col in pre.schema.numeric_cols:
        if col not in out.columns:
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        s_mean = float(vals.mean()) if vals.notna().any() else 0.0
        s_std = float(vals.std(ddof=0)) if vals.notna().any() else 1.0
        t_mean = float(num_mean.get(col, s_mean))
        t_std = float(num_std.get(col, s_std if s_std > 1e-8 else 1.0))
        if s_std <= 1e-8:
            aligned = pd.Series(np.full(len(vals), t_mean), index=vals.index)
        else:
            aligned = (vals - s_mean) / s_std * t_std + t_mean
        lo = num_min.get(col)
        hi = num_max.get(col)
        if lo is not None:
            aligned = aligned.where(aligned >= lo, lo)
        if hi is not None:
            aligned = aligned.where(aligned <= hi, hi)
        out[col] = aligned

    rng = np.random.default_rng(42)
    for col in pre.schema.categorical_cols:
        if col not in out.columns:
            continue
        filt = spec.filters.get(col)
        if filt and filt.get("type") == "exact":
            continue
        if col in spec.distribution_hints:
            continue
        probs = cat_probs.get(col)
        if not probs:
            continue
        cats = list(probs.keys())
        weights = np.array([float(probs[c]) for c in cats], dtype=np.float64)
        total = float(weights.sum())
        if total <= 0:
            continue
        weights = weights / total
        sampled = rng.choice(cats, size=len(out), replace=True, p=weights)
        out[col] = sampled

    return out
