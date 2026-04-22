from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prompt_parser.models import PromptSpec


@dataclass
class SchemaInfo:
    columns: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    categorical_values: dict[str, list[str]]


class TabularPreprocessor:
    def __init__(self) -> None:
        self.schema: SchemaInfo | None = None
        self.num_mean: dict[str, float] = {}
        self.num_std: dict[str, float] = {}
        self.num_min: dict[str, float] = {}
        self.num_max: dict[str, float] = {}
        self.cat_probs: dict[str, dict[str, float]] = {}
        self.feature_slices: dict[str, tuple[int, int]] = {}
        self.feature_dim: int = 0

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        clean = df.copy()
        numeric_cols = [c for c in clean.columns if pd.api.types.is_numeric_dtype(clean[c])]
        categorical_cols = [c for c in clean.columns if c not in numeric_cols]

        cat_values: dict[str, list[str]] = {}
        for col in categorical_cols:
            series = clean[col].fillna("unknown").astype(str).str.strip().str.lower()
            values = sorted(set(series.tolist()))
            if "unknown" not in values:
                values.append("unknown")
            cat_values[col] = values
            clean[col] = series
            probs = series.value_counts(normalize=True)
            self.cat_probs[col] = {str(k): float(v) for k, v in probs.items()}

        for col in numeric_cols:
            series = pd.to_numeric(clean[col], errors="coerce")
            mean = float(series.mean()) if not math.isnan(float(series.mean())) else 0.0
            std = float(series.std(ddof=0)) if not math.isnan(float(series.std(ddof=0))) else 1.0
            if std <= 1e-8:
                std = 1.0
            self.num_mean[col] = mean
            self.num_std[col] = std
            filled = series.fillna(mean)
            self.num_min[col] = float(filled.min()) if len(filled) else mean
            self.num_max[col] = float(filled.max()) if len(filled) else mean
            clean[col] = series.fillna(mean)

        self.schema = SchemaInfo(
            columns=list(clean.columns),
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            categorical_values=cat_values,
        )
        self._build_feature_slices()
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.schema:
            raise ValueError("Preprocessor not fitted.")
        clean = df.copy()
        features: list[np.ndarray] = []
        for col in self.schema.numeric_cols:
            series = pd.to_numeric(clean[col], errors="coerce").fillna(self.num_mean[col]).astype(float)
            z = (series.to_numpy(dtype=np.float32) - self.num_mean[col]) / self.num_std[col]
            features.append(z.reshape(-1, 1))
        for col in self.schema.categorical_cols:
            values = self.schema.categorical_values[col]
            index = {v: i for i, v in enumerate(values)}
            series = clean[col].fillna("unknown").astype(str).str.strip().str.lower()
            arr = np.zeros((len(series), len(values)), dtype=np.float32)
            for r, val in enumerate(series.tolist()):
                arr[r, index.get(val, index["unknown"])] = 1.0
            features.append(arr)
        if not features:
            return np.zeros((len(clean), 0), dtype=np.float32)
        return np.concatenate(features, axis=1).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> pd.DataFrame:
        if not self.schema:
            raise ValueError("Preprocessor not fitted.")
        rows = x.shape[0]
        out: dict[str, list[Any]] = {col: [] for col in self.schema.columns}
        for r in range(rows):
            for col in self.schema.numeric_cols:
                s, e = self.feature_slices[col]
                z = float(x[r, s:e][0])
                val = z * self.num_std[col] + self.num_mean[col]
                out[col].append(val)
            for col in self.schema.categorical_cols:
                s, e = self.feature_slices[col]
                probs = x[r, s:e]
                idx = int(np.argmax(probs))
                values = self.schema.categorical_values[col]
                out[col].append(values[idx])
        return pd.DataFrame(out, columns=self.schema.columns)

    def condition_from_spec(self, spec: PromptSpec, n_rows: int) -> np.ndarray:
        if not self.schema:
            raise ValueError("Preprocessor not fitted.")
        cond = np.zeros((n_rows, self.feature_dim), dtype=np.float32)
        for col in self.schema.numeric_cols:
            filt = spec.filters.get(col)
            if not filt:
                continue
            if filt["type"] == "exact":
                raw = filt["value"]
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    s = str(raw).strip().lower()
                    if s in {"yes", "true"}:
                        v = 1.0
                    elif s in {"no", "false"}:
                        v = 0.0
                    else:
                        continue
            elif filt["type"] == "range":
                low = filt.get("min")
                high = filt.get("max")
                if low is not None and high is not None:
                    v = (float(low) + float(high)) / 2.0
                elif low is not None:
                    v = float(low)
                elif high is not None:
                    v = float(high)
                else:
                    continue
            else:
                continue
            z = (v - self.num_mean[col]) / self.num_std[col]
            s, e = self.feature_slices[col]
            cond[:, s:e] = np.float32(z)

        for col in self.schema.categorical_cols:
            s, e = self.feature_slices[col]
            values = self.schema.categorical_values[col]
            value_to_idx = {v: i for i, v in enumerate(values)}
            hint = spec.distribution_hints.get(col, {})
            filt = spec.filters.get(col)
            if filt and filt["type"] == "exact":
                idx = value_to_idx.get(str(filt["value"]).lower())
                if idx is not None:
                    cond[:, s:e] = 0.0
                    cond[:, s + idx] = 1.0
                    continue
            if hint:
                probs = np.zeros((len(values),), dtype=np.float32)
                for val, p in hint.items():
                    idx = value_to_idx.get(str(val).lower())
                    if idx is not None:
                        probs[idx] = float(p)
                total = float(probs.sum())
                if total > 0:
                    probs = probs / total
                    cond[:, s:e] = probs.reshape(1, -1)
        return cond

    def _build_feature_slices(self) -> None:
        if not self.schema:
            raise ValueError("Preprocessor not fitted.")
        idx = 0
        self.feature_slices = {}
        for col in self.schema.numeric_cols:
            self.feature_slices[col] = (idx, idx + 1)
            idx += 1
        for col in self.schema.categorical_cols:
            width = len(self.schema.categorical_values[col])
            self.feature_slices[col] = (idx, idx + width)
            idx += width
        self.feature_dim = idx

    def save(self, out_dir: str) -> None:
        path = Path(out_dir) / "preprocessor.pkl"
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(out_dir: str) -> "TabularPreprocessor":
        path = Path(out_dir) / "preprocessor.pkl"
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, TabularPreprocessor):
            raise TypeError("Invalid preprocessor object in checkpoint.")
        return obj
