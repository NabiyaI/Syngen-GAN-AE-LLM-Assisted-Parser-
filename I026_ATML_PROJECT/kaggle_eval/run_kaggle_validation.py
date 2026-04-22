from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import kagglehub
import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gan_ae_full.trainer import TrainConfig, train_pipeline
from generator import build_backend, generate_synthetic, write_csv
from prompt_parser.parse_router import parse_user_prompt


@dataclass(frozen=True)
class KaggleCase:
    name: str
    dataset_ref: str
    profile: str
    csv_filename: str | None = None
    checkpoint_dir: str | None = None
    parse_mode: str = "hybrid"
    max_rows: int | None = None


CORE_CASES = [
    KaggleCase(
        name="stroke_prediction",
        dataset_ref="fedesoriano/stroke-prediction-dataset",
        profile="healthcare_v1",
        csv_filename="healthcare-dataset-stroke-data.csv",
        checkpoint_dir="checkpoints/full_gan_ae_kaggle_stroke",
        max_rows=12000,
    ),
    KaggleCase(
        name="heart_failure",
        dataset_ref="fedesoriano/heart-failure-prediction",
        profile="healthcare_v1",
        csv_filename="heart.csv",
        checkpoint_dir="checkpoints/full_gan_ae_kaggle_heart_failure",
        max_rows=5000,
    ),
    KaggleCase(
        name="pima_diabetes",
        dataset_ref="uciml/pima-indians-diabetes-database",
        profile="healthcare_v1",
        csv_filename="diabetes.csv",
        checkpoint_dir="checkpoints/full_gan_ae_kaggle_pima_diabetes",
        max_rows=5000,
    ),
    KaggleCase(
        name="credit_card_fraud",
        dataset_ref="mlg-ulb/creditcardfraud",
        profile="finance_v1",
        csv_filename="creditcard.csv",
        checkpoint_dir="checkpoints/full_gan_ae_kaggle_credit_card_fraud",
        max_rows=50000,
    ),
    KaggleCase(
        name="adult_income",
        dataset_ref="uciml/adult-census-income",
        profile="finance_v1",
        csv_filename="adult.csv",
        checkpoint_dir="checkpoints/full_gan_ae_kaggle_adult_income",
        max_rows=20000,
    ),
]

COMPREHENSIVE_EXTRA_CASES = [
    KaggleCase(
        name="telco_churn",
        dataset_ref="blastchar/telco-customer-churn",
        profile="ecommerce_v1",
        csv_filename="WA_Fn-UseC_-Telco-Customer-Churn.csv",
        checkpoint_dir="checkpoints/full_gan_ae_kaggle_telco_churn",
        max_rows=12000,
    ),
    KaggleCase(
        name="insurance_charges",
        dataset_ref="mirichoi0218/insurance",
        profile="finance_v1",
        csv_filename="insurance.csv",
        checkpoint_dir="checkpoints/full_gan_ae_kaggle_insurance",
        max_rows=7000,
    ),
    KaggleCase(
        name="mall_customers",
        dataset_ref="vjchoudhary7/customer-segmentation-tutorial-in-python",
        profile="ecommerce_v1",
        csv_filename="Mall_Customers.csv",
        checkpoint_dir="checkpoints/full_gan_ae_kaggle_mall_customers",
        max_rows=7000,
    ),
    KaggleCase(
        name="social_network_ads",
        dataset_ref="rakeshrau/social-network-ads",
        profile="ecommerce_v1",
        csv_filename="Social_Network_Ads.csv",
        checkpoint_dir="checkpoints/full_gan_ae_kaggle_social_ads",
        max_rows=5000,
    ),
    KaggleCase(
        name="house_sales",
        dataset_ref="harlfoxem/housesalesprediction",
        profile="finance_v1",
        csv_filename="kc_house_data.csv",
        checkpoint_dir="checkpoints/full_gan_ae_kaggle_house_sales",
        max_rows=20000,
    ),
    KaggleCase(
        name="wine_quality",
        dataset_ref="uciml/red-wine-quality-cortez-et-al-2009",
        profile="healthcare_v1",
        csv_filename="winequality-red.csv",
        checkpoint_dir="checkpoints/full_gan_ae_kaggle_wine_quality",
        max_rows=8000,
    ),
    KaggleCase(
        name="breast_cancer",
        dataset_ref="uciml/breast-cancer-wisconsin-data",
        profile="healthcare_v1",
        csv_filename="data.csv",
        checkpoint_dir="checkpoints/full_gan_ae_kaggle_breast_cancer",
        max_rows=7000,
    ),
    KaggleCase(
        name="iris",
        dataset_ref="uciml/iris",
        profile="healthcare_v1",
        csv_filename="Iris.csv",
        checkpoint_dir="checkpoints/full_gan_ae_kaggle_iris",
        max_rows=4000,
    ),
    KaggleCase(
        name="students_performance",
        dataset_ref="spscientist/students-performance-in-exams",
        profile="finance_v1",
        csv_filename="StudentsPerformance.csv",
        checkpoint_dir="checkpoints/full_gan_ae_kaggle_students_performance",
        max_rows=7000,
    ),
    KaggleCase(
        name="diamonds",
        dataset_ref="shivam2503/diamonds",
        profile="finance_v1",
        csv_filename="diamonds.csv",
        checkpoint_dir="checkpoints/full_gan_ae_kaggle_diamonds",
        max_rows=20000,
    ),
]

# High-quality defaults. Override via CLI if needed.
DEFAULT_AE_EPOCHS = 180
DEFAULT_GAN_EPOCHS = 320
DEFAULT_BATCH_SIZE = 512
DEFAULT_VARIANTS_PER_CASE = 3


def main() -> int:
    args = _build_parser().parse_args()
    cases = _select_cases(case_set=args.case_set, max_cases=args.max_cases)
    if args.preflight_only:
        return _run_preflight(expect_gpu=args.expect_gpu, cases=cases)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    cases_report: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        print(f"[INFO] ({idx}/{len(cases)}) Processing case: {case.name}")
        try:
            case_report = _process_case(case=case, args=args, out_root=out_root)
            cases_report.append(case_report)
            print(
                "[INFO] Completed"
                f" {case.name} with avg similarity {case_report['case_summary']['overall_similarity_avg']:.4f}"
            )
        except Exception as exc:  # pragma: no cover
            failed = {
                "case": case.name,
                "dataset_ref": case.dataset_ref,
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(limit=5),
            }
            cases_report.append(failed)
            print(f"[WARN] Failed case {case.name}: {exc}")
            if args.fail_fast:
                raise

    final_report = {
        "run_config": {
            "case_set": args.case_set,
            "max_cases": args.max_cases,
            "max_rows_override": args.max_rows_override,
            "ae_epochs": args.ae_epochs,
            "gan_epochs": args.gan_epochs,
            "batch_size": args.batch_size,
            "variants_per_case": args.variants_per_case,
            "base_seed": args.base_seed,
            "force_cpu": args.force_cpu,
        },
        "summary": _summarize(cases_report),
        "cases": cases_report,
    }

    report_tag = f"{args.case_set}_{len(cases)}cases_{args.variants_per_case}variants"
    report_json = out_root / f"kaggle_validation_{report_tag}_report.json"
    report_md = out_root / f"kaggle_validation_{report_tag}_report.md"
    report_json.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    report_md.write_text(_to_markdown(final_report), encoding="utf-8")

    # Keep backward-compatible report names as latest run pointers.
    (out_root / "kaggle_validation_report.json").write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    (out_root / "kaggle_validation_report.md").write_text(_to_markdown(final_report), encoding="utf-8")

    print(f"[DONE] Report written: {report_json}")
    print(f"[DONE] Report written: {report_md}")

    summary = final_report["summary"]
    if summary["successful_cases"] == 0:
        print("[DONE] No successful cases completed.")
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Kaggle -> Prompt -> Synthetic -> Validation benchmark pipeline.")
    p.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run runtime/data checks only; do not start training.",
    )
    p.add_argument(
        "--expect-gpu",
        action="store_true",
        help="Fail preflight if CUDA GPU is unavailable.",
    )
    p.add_argument(
        "--case-set",
        choices=["core", "comprehensive"],
        default="core",
        help="Choose benchmark scope. core=5 datasets, comprehensive=15 datasets.",
    )
    p.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional cap on number of selected benchmark cases.",
    )
    p.add_argument(
        "--max-rows-override",
        type=int,
        default=None,
        help="Optional global row cap applied to every dataset before training/evaluation.",
    )
    p.add_argument(
        "--ae-epochs",
        type=int,
        default=DEFAULT_AE_EPOCHS,
        help="Autoencoder epochs for each Kaggle case.",
    )
    p.add_argument(
        "--gan-epochs",
        type=int,
        default=DEFAULT_GAN_EPOCHS,
        help="Latent GAN epochs for each Kaggle case.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training.",
    )
    p.add_argument(
        "--variants-per-case",
        type=int,
        default=DEFAULT_VARIANTS_PER_CASE,
        help="How many synthetic datasets to generate and evaluate per Kaggle case.",
    )
    p.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed used for parsing/spec and synthetic generation variants.",
    )
    p.add_argument(
        "--output-dir",
        default="kaggle_eval/output",
        help="Directory for all generated synthetic CSVs and reports.",
    )
    p.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU training/evaluation even if CUDA is available.",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when a case fails instead of continuing.",
    )
    return p


def _select_cases(case_set: str, max_cases: int | None) -> list[KaggleCase]:
    selected = list(CORE_CASES)
    if case_set == "comprehensive":
        selected.extend(COMPREHENSIVE_EXTRA_CASES)
    if max_cases is not None:
        selected = selected[: max(0, max_cases)]
    return selected


def _run_preflight(expect_gpu: bool, cases: list[KaggleCase]) -> int:
    results: dict[str, Any] = {}
    try:
        import torch  # type: ignore

        results["torch_version"] = torch.__version__
        results["cuda_available"] = bool(torch.cuda.is_available())
        results["cuda_device_count"] = int(torch.cuda.device_count())
        results["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:
        print(json.dumps({"ok": False, "stage": "torch", "error": str(exc)}, indent=2))
        return 1

    if expect_gpu and not results["cuda_available"]:
        print(json.dumps({"ok": False, "stage": "gpu", "details": results}, indent=2))
        return 1

    # Verify Kaggle access using up to first 2 selected cases to catch repo-specific auth/data issues.
    download_checks: list[dict[str, Any]] = []
    for case in cases[:2]:
        try:
            base = Path(kagglehub.dataset_download(case.dataset_ref))
            resolved = _resolve_case_csv(base, case.csv_filename)
            download_checks.append(
                {
                    "case": case.name,
                    "dataset_ref": case.dataset_ref,
                    "ok": True,
                    "csv": str(resolved),
                }
            )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "stage": "kaggle_download",
                        "details": results,
                        "case": case.name,
                        "dataset_ref": case.dataset_ref,
                        "error": str(exc),
                    },
                    indent=2,
                )
            )
            return 1

    out_root = Path("kaggle_eval/output")
    out_root.mkdir(parents=True, exist_ok=True)
    test_file = out_root / "preflight_ok.txt"
    test_file.write_text("ok", encoding="utf-8")
    results["write_ok"] = test_file.exists()
    results["ready_to_start_training"] = True
    results["dataset_checks"] = download_checks
    print(json.dumps({"ok": True, "preflight": results}, indent=2))
    return 0


def _process_case(case: KaggleCase, args: argparse.Namespace, out_root: Path) -> dict[str, Any]:
    source_csv = _download_case(case)
    source_df = _load_df(source_csv)
    max_rows = args.max_rows_override if args.max_rows_override is not None else case.max_rows
    original_df = _cap_rows(source_df, max_rows)

    train_csv = source_csv
    if len(original_df) < len(source_df):
        train_csv = out_root / f"{case.name}_train_input.csv"
        original_df.to_csv(train_csv, index=False)

    ckpt_dir = case.checkpoint_dir or f"checkpoints/full_gan_ae_kaggle_{case.name}"
    device = "cpu" if args.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = TrainConfig(
        ae_epochs=args.ae_epochs,
        gan_epochs=args.gan_epochs,
        batch_size=args.batch_size,
        device=device,
        seed=args.base_seed,
    )
    train_report = train_pipeline(str(train_csv), ckpt_dir, train_cfg)

    backend = build_backend("gan_ae", ckpt_dir)
    variant_reports: list[dict[str, Any]] = []
    for variant_idx in range(args.variants_per_case):
        variant_num = variant_idx + 1
        seed = args.base_seed + variant_idx
        prompt = _derive_prompt(profile=case.profile, n_rows=len(original_df), seed=seed)
        spec = parse_user_prompt(prompt, mode=case.parse_mode)
        spec.n_rows = len(original_df)
        spec.seed = seed

        synthetic_rows = generate_synthetic(spec, backend=backend)
        synthetic_path = out_root / f"{case.name}_synthetic_v{variant_num}.csv"
        write_csv(synthetic_rows, str(synthetic_path))
        synthetic_df = pd.read_csv(synthetic_path)
        metrics = _compare_datasets(original_df, synthetic_df)

        variant_reports.append(
            {
                "variant": variant_num,
                "seed": seed,
                "prompt": prompt,
                "synthetic_csv": str(synthetic_path),
                "metrics": metrics,
            }
        )

    case_summary = _summarize_case_variants(variant_reports)
    return {
        "case": case.name,
        "dataset_ref": case.dataset_ref,
        "status": "ok",
        "rows_used": int(len(original_df)),
        "source_csv": str(source_csv),
        "train_csv": str(train_csv),
        "checkpoint_dir": ckpt_dir,
        "profile": case.profile,
        "parse_mode": case.parse_mode,
        "training_report": train_report,
        "case_summary": case_summary,
        "variants": variant_reports,
    }


def _download_case(case: KaggleCase) -> Path:
    base = Path(kagglehub.dataset_download(case.dataset_ref))
    return _resolve_case_csv(base, case.csv_filename)


def _resolve_case_csv(dataset_dir: Path, preferred_csv: str | None) -> Path:
    if preferred_csv:
        direct = dataset_dir / preferred_csv
        if direct.exists():
            return direct

    csv_files = sorted(dataset_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {dataset_dir}")

    if preferred_csv:
        pref = preferred_csv.lower()
        for candidate in csv_files:
            if candidate.name.lower() == pref:
                return candidate

    # Fallback: choose largest CSV in the dataset package.
    csv_files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return csv_files[0]


def _load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        if df[col].dtype == object:
            series = df[col].astype(str).str.strip()
            # Convert numeric-like string columns (e.g., Telco TotalCharges)
            maybe = pd.to_numeric(series.replace("", np.nan), errors="coerce")
            if maybe.notna().mean() > 0.9:
                df[col] = maybe.fillna(maybe.median())
            else:
                df[col] = series
    return df


def _cap_rows(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42).reset_index(drop=True)


def _derive_prompt(profile: str, n_rows: int, seed: int) -> str:
    if profile == "healthcare_v1":
        return (
            f"Generate {n_rows} healthcare records age 30-75 with glucose 90-220 and bmi 19-36, "
            f"mostly non smoker; industry=healthcare_v1; strict=false; seed={seed}"
        )
    if profile == "finance_v1":
        return (
            f"Generate {n_rows} finance records with credit score 600-790, income 30000-180000, "
            f"debt_to_income 0.08-0.45, mostly non defaulted; industry=finance_v1; strict=false; seed={seed}"
        )
    return (
        f"Generate {n_rows} ecommerce records with orders 1-16, monthly spend 40-2800, "
        f"conversion_rate 0.02-0.20, mostly mobile users; industry=ecommerce_v1; strict=false; seed={seed}"
    )


def _compare_datasets(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> dict[str, Any]:
    common = [c for c in real_df.columns if c in syn_df.columns]
    numeric: list[str] = []
    categorical: list[str] = []
    for col in common:
        real_num = pd.to_numeric(real_df[col], errors="coerce")
        syn_num = pd.to_numeric(syn_df[col], errors="coerce")
        if real_num.notna().mean() >= 0.8 and syn_num.notna().mean() >= 0.8:
            numeric.append(col)
        else:
            categorical.append(col)

    num_scores: list[float] = []
    num_details: list[dict[str, Any]] = []
    for col in numeric:
        a = pd.to_numeric(real_df[col], errors="coerce").dropna().to_numpy()
        b = pd.to_numeric(syn_df[col], errors="coerce").dropna().to_numpy()
        if len(a) < 5 or len(b) < 5:
            continue
        ks = float(ks_2samp(a, b).statistic)
        wd = float(wasserstein_distance(a, b))
        a_std = float(np.std(a))
        scale = a_std if a_std > 1e-8 else 1.0
        wd_norm = wd / scale

        mean_gap_rel = abs(float(np.mean(a)) - float(np.mean(b))) / scale
        std_gap_rel = abs(a_std - float(np.std(b))) / scale

        score = max(
            0.0,
            1.0
            - min(
                1.0,
                0.40 * ks
                + 0.30 * min(1.0, wd_norm)
                + 0.15 * min(1.0, mean_gap_rel)
                + 0.15 * min(1.0, std_gap_rel),
            ),
        )
        num_scores.append(score)
        num_details.append(
            {
                "column": col,
                "ks": ks,
                "wasserstein_norm": wd_norm,
                "mean_gap_rel": mean_gap_rel,
                "std_gap_rel": std_gap_rel,
                "score": score,
            }
        )

    cat_scores: list[float] = []
    cat_details: list[dict[str, Any]] = []
    for col in categorical:
        a = real_df[col].astype(str).str.lower().value_counts(normalize=True)
        b = syn_df[col].astype(str).str.lower().value_counts(normalize=True)
        idx = sorted(set(a.index).union(set(b.index)))
        av = np.array([float(a.get(i, 0.0)) for i in idx])
        bv = np.array([float(b.get(i, 0.0)) for i in idx])
        tv = 0.5 * float(np.abs(av - bv).sum())
        score = max(0.0, 1.0 - min(1.0, tv))
        cat_scores.append(score)
        cat_details.append({"column": col, "tv_distance": tv, "score": score})

    missing_gaps = []
    missing_details = []
    for col in common:
        real_missing = float(real_df[col].isna().mean())
        syn_missing = float(syn_df[col].isna().mean())
        gap = abs(real_missing - syn_missing)
        missing_gaps.append(gap)
        missing_details.append({"column": col, "real_missing": real_missing, "syn_missing": syn_missing, "gap": gap})
    missing_score = max(0.0, 1.0 - float(np.mean(missing_gaps))) if missing_gaps else 0.0

    corr_score = 0.0
    if len(numeric) >= 2:
        real_num = real_df[numeric].apply(pd.to_numeric, errors="coerce")
        syn_num = syn_df[numeric].apply(pd.to_numeric, errors="coerce")

        real_num = real_num.fillna(real_num.median(numeric_only=True)).fillna(0.0)
        syn_num = syn_num.fillna(syn_num.median(numeric_only=True)).fillna(0.0)

        corr_a = real_num.corr().fillna(0.0).to_numpy(dtype=float)
        corr_b = syn_num.corr().fillna(0.0).to_numpy(dtype=float)
        fro_norm = float(np.linalg.norm(corr_a - corr_b, ord="fro"))
        scale = float(np.sqrt(corr_a.size)) if corr_a.size > 0 else 1.0
        corr_diff = fro_norm / max(1.0, scale)
        corr_score = max(0.0, 1.0 - min(1.0, corr_diff))

    weighted_parts: list[tuple[float, float]] = []
    if num_scores:
        weighted_parts.append((0.55, float(np.mean(num_scores))))
    if cat_scores:
        weighted_parts.append((0.25, float(np.mean(cat_scores))))
    weighted_parts.append((0.10, missing_score))
    if len(numeric) >= 2:
        weighted_parts.append((0.10, corr_score))

    if weighted_parts:
        weight_sum = float(sum(w for w, _ in weighted_parts))
        overall = float(sum(w * s for w, s in weighted_parts) / weight_sum)
    else:
        overall = 0.0

    return {
        "common_columns": len(common),
        "numeric_columns": len(numeric),
        "categorical_columns": len(categorical),
        "numeric_score_avg": float(np.mean(num_scores)) if num_scores else 0.0,
        "categorical_score_avg": float(np.mean(cat_scores)) if cat_scores else 0.0,
        "missingness_score": missing_score,
        "correlation_score": corr_score,
        "overall_similarity_score": overall,
        "numeric_details": num_details,
        "categorical_details": cat_details,
        "missingness_details": missing_details,
    }


def _summarize_case_variants(variant_reports: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(v["metrics"]["overall_similarity_score"]) for v in variant_reports]
    if not scores:
        return {
            "variants": 0,
            "overall_similarity_avg": 0.0,
            "overall_similarity_std": 0.0,
            "best_variant": None,
            "worst_variant": None,
        }

    best = max(variant_reports, key=lambda v: float(v["metrics"]["overall_similarity_score"]))
    worst = min(variant_reports, key=lambda v: float(v["metrics"]["overall_similarity_score"]))
    return {
        "variants": len(variant_reports),
        "overall_similarity_avg": float(np.mean(scores)),
        "overall_similarity_std": float(np.std(scores)),
        "best_variant": {
            "variant": int(best["variant"]),
            "score": float(best["metrics"]["overall_similarity_score"]),
            "seed": int(best["seed"]),
        },
        "worst_variant": {
            "variant": int(worst["variant"]),
            "score": float(worst["metrics"]["overall_similarity_score"]),
            "seed": int(worst["seed"]),
        },
    }


def _summarize(cases_report: list[dict[str, Any]]) -> dict[str, Any]:
    success = [c for c in cases_report if c.get("status") == "ok"]
    failed = [c for c in cases_report if c.get("status") != "ok"]

    case_scores = [float(c["case_summary"]["overall_similarity_avg"]) for c in success]
    variant_scores = [
        float(v["metrics"]["overall_similarity_score"])
        for c in success
        for v in c.get("variants", [])
    ]

    best_case = (
        max(success, key=lambda c: float(c["case_summary"]["overall_similarity_avg"]))["case"] if success else None
    )
    worst_case = (
        min(success, key=lambda c: float(c["case_summary"]["overall_similarity_avg"]))["case"] if success else None
    )

    return {
        "total_cases": len(cases_report),
        "successful_cases": len(success),
        "failed_cases": len(failed),
        "total_synthetic_datasets": len(variant_scores),
        "overall_similarity_avg": float(np.mean(variant_scores)) if variant_scores else 0.0,
        "overall_similarity_std": float(np.std(variant_scores)) if variant_scores else 0.0,
        "case_similarity_avg": float(np.mean(case_scores)) if case_scores else 0.0,
        "best_case": best_case,
        "worst_case": worst_case,
        "failed_case_names": [c["case"] for c in failed],
    }


def _to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = ["# Syngen Project Report", ""]

    lines.append("## What This Project Does")
    lines.append("Syngen is a prompt-to-dataset system for synthetic tabular data generation.")
    lines.append(
        "Given a plain-language request, it produces a CSV with realistic rows that follow "
        "domain constraints, value ranges, and category tendencies."
    )
    lines.append("")

    lines.append("## What GAN+AE Actually Does In This Project")
    lines.append(
        "- Autoencoder (AE): learns a compact latent representation of each training row "
        "and a decoder that reconstructs rows back into table space."
    )
    lines.append(
        "- GAN in latent space: the generator creates latent vectors that resemble "
        "AE-encoded real rows, while the discriminator separates real encoded latents "
        "from generated latents."
    )
    lines.append("- Row synthesis: generated latent vectors are decoded by the AE decoder into full synthetic rows.")
    lines.append("- Post-processing: decoded values are mapped to typed tabular columns and written as a CSV.")
    lines.append("")

    lines.append("## End-to-End System Flow")
    lines.append("1. Parse user prompt into a structured generation spec.")
    lines.append("2. Route to the domain profile and matching trained checkpoint.")
    lines.append("3. Sample synthetic rows through GAN generation and AE decoding.")
    lines.append("4. Apply schema-aware cleanup and type conversion.")
    lines.append("5. Export final rows as CSV.")
    lines.append("6. Optionally run benchmark evaluation to measure fidelity.")
    lines.append("")

    lines.append("## Role Of Kaggle Evaluation In This Report")
    lines.append("Kaggle evaluation is a validation layer, not the core project function.")
    lines.append("It is included to measure how closely generated data matches public real datasets.")
    lines.append("")

    cfg = report.get("run_config", {})
    lines.append("## Run Configuration")
    lines.append(f"- Case set: {cfg.get('case_set')}")
    lines.append(f"- Max cases: {cfg.get('max_cases')}")
    lines.append(f"- Max rows override: {cfg.get('max_rows_override')}")
    lines.append(f"- AE epochs: {cfg.get('ae_epochs')}")
    lines.append(f"- GAN epochs: {cfg.get('gan_epochs')}")
    lines.append(f"- Batch size: {cfg.get('batch_size')}")
    lines.append(f"- Variants per case: {cfg.get('variants_per_case')}")
    lines.append("")

    s = report["summary"]
    lines.append("## Benchmark Snapshot")
    lines.append(f"- Total cases: {s['total_cases']}")
    lines.append(f"- Successful cases: {s['successful_cases']}")
    lines.append(f"- Failed cases: {s['failed_cases']}")
    lines.append(f"- Total synthetic datasets: {s['total_synthetic_datasets']}")
    lines.append(f"- Overall similarity average: {s['overall_similarity_avg']:.4f}")
    lines.append(f"- Overall similarity std: {s['overall_similarity_std']:.4f}")
    lines.append(f"- Best case: {s['best_case']}")
    lines.append(f"- Worst case: {s['worst_case']}")
    lines.append("")

    lines.append("## Case Overview")
    lines.append("| Case | Status | Rows | Variants | Avg similarity | Best variant | Worst variant |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for case in report["cases"]:
        if case.get("status") != "ok":
            lines.append(f"| {case['case']} | failed | - | - | - | - | - |")
            continue
        summary = case["case_summary"]
        best = summary.get("best_variant") or {}
        worst = summary.get("worst_variant") or {}
        lines.append(
            "| "
            f"{case['case']} | ok | {case['rows_used']} | {summary['variants']} | "
            f"{summary['overall_similarity_avg']:.4f} | "
            f"v{best.get('variant')} ({best.get('score', 0.0):.4f}) | "
            f"v{worst.get('variant')} ({worst.get('score', 0.0):.4f}) |"
        )
    lines.append("")

    failed_cases = [c for c in report["cases"] if c.get("status") != "ok"]
    if failed_cases:
        lines.append("## Failed Cases")
        for case in failed_cases:
            lines.append(f"- {case['case']}: {case.get('error', 'unknown error')}")
        lines.append("")

    for case in report["cases"]:
        if case.get("status") != "ok":
            continue
        lines.append(f"## {case['case']}")
        lines.append(f"- Dataset: {case['dataset_ref']}")
        lines.append(f"- Source CSV: {case['source_csv']}")
        lines.append(f"- Train CSV: {case['train_csv']}")
        lines.append(f"- Profile: {case['profile']}")
        lines.append(f"- Parse mode: {case['parse_mode']}")
        lines.append(f"- Checkpoint: {case['checkpoint_dir']}")
        tr = case.get("training_report", {})
        lines.append(f"- AE final MSE: {float(tr.get('ae_final_mse', 0.0)):.6f}")
        lines.append(f"- GAN final D loss: {float(tr.get('gan_final_d_loss', 0.0)):.6f}")
        lines.append(f"- GAN final G loss: {float(tr.get('gan_final_g_loss', 0.0)):.6f}")
        lines.append("")
        lines.append("| Variant | Seed | Similarity | Numeric | Categorical | Missingness | Correlation |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for variant in case["variants"]:
            m = variant["metrics"]
            lines.append(
                "| "
                f"{variant['variant']} | {variant['seed']} | {m['overall_similarity_score']:.4f} | "
                f"{m['numeric_score_avg']:.4f} | {m['categorical_score_avg']:.4f} | "
                f"{m['missingness_score']:.4f} | {m['correlation_score']:.4f} |"
            )
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
