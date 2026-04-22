from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from checkpoint_registry import (
    DEFAULT_REGISTRY_PATH,
    load_registry,
    resolve_checkpoint_for_profile,
    validate_checkpoint_dir,
)
from generator import build_backend, generate_synthetic, write_csv
from prompt_parser.parse_router import parse_user_prompt
from prompt_parser.parser import PromptParseException


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse synthetic data prompts into structured constraints.")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument(
        "--out",
        help="CSV output path. Defaults to output/synthetic_data.csv",
    )
    parser.add_argument(
        "--backend",
        default="baseline",
        choices=["baseline", "gan_ae"],
        help="Generation backend",
    )
    parser.add_argument(
        "--parse-mode",
        default="hybrid",
        choices=["rules", "llm", "hybrid"],
        help="Prompt parsing mode",
    )
    parser.add_argument("--ckpt-dir", help="Checkpoint directory for gan_ae backend")
    parser.add_argument(
        "--ckpt-registry",
        default=DEFAULT_REGISTRY_PATH,
        help="Checkpoint registry JSON for domain->checkpoint mapping",
    )
    return parser


def _timestamped_csv_path(out_arg: str | None) -> str:
    raw = out_arg.strip() if out_arg else "output/synthetic_data.csv"
    path = Path(raw)
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
    return str(path)


def main() -> int:
    args = build_parser().parse_args()
    try:
        spec = parse_user_prompt(args.prompt, mode=args.parse_mode)
        resolved_ckpt = args.ckpt_dir
        if args.backend == "gan_ae" and not resolved_ckpt:
            registry = load_registry(args.ckpt_registry)
            resolved_ckpt = resolve_checkpoint_for_profile(
                spec.target_dataset_profile,
                registry,
                prompt_text=args.prompt,
            )

        if args.backend == "gan_ae":
            missing = validate_checkpoint_dir(str(resolved_ckpt))
            if missing:
                raise FileNotFoundError(
                    "GAN+AE checkpoint is incomplete.\n"
                    f"Path: {resolved_ckpt}\n"
                    "Missing files:\n- " + "\n- ".join(missing)
                )

        backend = build_backend(args.backend, resolved_ckpt)
        out_path = _timestamped_csv_path(args.out)
        rows = generate_synthetic(spec, backend=backend)
        csv_path = write_csv(rows, out_path)
        rows_preview = rows[:3]

        payload = {
            "ok": True,
            "spec": {
                "n_rows": spec.n_rows,
                "filters": spec.filters,
                "distribution_hints": spec.distribution_hints,
                "priority_rules": spec.priority_rules,
                "strict_mode": spec.strict_mode,
                "target_dataset_profile": spec.target_dataset_profile,
                "seed": spec.seed,
                "warnings": spec.warnings,
            },
            "backend": args.backend,
            "parse_mode": args.parse_mode,
            "checkpoint_dir": resolved_ckpt,
            "output_csv": csv_path,
            "preview_rows": rows_preview,
        }
        print(json.dumps(payload, indent=2))
        return 0
    except (ValueError, FileNotFoundError) as exc:
        payload = {
            "ok": False,
            "error": {
                "message": str(exc),
                "conflicting_fields": [],
                "offending_clauses": [],
                "suggested_prompt": "Use --backend baseline or provide valid --ckpt-dir for gan_ae.",
                "warnings": [],
            },
        }
        print(json.dumps(payload, indent=2))
        return 1
    except PromptParseException as exc:
        payload = {
            "ok": False,
            "error": {
                "message": exc.report.message,
                "conflicting_fields": exc.report.conflicting_fields,
                "offending_clauses": exc.report.offending_clauses,
                "suggested_prompt": exc.report.suggested_prompt,
                "warnings": exc.report.warnings,
            },
        }
        print(json.dumps(payload, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
