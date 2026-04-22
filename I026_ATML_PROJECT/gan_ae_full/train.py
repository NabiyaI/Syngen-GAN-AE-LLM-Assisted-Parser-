from __future__ import annotations

import argparse
import json
from pathlib import Path

from .trainer import TrainConfig, train_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train full AutoEncoder + Latent GAN on tabular CSV.")
    p.add_argument("--csv", required=True, help="Input CSV path")
    p.add_argument("--out", default="checkpoints/full_gan_ae", help="Output checkpoint directory")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--latent-dim", type=int, default=None)
    p.add_argument("--noise-dim", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--ae-epochs", type=int, default=None)
    p.add_argument("--gan-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--preset",
        help="Optional JSON preset path (e.g., configs/train_balanced.json). CLI args override preset.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    preset: dict = {}
    if args.preset:
        p = Path(args.preset)
        if not p.exists():
            raise FileNotFoundError(f"Preset file not found: {args.preset}")
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Preset JSON must be an object.")
        preset = raw

    config = TrainConfig(
        latent_dim=int(args.latent_dim if args.latent_dim is not None else preset.get("latent_dim", 16)),
        noise_dim=int(args.noise_dim if args.noise_dim is not None else preset.get("noise_dim", 16)),
        hidden_dim=int(args.hidden_dim if args.hidden_dim is not None else preset.get("hidden_dim", 128)),
        ae_epochs=int(args.ae_epochs if args.ae_epochs is not None else preset.get("ae_epochs", 40)),
        gan_epochs=int(args.gan_epochs if args.gan_epochs is not None else preset.get("gan_epochs", 60)),
        batch_size=int(args.batch_size if args.batch_size is not None else preset.get("batch_size", 128)),
        device=args.device,
        seed=int(args.seed if args.seed is not None else preset.get("seed", 42)),
    )
    report = train_pipeline(args.csv, args.out, config)
    print(json.dumps({"ok": True, "checkpoint_dir": args.out, "report": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
