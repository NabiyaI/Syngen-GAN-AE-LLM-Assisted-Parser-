from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_main_cli_smoke_generates_csv(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_base = tmp_path / "synthetic.csv"
    cmd = [
        sys.executable,
        str(repo_root / "main.py"),
        "--prompt",
        "Generate 25 finance customers with credit score 650-700",
        "--backend",
        "baseline",
        "--parse-mode",
        "rules",
        "--out",
        str(out_base),
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    csv_path = Path(payload["output_csv"])
    assert csv_path.exists()
    assert csv_path.suffix.lower() == ".csv"

