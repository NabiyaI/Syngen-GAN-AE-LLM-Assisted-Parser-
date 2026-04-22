from __future__ import annotations

import sys
from pathlib import Path

# Ensure top-level modules (e.g., generator.py, checkpoint_registry.py) are importable
# in local and CI pytest runs.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

