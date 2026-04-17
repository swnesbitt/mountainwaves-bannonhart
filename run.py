#!/usr/bin/env python3
"""Launcher for the Mountain Waves Dash app.

Mountain Waves is a Rust + Python port of Dr. Robert E. (Bob) Hart's 1995
MATLAB mountain-wave model (https://moe.met.fsu.edu/~rhart/mtnwave.html).

Usage (with uv — recommended):
    uv run python run.py                  # start the web UI on http://127.0.0.1:8050
    uv run python run.py --port 9000      # alternate port
    uv run python run.py --host 0.0.0.0   # expose on LAN

Plain Python also works once dependencies are installed:
    python run.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the in-tree Python package importable without installing.
ROOT = Path(__file__).resolve().parent
PKG_DIR = ROOT / "python"
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from mountain_waves.app import create_app  # noqa: E402
from mountain_waves import backend_name     # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    print(f"[mountain-waves] compute backend: {backend_name()}")
    if backend_name() == "python":
        print(
            "[mountain-waves] running on the pure-Python reference solver.\n"
            "                 Build the Rust core with `uv run maturin develop --release --uv`\n"
            "                 (or `maturin develop --release`) for a ~20x speedup."
        )

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
