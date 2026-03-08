"""CLI entry point for PITMonitor additional experiments.

Usage
-----
python run_experiment.py
python run_experiment.py --plot
python run_experiment.py --output out
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_REPO_ROOT))

from config import VerificationConfig
from experiment import run_all
from plots import make_all_plots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PITMonitor additional experiments")
    p.add_argument("--compute", action="store_true", help="Run experiment")
    p.add_argument("--plot", action="store_true", help="Generate plots from results")
    p.add_argument("--output", type=str, default="out", help="Output directory")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--bins", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = VerificationConfig(seed=args.seed, alpha=args.alpha, n_bins=args.bins)

    # Default: run everything when no flags are given
    if not any([args.compute, args.plot]):
        args.compute = True
        args.plot = True

    out_dir = (
        (_THIS_DIR / args.output).resolve()
        if not Path(args.output).is_absolute()
        else Path(args.output)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"

    # ── Step 1: Compute ──────────────────────────────────────────────
    if args.compute:
        results = run_all(cfg)

        with results_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"Saved additional experiment results to {results_path}")
        print("Summary:")
        for k, v in results["summary"].items():
            print(f"  {k}: {v}")

    # ── Step 2: Plot ─────────────────────────────────────────────────
    if args.plot:
        if not results_path.exists():
            print(f"No results at {results_path}; run with --compute first.")
            sys.exit(1)
        with results_path.open("r", encoding="utf-8") as f:
            results = json.load(f)
        make_all_plots(results, out_dir)


if __name__ == "__main__":
    main()
