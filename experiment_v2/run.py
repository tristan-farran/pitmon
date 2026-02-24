#!/usr/bin/env python3
"""Entry point for experiment v2.

Usage:
    python run.py                        # full run (compute + plot)
    python run.py --compute              # compute only
    python run.py --plot                 # plot from saved results
    python run.py --trials 50 --workers 4  # quick test run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the experiment package (and pitmon) are importable.
# Assumes directory layout: repo_root/experiment/ and repo_root/pitmon/
_this_dir = Path(__file__).resolve().parent
_repo_root = _this_dir.parent
sys.path.insert(0, str(_this_dir))
sys.path.insert(0, str(_repo_root))

from config import Config
from experiment import load_results, run_experiment, save_results
from plots import make_all_plots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PITMonitor vs River drift detectors on FriedmanDrift"
    )
    p.add_argument("--compute", action="store_true", help="Run experiment")
    p.add_argument("--plot", action="store_true", help="Generate plots")
    p.add_argument("--trials", type=int, default=10_000, help="MC trials per scenario")
    p.add_argument("--workers", type=int, default=8, help="Parallel workers")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="out", help="Output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        seed=args.seed,
        n_trials=args.trials,
        max_workers=args.workers,
        output_dir=args.output,
    )

    results_path = cfg.out_path / "results.json"

    if not args.compute and not args.plot:
        args.compute = True
        args.plot = True

    if args.compute:
        results = run_experiment(cfg)
        save_results(results, results_path)
        print(f"\nResults saved to {results_path}")

    if args.plot:
        if not results_path.exists():
            print(f"No results at {results_path}; run with --compute first.")
            sys.exit(1)
        results = load_results(results_path)
        make_all_plots(results, cfg.out_path)


if __name__ == "__main__":
    main()
