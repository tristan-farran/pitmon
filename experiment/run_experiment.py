"""Entry point for the PITMonitor experiment.

Typical workflow
----------------
1. Train the model once (saves weights to ``out/model.pkl``):

       python run.py --train

2. Run the full experiment and generate all plots:

       python run.py --compute --plot

3. Quick smoke-test with fewer trials:

       python run.py --train --compute --plot --trials 50 --workers 4

The train / compute / plot steps are independent and can be run in isolation
or combined.  If none of ``--train``, ``--compute``, ``--plot`` are given,
all three run automatically.

The n_bins sensitivity sweep and single-run visualization artifacts are
produced automatically as part of ``--compute``; no extra flag is needed.

Flags
-----
--train         Train ProbabilisticMLP and save bundle (skipped if bundle exists)
--force-train   Re-train even if a bundle already exists
--compute       Run the Monte-Carlo experiment and save results.json
--plot          Load results.json and regenerate all figures
--trials N      Override number of MC trials (useful for quick tests)
--epochs N      Override training epochs
--workers N     Thread pool size for parallel trial execution
--seed N        Master RNG seed
--output DIR    Output directory (default: out)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_this_dir = Path(__file__).resolve().parent
_repo_root = _this_dir.parent
sys.path.insert(0, str(_this_dir))
sys.path.insert(0, str(_repo_root))

from config import Config
from data import generate_stream
from experiment import load_results, run_experiment, save_results
from model import load_bundle, save_bundle, train_model
from plots import make_all_plots


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    p = argparse.ArgumentParser(
        description="PITMonitor vs River drift detectors on FriedmanDrift",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Flags")[0],
    )
    p.add_argument("--train", action="store_true", help="Train and save the model")
    p.add_argument(
        "--force-train", action="store_true", help="Re-train even if bundle exists"
    )
    p.add_argument("--compute", action="store_true", help="Run MC experiment")
    p.add_argument(
        "--plot", action="store_true", help="Generate plots from saved results"
    )
    p.add_argument("--trials", type=int, default=10_000, help="MC trials per scenario")
    p.add_argument("--epochs", type=int, default=500, help="NN training epochs")
    p.add_argument("--workers", type=int, default=8, help="Parallel workers")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--output", type=str, default="out", help="Output directory")
    return p.parse_args()


def main() -> None:
    """Orchestrate training, computation, and plotting."""
    args = parse_args()
    cfg = Config(
        seed=args.seed,
        epochs=args.epochs,
        n_trials=args.trials,
        max_workers=args.workers,
        output_dir=args.output,
    )

    # Default: run everything when no flags are given
    if not any([args.train, args.compute, args.plot]):
        args.train = True
        args.compute = True
        args.plot = True

    results_path = cfg.out_path / "results.json"

    # ── Step 1: Train ────────────────────────────────────────────────
    if args.train or args.force_train:
        if cfg.bundle_path.exists() and not args.force_train:
            print(
                f"Model bundle already exists at {cfg.bundle_path}  "
                f"(use --force-train to retrain)"
            )
        else:
            print("Training ProbabilisticMLP …")
            drift_type, tw = cfg.drift_scenarios[0]
            np.random.seed(cfg.seed)
            X, y = generate_stream(
                cfg, drift_type=drift_type, transition_window=tw, seed=cfg.seed
            )
            X_train, y_train = X[: cfg.n_train], y[: cfg.n_train]
            bundle = train_model(
                X_train, y_train, epochs=cfg.epochs, lr=cfg.lr, seed=cfg.seed
            )
            save_bundle(bundle, cfg.bundle_path)
            print(f"Bundle saved to {cfg.bundle_path}")

    # ── Step 2: Compute ──────────────────────────────────────────────
    if args.compute:
        bundle = load_bundle(cfg.bundle_path)
        results = run_experiment(cfg, bundle=bundle)
        save_results(results, results_path)
        print(f"\nResults saved to {results_path}")

    # ── Step 3: Plot ─────────────────────────────────────────────────
    if args.plot:
        if not results_path.exists():
            print(f"No results at {results_path}; run with --compute first.")
            sys.exit(1)
        results = load_results(results_path)
        make_all_plots(results, cfg.out_path)


if __name__ == "__main__":
    main()
