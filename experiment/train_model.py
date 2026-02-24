"""Standalone script: train the ProbabilisticMLP and save weights to disk.

This script trains a single model on one realisation of the pre-drift
FriedmanDrift distribution and saves the resulting ``ModelBundle`` (weights
+ normalization statistics) to ``<output_dir>/model.pkl``.

The saved model is then loaded by the main experiment loop (``experiment.py``),
avoiding the need to retrain for every Monte-Carlo trial.  This is scientifically
valid because we are evaluating the monitor's properties for a *fixed deployed
model* — exactly the scenario PITMonitor is designed for.

Usage
-----
    python train_model.py                    # train with default config
    python train_model.py --epochs 300       # override epochs
    python train_model.py --seed 7           # different seed

The saved bundle path is printed on success.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_this_dir = Path(__file__).resolve().parent
_repo_root = _this_dir.parent
sys.path.insert(0, str(_this_dir))
sys.path.insert(0, str(_repo_root))

from config import Config
from data import generate_stream
from model import save_bundle, train_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train ProbabilisticMLP for PITMonitor experiment")
    p.add_argument("--epochs", type=int, default=None, help="Training epochs (default: config)")
    p.add_argument("--lr", type=float, default=None, help="Learning rate (default: config)")
    p.add_argument("--seed", type=int, default=None, help="RNG seed (default: config)")
    p.add_argument("--output", type=str, default="out", help="Output directory")
    return p.parse_args()


def main() -> None:
    """Train the model and save it to disk."""
    args = parse_args()

    cfg_kwargs: dict = {"output_dir": args.output}
    if args.epochs is not None:
        cfg_kwargs["epochs"] = args.epochs
    if args.lr is not None:
        cfg_kwargs["lr"] = args.lr
    if args.seed is not None:
        cfg_kwargs["seed"] = args.seed

    cfg = Config(**cfg_kwargs)

    bundle_path = cfg.out_path / "model.pkl"
    if bundle_path.exists():
        print(f"Model bundle already exists at {bundle_path}")
        print("Delete it to force retraining, or pass a different --output directory.")
        return

    print(f"Training ProbabilisticMLP")
    print(f"  n_train={cfg.n_train}, epochs={cfg.epochs}, lr={cfg.lr}, seed={cfg.seed}")

    # Use the first drift scenario's pre-drift data for training.
    # The pre-drift distribution is the same across all drift types, so
    # we only need one training set.
    drift_type, tw = cfg.drift_scenarios[0]
    print(f"  Generating training data (drift_type={drift_type!r}, seed={cfg.seed})...")
    np.random.seed(cfg.seed)
    X, y = generate_stream(cfg, drift_type=drift_type, transition_window=tw, seed=cfg.seed)
    X_train, y_train = X[: cfg.n_train], y[: cfg.n_train]

    t0 = time.time()
    bundle = train_model(
        X_train,
        y_train,
        epochs=cfg.epochs,
        lr=cfg.lr,
        seed=cfg.seed,
    )
    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed:.1f}s")

    # Quick sanity check: PIT mean should be near 0.5 on training data
    from model import compute_pits
    pits_train = compute_pits(bundle, X_train, y_train)
    print(f"  PIT mean on training set: {pits_train.mean():.3f} (expected ~0.5)")
    print(f"  PIT std  on training set: {pits_train.std():.3f} (expected ~0.289 for U[0,1])")

    save_bundle(bundle, bundle_path)
    print(f"\nModel bundle saved to: {bundle_path}")


if __name__ == "__main__":
    main()
