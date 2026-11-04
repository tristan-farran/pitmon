"""Entry point for the PITMonitor experiment.

The train / compute / plot steps are independent and can be run in isolation
or combined.  If none of ``--train``, ``--compute``, ``--plot`` are given,
all three run automatically.

When ``--scenario`` is used with ``--compute``, only the named scenario(s) are
(re)computed.  Existing results for other scenarios are preserved by merging
back into ``results.json``.

Flags
-----
--train              Train ProbabilisticMLP and save bundle (skipped if bundle exists)
--force-train        Re-train even if a bundle already exists
--compute            Run MC experiment (all scenarios, or only those in --scenario)
--plot               Load results.json and regenerate all figures
--scenario S [S ...] Drift-type names to (re)compute, e.g. --scenario lea gra
--trials N           Override number of MC trials (useful for quick tests)
--epochs N           Override training epochs
--workers N          Thread pool size for parallel trial execution
--seed N             Master RNG seed
--output DIR         Output directory (default: out)
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

_this_dir = Path(__file__).resolve().parent
_repo_root = _this_dir.parent
sys.path.insert(0, str(_this_dir))
sys.path.insert(0, str(_repo_root))

from config import Config
from plots import make_all_plots
from data import generate_stream
from model import load_bundle, save_bundle, train_model
from experiment import load_results, run_experiment, save_results


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    p = argparse.ArgumentParser(
        description="PITMonitor vs River drift detectors on FriedmanDrift",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Flags")[0],
    )
    p.add_argument("--train", action="store_true", help="Train and save the model")
    p.add_argument("--force-train", action="store_true", help="Re-train")
    p.add_argument("--compute", action="store_true", help="Run experiment")
    p.add_argument("--plot", action="store_true", help="Generate plots from results")
    p.add_argument(
        "--scenario",
        nargs="+",
        metavar="DRIFT_TYPE",
        help="Drift type(s) to (re)compute",
    )
    p.add_argument("--trials", type=int, default=10_000, help="Trials per scenario")
    p.add_argument("--output", type=str, default="out", help="Output directory")
    p.add_argument("--epochs", type=int, default=500, help="NN training epochs")
    p.add_argument("--workers", type=int, default=8, help="Parallel workers")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    return p.parse_args()


def main() -> None:
    """Orchestrate training, computation, and plotting."""
    args = parse_args()
    cfg = Config(
        seed=args.seed,
        epochs=args.epochs,
        n_trials=args.trials,
        output_dir=args.output,
        max_workers=args.workers,
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

        # Build optional scenario filter from drift-type names
        scenario_filter: set | None = None
        if args.scenario:
            # Map drift-type names (e.g. "lea") to scenario keys (e.g. "lea_tw0")
            key_map = {dt: f"{dt}_tw{tw}" for dt, tw in cfg.drift_scenarios}
            unknown = [s for s in args.scenario if s not in key_map]
            if unknown:
                print(f"Unknown scenario(s): {unknown}")
                print(f"Valid names: {list(key_map)}")
                sys.exit(1)
            scenario_filter = {key_map[s] for s in args.scenario}
            print(f"Scenario filter: {sorted(scenario_filter)}")

        new_results = run_experiment(
            cfg, bundle=bundle, scenario_filter=scenario_filter
        )

        if scenario_filter is not None and results_path.exists():
            # Merge: preserve existing scenarios, overwrite only the rerun ones
            existing = load_results(results_path)
            existing["results"].update(new_results["results"])
            existing["single_runs"].update(new_results["single_runs"])
            results = existing
            print(f"Merged {sorted(scenario_filter)} into existing {results_path}")
        else:
            results = new_results

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
