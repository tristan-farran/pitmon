from __future__ import annotations

import argparse
from pathlib import Path

from config import DeliveryDemoConfig
from pipeline import compute_all, load_artifacts, save_artifacts
from plots import plot_comparison_panels, plot_power_panels, plot_single_run_panels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modular delivery PITMonitor demo")
    parser.add_argument(
        "--compute", action="store_true", help="Compute and save artifacts"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Render plots from artifacts"
    )
    parser.add_argument(
        "--artifact",
        type=str,
        default=None,
        help="Artifact path (defaults to config output_dir/artifact_name)",
    )
    parser.add_argument("--trials", type=int, default=1_000, help="Power trial count")
    parser.add_argument(
        "--trials-compare",
        type=int,
        default=1_000,
        help="Method comparison trial count",
    )
    parser.add_argument("--workers", type=int, default=8, help="Max worker threads")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DeliveryDemoConfig(
        n_trials=args.trials,
        n_trials_compare=args.trials_compare,
        max_workers=args.workers,
    ).normalized()

    artifact_path = Path(args.artifact) if args.artifact else cfg.artifact_path()

    if not args.compute and not args.plot:
        args.compute = True
        args.plot = True

    if args.compute:
        artifacts = compute_all(cfg)
        save_artifacts(artifacts, artifact_path)
        print(f"Saved artifacts: {artifact_path}")
        print(f"Total compute time: {artifacts['elapsed_seconds']:.1f}s")

    if args.plot:
        artifacts = load_artifacts(artifact_path)
        single_path = artifact_path.with_name("single_run_panels.png")
        power_path = artifact_path.with_name("power_panels.png")
        compare_path = artifact_path.with_name("comparison_panels.png")
        plot_single_run_panels(artifacts, save_path=single_path)
        plot_power_panels(artifacts, save_path=power_path)
        plot_comparison_panels(artifacts, save_path=compare_path)
        print(f"Saved plot: {single_path}")
        print(f"Saved plot: {power_path}")
        print(f"Saved plot: {compare_path}")


if __name__ == "__main__":
    main()
