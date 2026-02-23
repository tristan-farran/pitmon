from __future__ import annotations

import argparse
from pathlib import Path

from config import CifarDemoConfig
from pipeline import (
    augment_artifacts_with_baseline_h0,
    compute_all,
    load_artifacts,
    save_artifacts,
)
from plots import (
    plot_baseline_h0_panels,
    plot_comparison_panels,
    plot_power_panels,
    plot_single_run_panels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modular CIFAR PITMonitor demo")
    parser.add_argument(
        "--compute", action="store_true", help="Compute all data and save artifact"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Render plots from artifact"
    )
    parser.add_argument(
        "--artifact",
        type=str,
        default=None,
        help="Artifact path (defaults to config output_dir/artifact_name)",
    )
    parser.add_argument("--trials", type=int, default=1_000, help="Unified trial count")
    parser.add_argument("--workers", type=int, default=8, help="Max worker threads")
    parser.add_argument("--corruption", type=str, default="gaussian_noise")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CifarDemoConfig(
        n_trials=args.trials,
        max_workers=args.workers,
        corruption=args.corruption,
    ).normalized()

    artifact_path = Path(args.artifact) if args.artifact else cfg.artifact_path()

    if not args.compute and not args.plot:
        args.compute = True
        args.plot = True

    if args.compute:
        if artifact_path.exists():
            artifacts = load_artifacts(artifact_path)
            artifacts = augment_artifacts_with_baseline_h0(artifacts, cfg)
            save_artifacts(artifacts, artifact_path)
            print(
                f"Reused existing artifacts and updated H0 baselines: {artifact_path}"
            )
        else:
            artifacts = compute_all(cfg)
            save_artifacts(artifacts, artifact_path)
            print(f"Saved artifacts: {artifact_path}")
            print(f"Total compute time: {artifacts['elapsed_seconds']:.1f}s")

    if args.plot:
        artifacts = load_artifacts(artifact_path)
        single_path = artifact_path.with_name("single_run_panels.png")
        power_path = artifact_path.with_name("power_panels.png")
        compare_path = artifact_path.with_name("comparison_panels.png")
        baseline_h0_path = artifact_path.with_name("comparison_h0_panels.png")
        plot_single_run_panels(artifacts, save_path=single_path)
        plot_power_panels(artifacts, save_path=power_path)
        plot_comparison_panels(artifacts, save_path=compare_path)
        plot_baseline_h0_panels(artifacts, save_path=baseline_h0_path)
        print(f"Saved plot: {single_path}")
        print(f"Saved plot: {power_path}")
        print(f"Saved plot: {compare_path}")
        print(f"Saved plot: {baseline_h0_path}")


if __name__ == "__main__":
    main()
