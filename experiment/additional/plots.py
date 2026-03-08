"""Plots and LaTeX macro export for the additional experiments.

Produces:
    fig_proposition_formula_fit.png  – formula vs empirical E[e_t|A] curves
    fig_multistep_localization.png   – localization MAE and signed error distributions
    additional_macros.tex            – LaTeX \\newcommand macros for appendix tables

All plot helpers accept a *save_dir* Path and save to that directory.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ── Style ─────────────────────────────────────────────────────────────────────

_RC_OVERRIDES = {
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
}


def _apply_style() -> None:
    plt.rcParams.update(_RC_OVERRIDES)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Individual plots ──────────────────────────────────────────────────────────


def plot_proposition_fit(results: dict, save_dir: Path) -> None:
    """Plot formula vs empirical E[e_t|A] curves for aligned/misaligned priors."""
    cases = results["proposition_verification"]["cases"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)

    for ax, mode in zip(axes, ["aligned", "misaligned"]):
        rows = sorted(cases[mode]["rows"], key=lambda r: r["n_post_before_t"])
        n = np.array([r["n_post_before_t"] for r in rows], dtype=float)
        formula = np.array([r["formula_mean_e"] for r in rows], dtype=float)
        emp = np.array([r["empirical_mean_e"] for r in rows], dtype=float)
        se = np.array([r["empirical_se"] for r in rows], dtype=float)

        asymptotic = float(cases[mode]["asymptotic_target_Bsumtheta2"])
        gamma = float(cases[mode]["gamma"])
        n_star = float(cases[mode]["n_star_formula"])

        ax.plot(n, formula, color="#1f77b4", lw=1.8, label="Formula")
        ax.plot(n, emp, "o-", color="#d62728", markersize=4, lw=1.0, label="Empirical")
        ax.fill_between(
            n,
            emp - 1.96 * se,
            emp + 1.96 * se,
            color="#d62728",
            alpha=0.15,
            linewidth=0,
            label="Empirical 95% CI",
        )
        ax.axhline(1.0, color="0.35", lw=1.1, ls="--", label="Null baseline (1)")
        ax.axhline(
            asymptotic, color="#2ca02c", lw=1.2, ls="-.", label="Asymptotic target"
        )

        if gamma < 0 and n_star > 0:
            ax.axvline(n_star, color="#9467bd", lw=1.1, ls=":", label=r"Warmup $n^*$")

        ax.set_xscale("log")
        ax.set_title(f"{mode.capitalize()} prior")
        ax.set_xlabel("Post-shift lag n (samples)")

    axes[0].set_ylabel(r"$\mathbb{E}[e_t\mid A]$")
    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[1].legend(by_label.values(), by_label.keys(), loc="best")

    fig.suptitle("Finite-Time Mean Formula Verification", y=1.02)
    _save(fig, save_dir / "fig_proposition_formula_fit.png")


def plot_multistep_localization(results: dict, save_dir: Path) -> None:
    """Plot localization errors to onset and max-intensity phase."""
    loc = results.get("multistep_localization_verification")
    if not loc:
        return

    mae_onset = float(loc["mae_to_onset"])
    mae_max = float(loc["mae_to_max_phase"])
    signed_onset = np.asarray(loc.get("signed_error_to_onset", []), dtype=float)
    signed_max = np.asarray(loc.get("signed_error_to_max_phase", []), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    axes[0].bar(
        ["Onset", "Max phase"],
        [mae_onset, mae_max],
        color=["#4c78a8", "#f58518"],
        alpha=0.88,
        width=0.65,
    )
    axes[0].set_ylabel("Mean absolute localization error (samples)")
    axes[0].set_title("Localization MAE Targets")

    data = [signed_onset, signed_max]
    axes[1].boxplot(
        data,
        vert=False,
        tick_labels=["Error to onset", "Error to max phase"],
        widths=0.65,
        patch_artist=True,
        boxprops={"facecolor": "#9ecae1", "alpha": 0.8},
        medianprops={"color": "#d62728", "linewidth": 1.2},
    )
    axes[1].axvline(0.0, color="0.25", ls="--", lw=1.0)
    axes[1].set_xlabel("Signed localization error (estimate - reference)")
    axes[1].set_title("Signed Error Distribution")

    n_cp = int(loc.get("n_with_changepoint_estimate", 0))
    fig.suptitle(f"Multi-Step Drift Localization (n cp-estimates = {n_cp})", y=1.03)
    fig.tight_layout()
    _save(fig, save_dir / "fig_multistep_localization.png")


# ── LaTeX macros ──────────────────────────────────────────────────────────────


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}\\%"


def _f3(x: float) -> str:
    return f"{x:.3f}"


def _f1(x: float) -> str:
    return f"{x:.1f}"


def _as_int(x: float | int) -> str:
    return f"{int(x):,}".replace(",", "{,}")


def _emit(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def generate_experiment_macros(results: dict, save_dir: Path) -> str:
    r"""Generate a LaTeX file of \newcommand macros for additional experiment results.

    Parameters
    ----------
    results : dict
    save_dir : Path

    Returns
    -------
    str
        Complete LaTeX macro file source.
    """
    lines: list[str] = [
        "% Auto-generated by plots.py - do not edit manually.",
    ]

    cfg = results["config"]
    lines.append(_emit("addAlpha", str(cfg["alpha"])))
    lines.append(_emit("addBins", _as_int(cfg["n_bins"])))

    summary = results["summary"]
    lines.append(
        _emit(
            "addFiniteErrAligned",
            _f3(summary["proposition_formula_accuracy"]["aligned"]),
        )
    )
    lines.append(
        _emit(
            "addFiniteErrMisaligned",
            _f3(summary["proposition_formula_accuracy"]["misaligned"]),
        )
    )

    loc = results["multistep_localization_verification"]
    lines.append(_emit("addLocAlarmRate", _pct(loc["alarm_rate"])))
    lines.append(_emit("addLocMAEOnset", _f1(loc["mae_to_onset"])))
    lines.append(_emit("addLocMAEMax", _f1(loc["mae_to_max_phase"])))
    lines.append(
        _emit(
            "addLocMedSignedOnset", _as_int(round(loc["median_signed_error_to_onset"]))
        )
    )
    lines.append(
        _emit(
            "addLocMedSignedMax",
            _as_int(round(loc["median_signed_error_to_max_phase"])),
        )
    )
    lines.append(
        _emit("addLocCloserOnset", _as_int(loc["closer_reference_counts"]["onset"]))
    )
    lines.append(
        _emit("addLocCloserMax", _as_int(loc["closer_reference_counts"]["max_phase"]))
    )
    lines.append(_emit("addLocN", _as_int(loc["n_with_changepoint_estimate"])))

    latex_src = "\n".join(lines) + "\n"
    out = save_dir / "additional_macros.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex_src, encoding="utf-8")
    print(f"  Saved {out}")
    return latex_src


# ── Master ────────────────────────────────────────────────────────────────────


def make_all_plots(results: dict, save_dir: Path) -> None:
    """Generate all publication figures and LaTeX macros from saved results."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()
    print("\nGenerating plots:")
    plot_proposition_fit(results, save_dir)
    plot_multistep_localization(results, save_dir)
    generate_experiment_macros(results, save_dir)
    print("Done.")
