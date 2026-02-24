"""Plots for the drift detection comparison.

Produces:
    fig_detection_rates.png    – horizontal bar chart: TPR / FPR per method × scenario
    fig_delay_distributions.png – box plots of detection delay (true positives only)
    fig_summary_table.png      – colour-coded summary table
    fig_single_run_<scen>.png  – 4-panel single-run diagnostic per scenario
    fig_nbins_sweep.png        – PITMonitor TPR / FPR / delay vs n_bins

All plot helpers accept a *save_dir* Path and save to that directory.
They also return the Figure so callers can embed figures programmatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from detectors import ALL_DETECTOR_NAMES


# ─── Style ───────────────────────────────────────────────────────────

SCENARIO_LABELS: dict[str, str] = {
    "gra_tw0": "Abrupt (GRA)",
    "gsg_tw500": "Gradual (GSG)",
    "lea_tw0": "Local (LEA)",
}

METHOD_COLORS: dict[str, str] = {
    "PITMonitor": "#2489cd",
    "ADWIN":       "#9e7ade",
    "KSWIN":       "#41c0c4",
    "PageHinkley": "#68c46a",
    "DDM":         "#fec229",
    "EDDM":        "#d95f0e",
    "HDDM_A":      "#f768a1",
    "HDDM_W":      "#c51b8a",
}

_RC_OVERRIDES = {
    "font.family":     "serif",
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi":      150,
}


def _apply_style() -> None:
    """Apply shared rcParams to all subsequent matplotlib calls."""
    plt.rcParams.update(_RC_OVERRIDES)


# ─── Figure 1: Detection rate comparison ────────────────────────────


def plot_detection_rates(results: dict, save_dir: Path) -> plt.Figure:
    """Horizontal bar charts for TPR and FPR per method and scenario.

    Parameters
    ----------
    results : dict
        Output from ``run_experiment`` (loaded from JSON).
    save_dir : Path
        Directory where the figure is saved.

    Returns
    -------
    plt.Figure
    """
    _apply_style()
    scenarios = [k for k in results["results"] if k in SCENARIO_LABELS]
    if not scenarios:
        scenarios = list(results["results"].keys())

    alpha = results["config"]["alpha"]
    n_scenarios = len(scenarios)

    fig, axes = plt.subplots(
        n_scenarios, 2, figsize=(12, 4 * n_scenarios), sharex="col"
    )
    if n_scenarios == 1:
        axes = np.array([axes])

    for i, scen in enumerate(scenarios):
        scen_label = SCENARIO_LABELS.get(scen, scen)
        methods, tprs, fprs, tpr_err, fpr_err, colors = [], [], [], [], [], []

        for method in ALL_DETECTOR_NAMES:
            s = results["results"][scen].get(method, {})
            tpr = s.get("tpr", 0.0)
            fpr = s.get("fpr", 0.0)
            methods.append(method)
            tprs.append(tpr)
            fprs.append(fpr)

            ci = s.get("tpr_ci", (tpr, tpr))
            tpr_err.append([tpr - ci[0], ci[1] - tpr])
            ci = s.get("fpr_ci", (fpr, fpr))
            fpr_err.append([fpr - ci[0], ci[1] - fpr])
            colors.append(METHOD_COLORS.get(method, "#999999"))

        y = np.arange(len(methods))
        tpr_err_arr = np.abs(np.array(tpr_err).T)
        fpr_err_arr = np.abs(np.array(fpr_err).T)

        # TPR panel
        ax = axes[i, 0]
        ax.barh(y, tprs, xerr=tpr_err_arr, color=colors, alpha=0.9, error_kw={"lw": 1.2})
        ax.set_yticks(y)
        ax.set_yticklabels(methods)
        ax.set_xlim(0, 1.05)
        ax.set_title(f"{scen_label} — True Positive Rate")
        ax.grid(axis="x", alpha=0.3)

        # FPR panel
        ax = axes[i, 1]
        ax.barh(y, fprs, xerr=fpr_err_arr, color=colors, alpha=0.9, error_kw={"lw": 1.2})
        ax.axvline(alpha, color="crimson", ls="--", lw=1.5, label=f"α = {alpha}")
        ax.set_yticks(y)
        ax.set_yticklabels([])
        # Clip x-axis to make small values visible without wasting space
        max_fpr = max(max(fprs), alpha * 3)
        ax.set_xlim(0, min(max_fpr * 1.1, 1.05))
        ax.set_title(f"{scen_label} — False Positive Rate")
        ax.grid(axis="x", alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    axes[-1, 0].set_xlabel("Rate")
    axes[-1, 1].set_xlabel("Rate")
    fig.suptitle(
        "Drift Detection Comparison on FriedmanDrift", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    out = save_dir / "fig_detection_rates.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return fig


# ─── Figure 2: Detection delay box plots ────────────────────────────


def plot_delay_distributions(results: dict, save_dir: Path) -> plt.Figure:
    """Box plots of detection delay for true-positive trials.

    Detectors with no true-positive detections are omitted per scenario.
    Y-axis limits are set to the 5th–95th percentile range to avoid
    extreme outliers crushing the interquartile structure.

    Parameters
    ----------
    results : dict
    save_dir : Path

    Returns
    -------
    plt.Figure
    """
    _apply_style()
    scenarios = [k for k in results["results"] if k in SCENARIO_LABELS]
    if not scenarios:
        scenarios = list(results["results"].keys())

    n_scenarios = len(scenarios)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 5), sharey=False)
    if n_scenarios == 1:
        axes = [axes]

    for ax, scen in zip(axes, scenarios):
        scen_label = SCENARIO_LABELS.get(scen, scen)
        delay_data, labels, colors = [], [], []

        for method in ALL_DETECTOR_NAMES:
            delays = results["results"][scen].get(method, {}).get("delays", [])
            if len(delays) > 1:  # need at least 2 points for a box
                delay_data.append(delays)
                labels.append(method)
                colors.append(METHOD_COLORS.get(method, "#999999"))

        if not delay_data:
            ax.text(0.5, 0.5, "No detections", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{scen_label}\n(no detections)")
            continue

        bp = ax.boxplot(
            delay_data,
            patch_artist=True,
            widths=0.55,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.8},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.82)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_title(scen_label)
        ax.grid(axis="y", alpha=0.3)

        # Tight y-limits based on data percentiles (excludes extreme outliers)
        all_vals = np.concatenate(delay_data)
        if len(all_vals) > 0:
            lo = max(0, float(np.percentile(all_vals, 2)))
            hi = float(np.percentile(all_vals, 98))
            pad = max((hi - lo) * 0.1, 1)
            ax.set_ylim(lo - pad, hi + pad)

    axes[0].set_ylabel("Detection Delay (samples)")
    fig.suptitle(
        "Detection Delay Distributions (True Positives Only)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    out = save_dir / "fig_delay_distributions.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return fig


# ─── Figure 3: Summary table ─────────────────────────────────────────


def plot_summary_table(results: dict, save_dir: Path) -> plt.Figure:
    """Colour-coded summary table of TPR / FPR / median delay per method × scenario.

    Parameters
    ----------
    results : dict
    save_dir : Path

    Returns
    -------
    plt.Figure
    """
    _apply_style()
    scenarios = [k for k in results["results"] if k in SCENARIO_LABELS]
    if not scenarios:
        scenarios = list(results["results"].keys())

    col_headers = []
    for scen in scenarios:
        label = SCENARIO_LABELS.get(scen, scen)
        col_headers.extend([f"{label}\nTPR", f"{label}\nFPR", f"{label}\nDelay"])

    cell_text = []
    row_labels = []
    cell_colors = []
    alpha_val = results["config"]["alpha"]

    for method in ALL_DETECTOR_NAMES:
        row, row_clr = [], []
        for scen in scenarios:
            s = results["results"][scen].get(method, {})
            tpr = s.get("tpr", float("nan"))
            fpr = s.get("fpr", float("nan"))
            delay = s.get("median_delay", float("nan"))

            row.append(f"{tpr:.1%}" if not np.isnan(tpr) else "—")
            row.append(f"{fpr:.1%}" if not np.isnan(fpr) else "—")
            row.append(f"{delay:.0f}" if not np.isnan(delay) else "—")

            # Green background for high TPR; red for elevated FPR
            row_clr.append((0.85, 0.95, 0.85, 1.0) if tpr > 0.8 else (1, 1, 1, 1))
            row_clr.append((1.0, 0.85, 0.85, 1.0) if fpr > alpha_val else (1, 1, 1, 1))
            row_clr.append((1, 1, 1, 1))

        cell_text.append(row)
        cell_colors.append(row_clr)
        row_labels.append(method)

    n_cols = len(col_headers)
    fig_width = max(3.2 * len(scenarios), 8)
    fig_height = 0.55 * len(ALL_DETECTOR_NAMES) + 2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_headers,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    for j in range(n_cols):
        table[0, j].set_text_props(fontweight="bold")
    for i in range(len(ALL_DETECTOR_NAMES)):
        table[i + 1, -1].set_text_props(fontweight="bold")

    fig.suptitle(
        f"Drift Detection Results — {results['config']['n_trials']} trials, "
        f"α = {results['config']['alpha']}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out = save_dir / "fig_summary_table.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return fig


# ─── Figure 4: Single-run panels ─────────────────────────────────────


def plot_single_run_panels(
    artifacts: dict,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Four-panel diagnostic for a single monitoring run.

    Panels:
        Top-left  – Raw predictions vs actual values with alarm markers.
        Top-right – PIT stream with rolling mean and shift indicator.
        Bottom-left – E-process on a log scale with threshold and changepoint.
        Bottom-right – Pre-shift vs post-shift PIT histograms.

    Parameters
    ----------
    artifacts : dict
        Dictionary returned by ``collect_single_run``.  Expected keys:
        ``true_shift_point``, ``true_labels``, ``predictions``, ``pits``,
        ``evidence_trace``, ``alarm_fired``, ``alarm_time``, ``monitor_alpha``,
        ``changepoint``.
    save_path : Path or None
        If given, figure is saved here (parent directories created).

    Returns
    -------
    plt.Figure
    """
    _apply_style()

    true_shift_point = int(artifacts["true_shift_point"])
    y_all = np.asarray(artifacts["true_labels"], dtype=float)
    preds = np.asarray(artifacts["predictions"], dtype=float)
    pits = np.asarray(artifacts["pits"], dtype=float)
    evidence = np.asarray(artifacts["evidence_trace"], dtype=float)
    times = np.arange(1, len(y_all) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "PITMonitor: FriedmanDrift Regime Shift (single run)",
        fontsize=13, fontweight="bold",
    )

    # ── Top-left: Predictions vs Reality ────────────────────────────
    ax = axes[0, 0]
    ax.scatter(times, y_all, s=7, alpha=0.35, c="steelblue", label="Actual")
    ax.scatter(times, preds, s=7, alpha=0.35, c="darkorange", label="Predicted mean")
    ax.axvline(true_shift_point, color="red", ls=":", lw=1.5, alpha=0.8, label="True shift")
    if artifacts["alarm_fired"] and artifacts["alarm_time"] is not None:
        ax.axvline(
            int(artifacts["alarm_time"]),
            color="orange", ls="--", lw=1.5,
            label=f"Alarm (t={artifacts['alarm_time']})",
        )
    ax.set(xlabel="Sample", ylabel="Target value", title="Predictions vs Reality")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    # ── Top-right: PIT stream ────────────────────────────────────────
    ax = axes[0, 1]
    point_colors = np.where(times < true_shift_point, "steelblue", "crimson")
    ax.scatter(times, pits, s=5, alpha=0.45, c=point_colors)
    # Patch legend manually since scatter with array colours has no single label
    from matplotlib.lines import Line2D
    leg_handles = [
        Line2D([0], [0], marker="o", ls="", color="steelblue", ms=5, label="Pre-shift"),
        Line2D([0], [0], marker="o", ls="", color="crimson",   ms=5, label="Post-shift"),
    ]
    if len(pits) >= 30:
        rolling = np.convolve(pits, np.ones(30) / 30, mode="valid")
        ax.plot(
            np.arange(30, len(pits) + 1), rolling,
            color="black", lw=1.3, label="Rolling mean (w=30)",
        )
        leg_handles.append(Line2D([0], [0], color="black", lw=1.3, label="Rolling mean (w=30)"))
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="Reference (0.5)")
    ax.axvline(true_shift_point, color="red", ls=":", lw=1.5, alpha=0.8, label="True shift")
    if artifacts["alarm_fired"] and artifacts["alarm_time"] is not None:
        ax.axvline(
            int(artifacts["alarm_time"]),
            color="orange", ls="--", lw=1.5,
            label=f"Alarm (t={artifacts['alarm_time']})",
        )
        leg_handles.append(
            Line2D([0], [0], color="orange", ls="--", lw=1.5,
                   label=f"Alarm (t={artifacts['alarm_time']})")
        )
    leg_handles += [
        Line2D([0], [0], color="gray",  ls="--", lw=1,   label="Reference (0.5)"),
        Line2D([0], [0], color="red",   ls=":",  lw=1.5,  label="True shift"),
    ]
    ax.set(xlabel="Sample", ylabel="PIT", title="PIT stream", ylim=(-0.05, 1.05))
    ax.legend(handles=leg_handles, fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2)

    # ── Bottom-left: E-process ───────────────────────────────────────
    ax = axes[1, 0]
    ax.semilogy(
        times, np.maximum(evidence, 1e-10),
        color="steelblue", lw=1.8, label="E-process",
    )
    threshold = 1.0 / float(artifacts["monitor_alpha"])
    ax.axhline(
        threshold, color="crimson", ls="--", lw=1.8,
        label=f"Threshold (1/α = {threshold:.0f})",
    )
    ax.axvline(true_shift_point, color="red", ls=":", lw=1.5, alpha=0.8, label="True shift")
    if artifacts.get("changepoint") is not None:
        ax.axvline(
            int(artifacts["changepoint"]),
            color="green", ls="--", lw=1.3, alpha=0.8, label="Changepoint estimate",
        )
    if artifacts["alarm_fired"] and artifacts["alarm_time"] is not None:
        ax.axvline(
            int(artifacts["alarm_time"]),
            color="orange", ls="--", lw=1.8,
            label=f"Alarm (t={artifacts['alarm_time']})",
        )
    ax.set(xlabel="Sample", ylabel="Evidence (log scale)", title="E-process")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.2)

    # ── Bottom-right: PIT histograms ─────────────────────────────────
    ax = axes[1, 1]
    bins = np.linspace(0, 1, 21)
    pre_pits = pits[: true_shift_point - 1]
    post_pits = pits[true_shift_point - 1:]
    if len(pre_pits) > 0:
        ax.hist(
            pre_pits, bins=bins, density=True, alpha=0.5,
            color="steelblue", edgecolor="white", label=f"Pre-shift (n={len(pre_pits)})",
        )
    if len(post_pits) > 0:
        ax.hist(
            post_pits, bins=bins, density=True, alpha=0.5,
            color="crimson", edgecolor="white", label=f"Post-shift (n={len(post_pits)})",
        )
    ax.axhline(1.0, color="black", ls="--", lw=1.3, label="U[0,1] reference")
    ax.set(xlabel="PIT", ylabel="Density", title="PIT distributions", xlim=(0, 1))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved {save_path}")
    return fig


# ─── Figure 5: n_bins sensitivity sweep ──────────────────────────────


def plot_nbins_sweep(sweep: dict, save_dir: Path, alpha: float = 0.05) -> plt.Figure:
    """Line plots of PITMonitor TPR / FPR / delay vs n_bins.

    Theory note
    -----------
    The histogram approximates the e-value density at each step.  With too few
    bins the density estimate is too coarse to detect concentrated p-values;
    with too many bins each cell is noisy (high variance) and the e-values are
    unreliable.  The optimal bin size balances these two error sources.
    For the FriedmanDrift post-drift distribution the sweet spot typically lies
    in the range [10, 25].

    Parameters
    ----------
    sweep : dict
        Output from ``run_bins_sweep``.
    save_dir : Path
    alpha : float
        Nominal FPR level; drawn as a reference line on the FPR sub-plot.

    Returns
    -------
    plt.Figure
    """
    _apply_style()
    scenarios = list(sweep["scenarios"].keys())
    n_bins_vals = sorted({int(k) for scen in sweep["scenarios"].values() for k in scen})

    n_scenarios = len(scenarios)
    fig, axes = plt.subplots(3, n_scenarios, figsize=(5 * n_scenarios, 10), sharex="col")
    if n_scenarios == 1:
        axes = axes.reshape(3, 1)

    row_titles = ["True Positive Rate", "False Positive Rate", "Median Detection Delay"]

    for col, scen in enumerate(scenarios):
        scen_data = sweep["scenarios"][scen]
        xs = [nb for nb in n_bins_vals if nb in scen_data or str(nb) in scen_data]
        def get_d(nb):
            return scen_data.get(nb, scen_data.get(str(nb), {}))

        tprs  = [get_d(nb).get("tpr",          float("nan")) for nb in xs]
        fprs  = [get_d(nb).get("fpr",          float("nan")) for nb in xs]
        delas = [get_d(nb).get("median_delay", float("nan")) for nb in xs]

        tpr_lo = [get_d(nb).get("tpr_ci", [float("nan"), float("nan")])[0] for nb in xs]
        tpr_hi = [get_d(nb).get("tpr_ci", [float("nan"), float("nan")])[1] for nb in xs]
        fpr_lo = [get_d(nb).get("fpr_ci", [float("nan"), float("nan")])[0] for nb in xs]
        fpr_hi = [get_d(nb).get("fpr_ci", [float("nan"), float("nan")])[1] for nb in xs]

        scen_label = SCENARIO_LABELS.get(scen, scen)
        color = METHOD_COLORS["PITMonitor"]

        # TPR
        ax = axes[0, col]
        ax.plot(xs, tprs, "o-", color=color, lw=2)
        ax.fill_between(xs, tpr_lo, tpr_hi, color=color, alpha=0.2)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(scen_label)
        if col == 0:
            ax.set_ylabel(row_titles[0])
        ax.grid(alpha=0.3)

        # FPR
        ax = axes[1, col]
        ax.plot(xs, fprs, "o-", color=color, lw=2)
        ax.fill_between(xs, fpr_lo, fpr_hi, color=color, alpha=0.2)
        ax.axhline(alpha, color="crimson", ls="--", lw=1.5, label=f"α = {alpha}")
        ax.set_ylim(-0.02, max(max([f for f in fprs if not np.isnan(f)], default=alpha), alpha) * 1.5)
        if col == 0:
            ax.set_ylabel(row_titles[1])
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Delay
        ax = axes[2, col]
        valid_dels = [(x, d) for x, d in zip(xs, delas) if not np.isnan(d)]
        if valid_dels:
            vx, vd = zip(*valid_dels)
            ax.plot(vx, vd, "o-", color=color, lw=2)
        if col == 0:
            ax.set_ylabel(row_titles[2])
        ax.set_xlabel("n_bins")
        ax.grid(alpha=0.3)

    fig.suptitle(
        "PITMonitor Sensitivity to n_bins\n"
        "(Theory: small n_bins → coarse density; large n_bins → high variance)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out = save_dir / "fig_nbins_sweep.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return fig


# ─── Master plot function ────────────────────────────────────────────


def make_all_plots(results: dict, save_dir: Path) -> None:
    """Generate all publication figures from saved results.

    Reads ``results["single_runs"]`` and ``results["bins_sweep"]`` directly
    from the results dict produced by ``run_experiment``; no separate sweep
    file is needed.

    Parameters
    ----------
    results : dict
        Full output of ``run_experiment`` / ``load_results``.
    save_dir : Path
        Output directory; created if absent.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating plots:")
    plot_detection_rates(results, save_dir)
    plot_delay_distributions(results, save_dir)
    plot_summary_table(results, save_dir)

    # Single-run panels — present in every results dict produced by run_experiment
    for scenario_key, artifacts in results.get("single_runs", {}).items():
        out_path = save_dir / f"fig_single_run_{scenario_key}.png"
        plot_single_run_panels(artifacts, save_path=out_path)

    # n_bins sweep — always present in results produced by run_experiment
    bins_sweep = results.get("bins_sweep")
    if bins_sweep:
        alpha = results["config"].get("alpha", 0.05)
        plot_nbins_sweep(bins_sweep, save_dir, alpha=alpha)

    print("Done.")
