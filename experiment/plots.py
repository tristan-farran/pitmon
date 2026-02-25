"""Publication-quality plots for the drift detection comparison.

Produces:
    fig_detection_rates.png    – grouped bar chart: TPR / FPR per method × scenario
    fig_delay_distributions.png – violin + strip plots of detection delay
    fig_single_run_<scen>.png  – 4-panel single-run diagnostic per scenario
    fig_nbins_sweep.png        – PITMonitor TPR / FPR / delay vs n_bins
    table_results.tex          – LaTeX booktabs table for the paper

All plot helpers accept a *save_dir* Path and save to that directory.
They also return the Figure so callers can embed figures programmatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np

from detectors import ALL_DETECTOR_NAMES


# ─── Style ───────────────────────────────────────────────────────────

SCENARIO_LABELS: dict[str, str] = {
    "gra_tw0": "Abrupt (GRA)",
    "gsg_tw500": "Gradual (GSG)",
    "lea_tw0": "Local (LEA)",
}

# Short labels for compact table/axis use
SCENARIO_SHORT: dict[str, str] = {
    "gra_tw0": "GRA",
    "gsg_tw500": "GSG",
    "lea_tw0": "LEA",
}

METHOD_COLORS: dict[str, str] = {
    "PITMonitor": "#1b6ec2",
    "ADWIN":       "#7c3aed",
    "KSWIN":       "#0d9488",
    "PageHinkley": "#16a34a",
    "DDM":         "#ca8a04",
    "EDDM":        "#ea580c",
    "HDDM_A":      "#db2777",
    "HDDM_W":      "#9d174d",
}

# Hatching patterns to distinguish methods in greyscale printing
METHOD_HATCHES: dict[str, str] = {
    "PITMonitor": "",
    "ADWIN":       "//",
    "KSWIN":       "\\\\",
    "PageHinkley": "xx",
    "DDM":         "..",
    "EDDM":        "++",
    "HDDM_A":      "--",
    "HDDM_W":      "||",
}

_RC_OVERRIDES = {
    "font.family":         "serif",
    "font.serif":          ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset":    "cm",
    "font.size":           9,
    "axes.titlesize":      10,
    "axes.titleweight":    "bold",
    "axes.labelsize":      9,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "legend.fontsize":     7.5,
    "legend.framealpha":   0.85,
    "legend.edgecolor":    "0.7",
    "figure.dpi":          200,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.alpha":          0.25,
    "grid.linewidth":      0.5,
    "lines.linewidth":     1.5,
    "patch.linewidth":     0.5,
}


def _apply_style() -> None:
    """Apply shared rcParams to all subsequent matplotlib calls."""
    plt.rcParams.update(_RC_OVERRIDES)


def _fmt_pct(x: float) -> str:
    """Format a rate as a percentage string."""
    if np.isnan(x):
        return "—"
    return f"{x:.1%}"


def _fmt_delay(x: float) -> str:
    """Format a delay value."""
    if np.isnan(x):
        return "—"
    return f"{x:.0f}"


# ─── Figure 1: Detection rate comparison ────────────────────────────


def plot_detection_rates(results: dict, save_dir: Path) -> plt.Figure:
    """Grouped bar chart for TPR and FPR per method and scenario.

    Parameters
    ----------
    results : dict
        Output from ``run_experiment`` (loaded from JSON).
    save_dir : Path

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
        n_scenarios, 2, figsize=(7.0, 2.4 * n_scenarios),
        gridspec_kw={"wspace": 0.35, "hspace": 0.4},
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
        bars = ax.barh(
            y, tprs, xerr=tpr_err_arr, color=colors, alpha=0.88,
            edgecolor="white", linewidth=0.5,
            error_kw={"lw": 0.8, "capsize": 2, "capthick": 0.8},
        )
        for bar, method in zip(bars, methods):
            bar.set_hatch(METHOD_HATCHES.get(method, ""))
        ax.set_yticks(y)
        ax.set_yticklabels(methods, fontsize=8)
        ax.set_xlim(0, 1.08)
        ax.set_title(f"{scen_label} — TPR")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))

        # FPR panel
        ax = axes[i, 1]
        bars = ax.barh(
            y, fprs, xerr=fpr_err_arr, color=colors, alpha=0.88,
            edgecolor="white", linewidth=0.5,
            error_kw={"lw": 0.8, "capsize": 2, "capthick": 0.8},
        )
        for bar, method in zip(bars, methods):
            bar.set_hatch(METHOD_HATCHES.get(method, ""))
        ax.axvline(
            alpha, color="crimson", ls="--", lw=1.2, alpha=0.8,
            label=f"α = {alpha}",
        )
        ax.set_yticks(y)
        ax.set_yticklabels([])
        max_fpr = max(max(fprs), alpha * 3)
        ax.set_xlim(0, min(max_fpr * 1.15, 1.08))
        ax.set_title(f"{scen_label} — FPR")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))
        if i == 0:
            ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        "Drift Detection: True and False Positive Rates",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = save_dir / "fig_detection_rates.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return fig


# ─── Figure 2: Detection delay distributions ────────────────────────


def plot_delay_distributions(results: dict, save_dir: Path) -> plt.Figure:
    """Violin + jittered strip plots of detection delay (true positives).

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
    fig, axes = plt.subplots(
        1, n_scenarios, figsize=(4.0 * n_scenarios, 4.0), sharey=False,
    )
    if n_scenarios == 1:
        axes = [axes]

    for ax, scen in zip(axes, scenarios):
        scen_label = SCENARIO_LABELS.get(scen, scen)
        delay_data, labels, colors = [], [], []

        for method in ALL_DETECTOR_NAMES:
            delays = results["results"][scen].get(method, {}).get("delays", [])
            if len(delays) > 1:
                delay_data.append(delays)
                labels.append(method)
                colors.append(METHOD_COLORS.get(method, "#999999"))

        if not delay_data:
            ax.text(
                0.5, 0.5, "No detections",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, fontstyle="italic", color="0.5",
            )
            ax.set_title(scen_label)
            continue

        # Violin plot with custom styling
        parts = ax.violinplot(
            delay_data, positions=range(1, len(delay_data) + 1),
            showmedians=False, showextrema=False,
        )
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.3)
            body.set_edgecolor(color)
            body.set_linewidth(0.8)

        # Overlay box plot (thin)
        bp = ax.boxplot(
            delay_data,
            positions=range(1, len(delay_data) + 1),
            patch_artist=True,
            widths=0.20,
            showfliers=False,
            medianprops={"color": "white", "linewidth": 1.6},
            whiskerprops={"linewidth": 1.0, "color": "0.4"},
            capprops={"linewidth": 1.0, "color": "0.4"},
            boxprops={"linewidth": 0.8},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7.5)
        ax.set_title(scen_label)

        # Tight y-limits
        all_vals = np.concatenate(delay_data)
        if len(all_vals) > 0:
            lo = max(0, float(np.percentile(all_vals, 1)))
            hi = float(np.percentile(all_vals, 99))
            pad = max((hi - lo) * 0.08, 5)
            ax.set_ylim(lo - pad, hi + pad)

    axes[0].set_ylabel("Detection delay (samples)")
    fig.suptitle(
        "Detection Delay Distributions (True Positives Only)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = save_dir / "fig_delay_distributions.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return fig


# ─── LaTeX summary table ────────────────────────────────────────────


def generate_latex_table(results: dict, save_dir: Path) -> str:
    r"""Generate a LaTeX booktabs table with all summary statistics.

    Columns per scenario: TPR, FPR, Mean Delay, and (for PITMonitor only)
    Mean Changepoint Error.  The table is saved as ``table_results.tex``
    and also returned as a string.

    Parameters
    ----------
    results : dict
    save_dir : Path

    Returns
    -------
    str
        Complete LaTeX table source (ready to \\input{} in a paper).
    """
    scenarios = [k for k in results["results"] if k in SCENARIO_LABELS]
    if not scenarios:
        scenarios = list(results["results"].keys())

    n_scen = len(scenarios)
    alpha = results["config"]["alpha"]
    n_trials = results["config"]["n_trials"]

    # Build column spec: Method | (TPR FPR Delay) × n_scenarios | CP Err
    col_spec = "l" + " ccc" * n_scen + " c"

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Drift detection results on FriedmanDrift "
        rf"({n_trials:,} trials, $\alpha={alpha}$). "
        r"Mean delay and changepoint error are in samples. "
        r"Best TPR per scenario is \textbf{bolded}; "
        r"FPR exceeding $\alpha$ is \underline{underlined}.}"
    )
    lines.append(r"\label{tab:results}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header row 1: scenario names spanning 3 columns each
    header1_parts = [r"\multicolumn{1}{l}{}"]
    for scen in scenarios:
        label = SCENARIO_SHORT.get(scen, scen)
        header1_parts.append(rf"\multicolumn{{3}}{{c}}{{{label}}}")
    header1_parts.append(r"\multicolumn{1}{c}{}")
    lines.append(" & ".join(header1_parts) + r" \\")

    # Cmidrules under each scenario group
    cmidrules = []
    for i, _scen in enumerate(scenarios):
        start = 2 + i * 3
        end = start + 2
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
    lines.append(" ".join(cmidrules))

    # Header row 2: metric names
    header2_parts = ["Method"]
    for _scen in scenarios:
        header2_parts.extend(["TPR", "FPR", "Delay"])
    header2_parts.append(r"$\overline{|\hat\tau - \tau|}$")
    lines.append(" & ".join(header2_parts) + r" \\")
    lines.append(r"\midrule")

    # Find best TPR per scenario for bolding
    best_tpr = {}
    for scen in scenarios:
        tprs = []
        for method in ALL_DETECTOR_NAMES:
            s = results["results"][scen].get(method, {})
            tprs.append(s.get("tpr", 0.0))
        best_tpr[scen] = max(tprs)

    # Data rows
    for method in ALL_DETECTOR_NAMES:
        row_parts = [method.replace("_", r"\_")]

        # Collect changepoint errors across scenarios (PITMonitor only)
        cp_errors_across = []

        for scen in scenarios:
            s = results["results"][scen].get(method, {})
            tpr = s.get("tpr", float("nan"))
            fpr = s.get("fpr", float("nan"))
            delay = s.get("mean_delay", float("nan"))

            # Format TPR (bold if best)
            tpr_str = _fmt_pct(tpr)
            if not np.isnan(tpr) and abs(tpr - best_tpr[scen]) < 1e-6:
                tpr_str = rf"\textbf{{{tpr_str}}}"

            # Format FPR (underline if > alpha)
            fpr_str = _fmt_pct(fpr)
            if not np.isnan(fpr) and fpr > alpha:
                fpr_str = rf"\underline{{{fpr_str}}}"

            delay_str = _fmt_delay(delay)

            row_parts.extend([tpr_str, fpr_str, delay_str])

            # Collect CP error for PITMonitor
            if method == "PITMonitor":
                cp_err = s.get("mean_cp_error", float("nan"))
                if not np.isnan(cp_err):
                    cp_errors_across.append(cp_err)

        # Changepoint error column (only PITMonitor has values)
        if method == "PITMonitor" and cp_errors_across:
            # Average across scenarios that have CP estimates
            avg_cp = np.mean(cp_errors_across)
            row_parts.append(_fmt_delay(avg_cp))
        else:
            row_parts.append("—")

        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex_src = "\n".join(lines)

    out = save_dir / "table_results.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex_src)
    print(f"  Saved {out}")
    return latex_src


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
        Dictionary returned by ``collect_single_run``.
    save_path : Path or None

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

    # Determine scenario label for title
    scen_key = artifacts.get("scenario_key", "")
    scen_label = SCENARIO_LABELS.get(scen_key, scen_key)

    fig, axes = plt.subplots(
        2, 2, figsize=(7.5, 5.5),
        gridspec_kw={"hspace": 0.38, "wspace": 0.32},
    )
    fig.suptitle(
        f"PITMonitor Single Run — {scen_label}",
        fontsize=11, fontweight="bold",
    )

    # Colour definitions for consistency
    c_pre = "#4a90d9"
    c_post = "#d94a4a"
    c_alarm = "#e8920d"
    c_shift = "#c41e3a"
    c_cp = "#2e8b57"

    # ── Top-left: Predictions vs Reality ────────────────────────────
    ax = axes[0, 0]
    mask_pre = times < true_shift_point
    mask_post = ~mask_pre

    ax.scatter(
        times[mask_pre], y_all[mask_pre], s=3, alpha=0.25,
        c=c_pre, rasterized=True,
    )
    ax.scatter(
        times[mask_post], y_all[mask_post], s=3, alpha=0.25,
        c=c_post, rasterized=True,
    )
    ax.scatter(
        times, preds, s=2, alpha=0.20, c="0.3", rasterized=True,
    )
    ax.axvline(
        true_shift_point, color=c_shift, ls=":", lw=1.2, alpha=0.8,
    )
    if artifacts["alarm_fired"] and artifacts["alarm_time"] is not None:
        ax.axvline(
            int(artifacts["alarm_time"]), color=c_alarm, ls="--", lw=1.2,
        )
    legend_handles = [
        Line2D([0], [0], marker="o", ls="", color=c_pre, ms=3, label="Actual (pre)"),
        Line2D([0], [0], marker="o", ls="", color=c_post, ms=3, label="Actual (post)"),
        Line2D([0], [0], marker="o", ls="", color="0.3", ms=3, label="Predicted"),
        Line2D([0], [0], color=c_shift, ls=":", lw=1.2, label="True shift"),
    ]
    if artifacts["alarm_fired"] and artifacts["alarm_time"] is not None:
        legend_handles.append(
            Line2D([0], [0], color=c_alarm, ls="--", lw=1.2,
                   label=f"Alarm (t={artifacts['alarm_time']})")
        )
    ax.legend(handles=legend_handles, fontsize=6.5, loc="upper right", ncol=2)
    ax.set(xlabel="Sample", ylabel="Target value", title="(a) Predictions vs reality")

    # ── Top-right: PIT stream ────────────────────────────────────────
    ax = axes[0, 1]
    point_colors = np.where(times < true_shift_point, c_pre, c_post)
    ax.scatter(times, pits, s=2, alpha=0.30, c=point_colors, rasterized=True)

    if len(pits) >= 50:
        w = 50
        rolling = np.convolve(pits, np.ones(w) / w, mode="valid")
        ax.plot(
            np.arange(w, len(pits) + 1), rolling,
            color="0.15", lw=1.0, alpha=0.8,
        )
    ax.axhline(0.5, color="0.5", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(true_shift_point, color=c_shift, ls=":", lw=1.2, alpha=0.8)
    if artifacts["alarm_fired"] and artifacts["alarm_time"] is not None:
        ax.axvline(int(artifacts["alarm_time"]), color=c_alarm, ls="--", lw=1.2)

    leg = [
        Line2D([0], [0], marker="o", ls="", color=c_pre, ms=3, label="Pre-shift"),
        Line2D([0], [0], marker="o", ls="", color=c_post, ms=3, label="Post-shift"),
        Line2D([0], [0], color="0.15", lw=1.0, label="Rolling mean"),
        Line2D([0], [0], color="0.5", ls="--", lw=0.8, label="Ref (0.5)"),
        Line2D([0], [0], color=c_shift, ls=":", lw=1.2, label="True shift"),
    ]
    ax.legend(handles=leg, fontsize=6.5, loc="upper right", ncol=2)
    ax.set(xlabel="Sample", ylabel="PIT", title="(b) PIT stream", ylim=(-0.05, 1.05))

    # ── Bottom-left: E-process ───────────────────────────────────────
    ax = axes[1, 0]
    ax.semilogy(
        times, np.maximum(evidence, 1e-10),
        color=c_pre, lw=1.3,
    )
    threshold = 1.0 / float(artifacts["monitor_alpha"])
    ax.axhline(
        threshold, color=c_shift, ls="--", lw=1.2,
        label=f"Threshold (1/α = {threshold:.0f})",
    )
    ax.axvline(
        true_shift_point, color=c_shift, ls=":", lw=1.2, alpha=0.8,
        label="True shift",
    )
    if artifacts.get("changepoint") is not None:
        ax.axvline(
            int(artifacts["changepoint"]), color=c_cp, ls="--", lw=1.2,
            alpha=0.8, label=f"CP est. (t≈{artifacts['changepoint']})",
        )
    if artifacts["alarm_fired"] and artifacts["alarm_time"] is not None:
        ax.axvline(
            int(artifacts["alarm_time"]), color=c_alarm, ls="--", lw=1.2,
            label=f"Alarm (t={artifacts['alarm_time']})",
        )
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set(xlabel="Sample", ylabel="Evidence (log scale)", title="(c) E-process")

    # ── Bottom-right: PIT histograms ─────────────────────────────────
    ax = axes[1, 1]
    bins = np.linspace(0, 1, 21)
    pre_pits = pits[: true_shift_point - 1]
    post_pits = pits[true_shift_point - 1:]
    if len(pre_pits) > 0:
        ax.hist(
            pre_pits, bins=bins, density=True, alpha=0.50,
            color=c_pre, edgecolor="white", linewidth=0.4,
            label=f"Pre-shift (n={len(pre_pits)})",
        )
    if len(post_pits) > 0:
        ax.hist(
            post_pits, bins=bins, density=True, alpha=0.50,
            color=c_post, edgecolor="white", linewidth=0.4,
            label=f"Post-shift (n={len(post_pits)})",
        )
    ax.axhline(1.0, color="0.3", ls="--", lw=1.0, label="U[0,1] reference")
    ax.legend(fontsize=6.5)
    ax.set(
        xlabel="PIT", ylabel="Density",
        title="(d) PIT distributions", xlim=(0, 1),
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  Saved {save_path}")
    plt.close(fig)
    return fig


# ─── Figure 5: n_bins sensitivity sweep ──────────────────────────────


def plot_nbins_sweep(sweep: dict, save_dir: Path, alpha: float = 0.05) -> plt.Figure:
    """Line plots of PITMonitor TPR / FPR / delay vs n_bins.

    Parameters
    ----------
    sweep : dict
    save_dir : Path
    alpha : float

    Returns
    -------
    plt.Figure
    """
    _apply_style()
    scenarios = list(sweep["scenarios"].keys())
    n_bins_vals = sorted({int(k) for scen in sweep["scenarios"].values() for k in scen})

    n_scenarios = len(scenarios)
    fig, axes = plt.subplots(
        3, n_scenarios, figsize=(3.5 * n_scenarios, 7.0),
        gridspec_kw={"hspace": 0.35, "wspace": 0.30},
    )
    if n_scenarios == 1:
        axes = axes.reshape(3, 1)

    color = METHOD_COLORS["PITMonitor"]

    for col, scen in enumerate(scenarios):
        scen_data = sweep["scenarios"][scen]
        xs = [nb for nb in n_bins_vals if nb in scen_data or str(nb) in scen_data]
        def get_d(nb):
            return scen_data.get(nb, scen_data.get(str(nb), {}))

        tprs  = [get_d(nb).get("tpr",          float("nan")) for nb in xs]
        fprs  = [get_d(nb).get("fpr",          float("nan")) for nb in xs]
        delas = [get_d(nb).get("mean_delay",
                  get_d(nb).get("median_delay", float("nan"))) for nb in xs]

        tpr_lo = [get_d(nb).get("tpr_ci", [float("nan"), float("nan")])[0] for nb in xs]
        tpr_hi = [get_d(nb).get("tpr_ci", [float("nan"), float("nan")])[1] for nb in xs]
        fpr_lo = [get_d(nb).get("fpr_ci", [float("nan"), float("nan")])[0] for nb in xs]
        fpr_hi = [get_d(nb).get("fpr_ci", [float("nan"), float("nan")])[1] for nb in xs]

        scen_label = SCENARIO_LABELS.get(scen, scen)

        # TPR
        ax = axes[0, col]
        ax.plot(xs, tprs, "o-", color=color, lw=1.5, ms=4)
        ax.fill_between(xs, tpr_lo, tpr_hi, color=color, alpha=0.15)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(scen_label)
        if col == 0:
            ax.set_ylabel("TPR")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))

        # FPR
        ax = axes[1, col]
        ax.plot(xs, fprs, "o-", color=color, lw=1.5, ms=4)
        ax.fill_between(xs, fpr_lo, fpr_hi, color=color, alpha=0.15)
        ax.axhline(alpha, color="crimson", ls="--", lw=1.0, label=f"α = {alpha}")
        valid_fprs = [f for f in fprs if not np.isnan(f)]
        ax.set_ylim(
            -0.01,
            max(max(valid_fprs, default=alpha), alpha) * 1.6,
        )
        if col == 0:
            ax.set_ylabel("FPR")
            ax.legend(fontsize=7)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))

        # Delay
        ax = axes[2, col]
        valid = [(x, d) for x, d in zip(xs, delas) if not np.isnan(d)]
        if valid:
            vx, vd = zip(*valid)
            ax.plot(vx, vd, "o-", color=color, lw=1.5, ms=4)
        if col == 0:
            ax.set_ylabel("Mean delay (samples)")
        ax.set_xlabel("n_bins")

    fig.suptitle(
        "PITMonitor Sensitivity to n_bins",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = save_dir / "fig_nbins_sweep.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return fig


# ─── Master plot function ────────────────────────────────────────────


def make_all_plots(results: dict, save_dir: Path) -> None:
    """Generate all publication figures and the LaTeX table from saved results.

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

    # LaTeX table (replaces the old PNG summary table)
    generate_latex_table(results, save_dir)

    # Single-run panels
    for scenario_key, artifacts in results.get("single_runs", {}).items():
        out_path = save_dir / f"fig_single_run_{scenario_key}.png"
        plot_single_run_panels(artifacts, save_path=out_path)

    # n_bins sweep
    bins_sweep = results.get("bins_sweep")
    if bins_sweep:
        alpha = results["config"].get("alpha", 0.05)
        plot_nbins_sweep(bins_sweep, save_dir, alpha=alpha)

    print("Done.")
