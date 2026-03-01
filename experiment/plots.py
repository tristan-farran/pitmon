"""Publication-quality plots for the drift detection comparison.

Produces:
    fig_detection_rates.png      – lollipop chart: TPR / FPR per method × scenario
    fig_delay_distributions.png  – horizontal violin + box plots of detection delay
    fig_cp_error_distribution.png – histogram of PITMonitor changepoint estimation error
    fig_single_run_<scen>.png    – 4-panel single-run diagnostic per scenario
    table_results.tex            – LaTeX booktabs table for the paper
    experiment_macros.tex        – LaTeX \\newcommand macros for experiment params & results

All plot helpers accept a *save_dir* Path and save to that directory.
They also return the Figure so callers can embed figures programmatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import numpy as np

from detectors import ALL_DETECTOR_NAMES


# ── Design tokens ─────────────────────────────────────────────────────────────

SCENARIO_LABELS: dict[str, str] = {
    "gra_tw0":   "Abrupt (GRA)",
    "gsg_tw500": "Gradual (GSG)",
    "lea_tw0":   "Local (LEA)",
}

SCENARIO_SHORT: dict[str, str] = {
    "gra_tw0":   "GRA",
    "gsg_tw500": "GSG",
    "lea_tw0":   "LEA",
}

# Okabe–Ito colorblind-safe palette — deep blue reserved for PITMonitor.
METHOD_COLORS: dict[str, str] = {
    "PITMonitor":  "#0072B2",   # deep blue    ← our method
    "ADWIN":       "#D55E00",   # vermillion
    "KSWIN":       "#009E73",   # bluish-green
    "PageHinkley": "#CC79A7",   # reddish-purple
    "DDM":         "#E69F00",   # amber
    "EDDM":        "#56B4E9",   # sky blue
    "HDDM_A":      "#A07800",   # dark gold
    "HDDM_W":      "#777777",   # neutral grey
}

# Marker shapes for secondary encoding (greyscale / accessibility)
METHOD_SHAPES: dict[str, str] = {
    "PITMonitor":  "D",   # filled diamond
    "ADWIN":       "o",   # circle
    "KSWIN":       "s",   # square
    "PageHinkley": "^",   # triangle-up
    "DDM":         "v",   # triangle-down
    "EDDM":        "P",   # fat plus
    "HDDM_A":      "X",   # fat cross
    "HDDM_W":      "h",   # hexagon
}

# Kept for backward-compatibility (no longer used in plots, but may be imported)
METHOD_HATCHES: dict[str, str] = {
    "PITMonitor": "",
    "ADWIN":      "//",
    "KSWIN":      "\\\\",
    "PageHinkley": "xx",
    "DDM":        "..",
    "EDDM":       "++",
    "HDDM_A":     "--",
    "HDDM_W":     "||",
}

_RC_OVERRIDES = {
    # Font — Computer Modern for LaTeX visual consistency
    "font.family":         "serif",
    "font.serif":          ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset":    "cm",
    "axes.unicode_minus":  False,
    # Sizes — calibrated for 3.3" single-col / 6.8" double-col paper figures
    "font.size":           8,
    "axes.titlesize":      9,
    "axes.titleweight":    "bold",
    "axes.labelsize":      8,
    "xtick.labelsize":     7,
    "ytick.labelsize":     7,
    "legend.fontsize":     7,
    "legend.title_fontsize": 7.5,
    # Legend aesthetics
    "legend.framealpha":   0.94,
    "legend.edgecolor":    "0.78",
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.4,
    "legend.borderpad":    0.5,
    # Resolution
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.04,
    # Axes
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.linewidth":      0.7,
    "axes.grid":           True,
    "grid.alpha":          0.22,
    "grid.linewidth":      0.35,
    "grid.linestyle":      ":",
    # Ticks
    "xtick.major.size":    2.5,
    "ytick.major.size":    2.5,
    "xtick.major.width":   0.6,
    "ytick.major.width":   0.6,
    # Lines & patches
    "lines.linewidth":     1.2,
    "patch.linewidth":     0.5,
}

plt.rcParams.update(_RC_OVERRIDES)


def _apply_style() -> None:
    """Re-apply shared rcParams (idempotent)."""
    plt.rcParams.update(_RC_OVERRIDES)


def _c(method: str) -> str:
    return METHOD_COLORS.get(method, "#666666")


def _m(method: str) -> str:
    return METHOD_SHAPES.get(method, "o")


def _fmt_pct(x: float) -> str:
    if np.isnan(x):
        return "—"
    return f"{x:.1%}".replace("%", r"\%")


def _fmt_delay(x: float) -> str:
    if np.isnan(x):
        return "—"
    return f"{x:.0f}"


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def _canonical_order(results: dict) -> list[str]:
    """Return ALL_DETECTOR_NAMES sorted by mean TPR on non-LEA scenarios (desc).

    Applied consistently to every figure so the same method always occupies
    the same row/column, making cross-figure comparison effortless.
    """
    base = [s for s in results["results"] if "lea" not in s]
    if not base:
        base = list(results["results"].keys())
    mean_tprs = {
        m: np.mean([results["results"][s].get(m, {}).get("tpr", 0.0) for s in base])
        for m in ALL_DETECTOR_NAMES
    }
    return sorted(ALL_DETECTOR_NAMES, key=lambda m: mean_tprs[m], reverse=True)


# ── Figure 1: Detection rates (lollipop) ──────────────────────────────────────

def plot_detection_rates(results: dict, save_dir: Path) -> plt.Figure:
    """Lollipop chart: TPR (top row) and FPR (bottom row) per detector × scenario.

    Methods are sorted by mean TPR on non-local scenarios so the best
    performers consistently appear at the top of every panel.
    FPR is capped at 15 % for readability; clipped values annotated.
    PITMonitor rows receive a subtle blue tint as a visual anchor.
    95 % confidence intervals shown as thin horizontal whiskers.
    """
    _apply_style()
    scenarios = [k for k in SCENARIO_LABELS if k in results["results"]]
    if not scenarios:
        scenarios = list(results["results"].keys())

    alpha = results["config"]["alpha"]
    n = len(scenarios)
    FPR_CAP = 0.15

    fig, axes = plt.subplots(
        2, n,
        figsize=(2.9 * n, 5.5),
        gridspec_kw={"wspace": 0.10, "hspace": 0.50},
    )
    fig.subplots_adjust(top=0.90)
    if n == 1:
        axes = axes.reshape(2, 1)

    ordered = _canonical_order(results)
    y = np.arange(len(ordered))
    pm_row = ordered.index("PITMonitor")

    for col, scen in enumerate(scenarios):
        is_left = col == 0

        # Collect metrics
        tprs, tpr_cis = [], []
        fprs_raw, fprs_disp, fpr_cis = [], [], []
        for m in ordered:
            s = results["results"][scen].get(m, {})
            tpr = s.get("tpr", 0.0)
            fpr = s.get("fpr", 0.0)
            tprs.append(tpr)
            ci_t = s.get("tpr_ci", [tpr, tpr])
            tpr_cis.append([tpr - ci_t[0], ci_t[1] - tpr])
            fprs_raw.append(fpr)
            fprs_disp.append(min(fpr, FPR_CAP))
            ci_f = s.get("fpr_ci", [fpr, fpr])
            fpr_cis.append(
                [fpr - ci_f[0], ci_f[1] - fpr] if fpr < FPR_CAP * 0.97 else [0.0, 0.0]
            )

        # ── TPR panel (row 0) ──────────────────────────────────────────────────
        ax = axes[0, col]
        ax.axhspan(pm_row - 0.44, pm_row + 0.44,
                   color=_c("PITMonitor"), alpha=0.06, linewidth=0, zorder=0)
        for i, (m, tpr, ci) in enumerate(zip(ordered, tprs, tpr_cis)):
            c = _c(m)
            ax.plot([0, tpr], [i, i], color=c, lw=0.9, alpha=0.38, solid_capstyle="round")
            ax.errorbar(
                tpr, i, xerr=[[ci[0]], [ci[1]]],
                fmt=_m(m), color=c,
                markersize=6.5 if m == "PITMonitor" else 4.5,
                markeredgewidth=0,
                elinewidth=0.7, capsize=1.8, capthick=0.7,
                zorder=5 if m == "PITMonitor" else 3,
            )
        ax.set_xlim(0, 1.06)
        ax.set_yticks(y)
        ax.set_ylim(len(ordered) - 0.4, -0.6)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))
        ax.set_title(SCENARIO_LABELS.get(scen, scen), pad=5, fontsize=9)
        if is_left:
            ax.set_yticklabels(ordered, fontsize=7)
            ax.set_ylabel("True positive rate", labelpad=3)
        else:
            ax.set_yticklabels([])

        # ── FPR panel (row 1) ──────────────────────────────────────────────────
        ax = axes[1, col]
        ax.axhspan(pm_row - 0.44, pm_row + 0.44,
                   color=_c("PITMonitor"), alpha=0.06, linewidth=0, zorder=0)
        for i, (m, fpr_d, fpr_raw, ci) in enumerate(
            zip(ordered, fprs_disp, fprs_raw, fpr_cis)
        ):
            c = _c(m)
            ax.plot([0, fpr_d], [i, i], color=c, lw=0.9, alpha=0.38, solid_capstyle="round")
            ax.errorbar(
                fpr_d, i, xerr=[[ci[0]], [ci[1]]],
                fmt=_m(m), color=c,
                markersize=6.5 if m == "PITMonitor" else 4.5,
                markeredgewidth=0,
                elinewidth=0.7, capsize=1.8, capthick=0.7,
                zorder=5 if m == "PITMonitor" else 3,
            )
        ax.axvline(alpha, color="0.25", ls="--", lw=0.9, alpha=0.8,
                   label=f"α = {alpha}")
        ax.set_xlim(0, FPR_CAP * 1.65)
        ax.set_yticks(y)
        ax.set_ylim(len(ordered) - 0.4, -0.6)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))
        if is_left:
            ax.set_yticklabels(ordered, fontsize=7)
            ax.set_ylabel("False positive rate", labelpad=3)
            ax.legend(fontsize=6.5, loc="upper right")
        else:
            ax.set_yticklabels([])

    fig.suptitle("Drift Detection: True and False Positive Rates",
                 fontsize=10, fontweight="bold")
    out = save_dir / "fig_detection_rates.png"
    _save(fig, out)
    return fig


# ── Figure 2: Detection delay distributions ───────────────────────────────────

def plot_delay_distributions(results: dict, save_dir: Path) -> plt.Figure:
    """Horizontal violin + box overlay for detection delays (true positives only).

    Uses horizontal orientation for better method-label readability.
    Methods use the global canonical ordering (by mean TPR on non-LEA scenarios)
    so the row assignment is identical to fig_detection_rates.
    Violin widths are scaled by sqrt(n_TP / max_n_TP) to warn the reader about
    methods whose distribution rests on very few events (e.g. PageHinkley).
    Zero-detection methods show a slim bar at x=0 in lieu of a violin.
    Sample counts annotated on the right margin of each row.
    """
    _apply_style()
    scenarios = [k for k in SCENARIO_LABELS if k in results["results"]]
    if not scenarios:
        scenarios = list(results["results"].keys())

    ordered = _canonical_order(results)
    y = np.arange(len(ordered))
    pm_row = ordered.index("PITMonitor")
    n = len(scenarios)

    fig, axes = plt.subplots(
        1, n,
        figsize=(4.0 * n, 6.2),
        gridspec_kw={"wspace": 0.55},
    )
    if n == 1:
        axes = [axes]

    for ax, scen in zip(axes, scenarios):
        scen_label = SCENARIO_LABELS.get(scen, scen)

        method_delays = {
            m: np.array(results["results"][scen].get(m, {}).get("delays", []))
            for m in ordered
        }

        # PITMonitor highlight band
        ax.axhspan(pm_row - 0.44, pm_row + 0.44,
                   color=_c("PITMonitor"), alpha=0.06, linewidth=0, zorder=0)

        violin_data, violin_pos, violin_cols, violin_ndets = [], [], [], []
        strip_data,  strip_pos,  strip_cols               = [], [], []
        no_det_indices                                      = []

        for i, m in enumerate(ordered):
            d = method_delays[m]
            c = _c(m)
            if len(d) == 0:
                no_det_indices.append(i)
            elif len(d) < 5 or float(np.std(d)) < 1.0:
                strip_data.append(d)
                strip_pos.append(i)
                strip_cols.append(c)
            else:
                violin_data.append(d)
                violin_pos.append(i)
                violin_cols.append(c)
                violin_ndets.append(len(d))

        # ── Violin plots with width ∝ sqrt(n_TP) ──────────────────────────────
        x_right_violin = 0.0
        if violin_data:
            max_n = max(violin_ndets)
            widths = [max(0.65, np.sqrt(nd / max_n)) * 0.72 for nd in violin_ndets]
            parts = ax.violinplot(
                violin_data, positions=violin_pos,
                vert=False, showmedians=False, showextrema=False,
                widths=widths,
            )
            for body, col in zip(parts["bodies"], violin_cols):
                body.set_facecolor(col)
                body.set_alpha(0.20)
                body.set_edgecolor(col)
                body.set_linewidth(0.9)
                # Track actual violin x-extent (KDE can extend past data range)
                verts = body.get_paths()[0].vertices
                x_right_violin = max(x_right_violin, float(verts[:, 0].max()))

            # IQR box + white median line
            bp = ax.boxplot(
                violin_data, positions=violin_pos,
                vert=False, patch_artist=True, widths=0.22,
                showfliers=False,
                medianprops={"color": "white", "linewidth": 2.0,
                             "solid_capstyle": "round"},
                whiskerprops={"linewidth": 0.7, "color": "0.5"},
                capprops={"linewidth": 0.7, "color": "0.5"},
                boxprops={"linewidth": 0.6},
            )
            for patch, col in zip(bp["boxes"], violin_cols):
                patch.set_facecolor(col)
                patch.set_alpha(0.85)

        # ── Strip plots for low-variance / small-n methods ────────────────────
        x_right_strip = 0.0
        rng = np.random.default_rng(42)
        for d, i, col in zip(strip_data, strip_pos, strip_cols):
            jitter = rng.normal(0, 0.10, size=len(d))
            ax.scatter(d, i + jitter, s=18, alpha=0.55, color=col,
                       zorder=4, rasterized=True)
            med = float(np.median(d))
            ax.plot([med, med], [i - 0.18, i + 0.18], color=col, lw=2.2,
                    solid_capstyle="round", zorder=5)
            x_right_strip = max(x_right_strip, float(np.max(d)))

        # ── No-detection: thin bar at x=0 ────────────────────────────────────
        # A clean vertical tick mark — visually distinct from the violin shapes
        # so it is not mistaken for a distribution of delays at zero.
        for i in no_det_indices:
            m = ordered[i]
            c = _c(m)
            ax.vlines(0, i - 0.32, i + 0.32, lw=1.5, colors=c,
                      alpha=0.80, zorder=6)

        # ── X-axis limits from actual rendered violin extent ──────────────────
        x_right_data = max(x_right_violin, x_right_strip)
        if x_right_data == 0.0:
            all_flat = np.concatenate(
                [d for d in method_delays.values() if len(d) > 0]
            )
            x_right_data = float(np.percentile(all_flat, 99.5)) if len(all_flat) > 0 else 100.0
        right = x_right_data * 1.06
        ax.set_xlim(left=-right * 0.02, right=right)

        # ── Sample-count annotations on right margin ──────────────────────────
        trans_ay = blended_transform_factory(ax.transAxes, ax.transData)
        for i, m in enumerate(ordered):
            n_tp = len(method_delays[m])
            label = f"n={n_tp:,}" if n_tp > 0 else "n=0"
            ax.text(1.02, i, label, transform=trans_ay,
                    va="center", ha="left", fontsize=5.5,
                    color=_c(m), alpha=0.85, clip_on=False)

        ax.set_yticks(y)
        ax.set_yticklabels(ordered, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Detection delay (samples)")
        ax.set_title(scen_label)
        ax.set_ylim(len(ordered) - 0.5, -0.5)

    fig.suptitle("Detection Delay Distributions (True Positives Only)",
                 fontsize=10, fontweight="bold")
    out = save_dir / "fig_delay_distributions.png"
    _save(fig, out)
    return fig


# ── Figure 3: Changepoint estimation error ────────────────────────────────────

def plot_cp_error_distribution(
    results: dict, save_dir: Path
) -> Optional[plt.Figure]:
    """Histogram of PITMonitor changepoint estimation error across scenarios.

    Scenarios with no PITMonitor true-positive detections (e.g. LEA) are omitted.
    Mean (red dashed) and median (blue dotted) are shown as vertical lines.
    """
    _apply_style()
    scenarios = [k for k in SCENARIO_LABELS if k in results["results"]]
    if not scenarios:
        scenarios = list(results["results"].keys())

    plot_data = []
    for scen in scenarios:
        pm = results["results"][scen].get("PITMonitor", {})
        errs = pm.get("cp_errors", [])
        if errs:
            plot_data.append({
                "label":  SCENARIO_LABELS.get(scen, scen),
                "errors": np.array(errs),
                "mean":   pm.get("mean_cp_error", float("nan")),
                "median": pm.get("median_cp_error", float("nan")),
            })

    if not plot_data:
        print("  Skipped fig_cp_error_distribution.png (no CP data)")
        return None

    n = len(plot_data)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 3.5),
                             gridspec_kw={"wspace": 0.45},
                             squeeze=False)
    fig.subplots_adjust(top=0.82)
    axes = axes[0]

    c_bar    = _c("PITMonitor")
    C_MEAN   = "#C0392B"   # red  — mean
    C_MEDIAN = "#0072B2"   # blue — median

    for ax, data in zip(axes, plot_data):
        errs = data["errors"]
        p995 = int(np.ceil(np.percentile(errs, 99.5)))
        bins = np.arange(-0.5, p995 + 1.5, 1)

        ax.hist(errs, bins=bins, color=c_bar, alpha=0.75,
                edgecolor="white", linewidth=0.4, density=False, zorder=2)

        ax.axvline(data["mean"],   color=C_MEAN,   ls="--", lw=1.2,
                   label=f"Mean = {data['mean']:.1f}")
        ax.axvline(data["median"], color=C_MEDIAN, ls=":",  lw=1.6,
                   label=f"Median = {data['median']:.1f}")

        ax.set_xlabel(r"Changepoint error $|\hat{\tau} - \tau|$ (samples)")
        ax.set_ylabel("Count")
        ax.set_title(data["label"])
        ax.set_xlim(-0.5, p995 + 0.5)
        ax.legend(fontsize=6.5, handlelength=1.2)

    fig.suptitle("PITMonitor Changepoint Estimation Error",
                 fontsize=10, fontweight="bold")
    out = save_dir / "fig_cp_error_distribution.png"
    _save(fig, out)
    return fig


# ── Figure 4: Single-run diagnostic panels ────────────────────────────────────

def plot_single_run_panels(
    artifacts: dict,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Four-panel diagnostic for a single monitoring run.

    Panels:
        (a) Predictions vs actual — scatter coloured by pre/post drift region.
        (b) PIT stream — dots coloured pre/post + 50-sample rolling mean.
            Only the true-shift line is shown here; alarm/CP appear in (c).
        (c) E-process — log scale with threshold / alarm / CP-estimate lines,
            each annotated with a staggered label to disambiguate close events.
        (d) PIT distributions — overlapping density histograms pre vs post shift.
    """
    _apply_style()

    shift = int(artifacts["true_shift_point"])
    y_all = np.asarray(artifacts["true_labels"],    dtype=float)
    preds = np.asarray(artifacts["predictions"],    dtype=float)
    pits  = np.asarray(artifacts["pits"],           dtype=float)
    evid  = np.asarray(artifacts["evidence_trace"], dtype=float)
    t     = np.arange(1, len(y_all) + 1)

    scen_label = SCENARIO_LABELS.get(artifacts.get("scenario_key", ""), "")

    C_PRE   = "#4A90D9"   # pre-drift: cool blue
    C_POST  = "#C0392B"   # post-drift: warm red
    C_SHIFT = "#2C3E50"   # true shift: near-black
    C_ALARM = "#E67E22"   # alarm: orange
    C_CP    = "#27AE60"   # changepoint estimate: green
    C_EVID  = _c("PITMonitor")

    alarm_fired = bool(artifacts.get("alarm_fired", False))
    alarm_time  = artifacts.get("alarm_time")
    cp_est      = artifacts.get("changepoint")
    threshold   = 1.0 / float(artifacts["monitor_alpha"])

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.6))
    fig.subplots_adjust(top=0.92, hspace=0.52, wspace=0.40)
    fig.suptitle(f"PITMonitor Single Run — {scen_label}",
                 fontsize=10, fontweight="bold")

    # ── (a) Predictions vs reality ────────────────────────────────────────────
    ax = axes[0, 0]
    ax.axvspan(t[0],  shift,  color=C_PRE,  alpha=0.035, linewidth=0)
    ax.axvspan(shift, t[-1], color=C_POST, alpha=0.035, linewidth=0)

    mask = t < shift
    ax.scatter(t[mask],  y_all[mask],  s=2.0, alpha=0.20, c=C_PRE,  rasterized=True)
    ax.scatter(t[~mask], y_all[~mask], s=2.0, alpha=0.20, c=C_POST, rasterized=True)
    ax.scatter(t, preds, s=1.5, alpha=0.15, c="0.40", rasterized=True)

    ax.axvline(shift, color=C_SHIFT, ls="--", lw=1.4, alpha=0.85, zorder=2)
    if alarm_fired and alarm_time is not None:
        ax.axvline(int(alarm_time), color=C_ALARM, ls="-", lw=1.8, alpha=0.85, zorder=2)

    leg = [
        Line2D([0], [0], marker="o", ls="", color=C_PRE,  ms=3.5, label="Actual (pre)"),
        Line2D([0], [0], marker="o", ls="", color=C_POST, ms=3.5, label="Actual (post)"),
        Line2D([0], [0], color=C_SHIFT, ls="--", lw=1.4, label="True shift"),
    ]
    if alarm_fired and alarm_time is not None:
        leg.append(Line2D([0], [0], color=C_ALARM, ls="-", lw=1.8,
                          label=f"Alarm (t={int(alarm_time)})"))
    ax.legend(handles=leg, fontsize=6, loc="upper right",
              ncol=2, columnspacing=0.6, handlelength=1.2)
    ax.set(xlabel="Sample", ylabel="Target value", title="(a) Predictions vs reality")

    # ── (b) PIT stream ────────────────────────────────────────────────────────
    # Separate scatter calls (not a colour array) prevent rasterisation
    # artefacts at the panel bottom.  Only the true-shift line is drawn here;
    # the alarm and CP are shown in panel (c) to keep this panel clean.
    ax = axes[0, 1]
    mask = t < shift
    ax.scatter(t[mask],  pits[mask],  s=2.0, alpha=0.09, c=C_PRE,
               rasterized=False, linewidths=0, zorder=3)
    ax.scatter(t[~mask], pits[~mask], s=2.0, alpha=0.09, c=C_POST,
               rasterized=False, linewidths=0, zorder=3)

    w = 50
    if len(pits) >= w:
        roll = np.convolve(pits, np.ones(w) / w, mode="valid")
        ax.plot(np.arange(w, len(pits) + 1), roll,
                color="0.10", lw=1.1, alpha=0.90, zorder=4)

    ax.axhline(0.5, color="0.55", ls=":", lw=0.8)

    # True-shift line only — alarm omitted (alarm & CP are shown fully in panel c)
    ax.axvline(shift, color=C_SHIFT, ls="--", lw=1.4, alpha=0.85, zorder=2)

    leg = [
        Line2D([0], [0], marker="o", ls="", color=C_PRE,  ms=3.5, label="Pre-shift"),
        Line2D([0], [0], marker="o", ls="", color=C_POST, ms=3.5, label="Post-shift"),
        Line2D([0], [0], color="0.10", lw=1.1, label="Roll. mean (w=50)"),
        Line2D([0], [0], color=C_SHIFT, ls="--", lw=1.2, label="True shift"),
    ]
    ax.legend(handles=leg, fontsize=6, loc="upper right",
              ncol=2, columnspacing=0.6, handlelength=1.2)
    ax.set(xlabel="Sample", ylabel="PIT value",
           title="(b) PIT stream")
    ax.set_ylim(0, 1)

    # ── (c) E-process ─────────────────────────────────────────────────────────
    ax = axes[1, 0]
    evid_pos = np.maximum(evid, 1e-12)

    ax.semilogy(t, evid_pos, color=C_EVID, lw=1.1, zorder=4)
    ax.fill_between(t, threshold, evid_pos,
                    where=evid_pos >= threshold,
                    color=C_EVID, alpha=0.12, linewidth=0)

    # Horizontal threshold line
    ax.axhline(threshold, color=C_ALARM, ls="--", lw=1.0, alpha=0.85)

    # Vertical event lines — three visually distinct styles.
    # Always draw shift and CP separately (even when they coincide spatially)
    # so the legend structure is identical across all single-run figures.
    leg_c = [Line2D([0], [0], color=C_ALARM, ls="--", lw=1.0,
                    label=f"Threshold 1/α = {threshold:.0f}")]

    ax.axvline(shift, color=C_SHIFT, ls="--", lw=1.4, alpha=0.85, zorder=2)
    leg_c.append(Line2D([0], [0], color=C_SHIFT, ls="--", lw=1.4,
                         label=f"True shift (t={shift})"))

    if cp_est is not None:
        cp_int = int(cp_est)
        ax.axvline(cp_int, color=C_CP, ls=":", lw=2.0, alpha=0.90, zorder=2)
        leg_c.append(Line2D([0], [0], color=C_CP, ls=":", lw=2.0,
                             label=f"CP est. (t={cp_int})"))

    if alarm_fired and alarm_time is not None:
        ax.axvline(int(alarm_time), color=C_ALARM, ls="-", lw=1.8, alpha=0.85, zorder=2)
        leg_c.append(Line2D([0], [0], color=C_ALARM, ls="-", lw=1.8,
                             label=f"Alarm (t={int(alarm_time)})"))

    ax.legend(handles=leg_c, fontsize=6, loc="upper left", handlelength=1.4)

    valid = evid_pos[evid_pos > 1e-11]
    y_bot = float(valid.min()) * 0.3 if len(valid) > 0 else 1e-8
    y_top = max(float(evid_pos.max()), threshold) * 4.0
    ax.set_ylim(y_bot, y_top)
    ax.set_xlim(t[0], t[-1])
    ax.set(xlabel="Sample", ylabel="Evidence (log scale)", title="(c) E-process")

    # ── (d) PIT distributions ─────────────────────────────────────────────────
    ax = axes[1, 1]
    bins = np.linspace(0, 1, 21)
    pre_pits  = pits[:shift - 1]
    post_pits = pits[shift - 1:]
    kw = dict(bins=bins, density=True, edgecolor="white", linewidth=0.4)
    if len(pre_pits) > 0:
        ax.hist(pre_pits,  alpha=0.55, color=C_PRE,
                label=f"Pre-shift  (n={len(pre_pits):,})", **kw)
    if len(post_pits) > 0:
        ax.hist(post_pits, alpha=0.55, color=C_POST,
                label=f"Post-shift (n={len(post_pits):,})", **kw)
    ax.axhline(1.0, color="0.35", ls=":", lw=1.0, label="U[0,1] reference")
    ax.legend(fontsize=6.5)
    ax.set(xlabel="PIT value", ylabel="Density",
           title="(d) PIT distributions", xlim=(0, 1))

    if save_path is not None:
        save_path = Path(save_path)
        _save(fig, save_path)
    else:
        plt.close(fig)
    return fig


# ── LaTeX table ───────────────────────────────────────────────────────────────

def generate_latex_table(results: dict, save_dir: Path) -> str:
    r"""Generate a LaTeX booktabs table with all summary statistics.

    Columns per scenario: TPR, FPR, Mean Delay.

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

    col_spec = "l" + " ccc" * n_scen

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Drift detection results on FriedmanDrift "
        rf"(\expNtrials{{}} trials, $\alpha=\expAlpha$). "
        r"Mean delay is in samples. "
        r"Best TPR per scenario is \textbf{bolded}; "
        r"FPR exceeding $\alpha$ is \underline{underlined}.}"
    )
    lines.append(r"\label{tab:results}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    header1_parts = [r"\multicolumn{1}{l}{}"]
    for scen in scenarios:
        label = SCENARIO_SHORT.get(scen, scen)
        header1_parts.append(rf"\multicolumn{{3}}{{c}}{{{label}}}")
    lines.append(" & ".join(header1_parts) + r" \\")

    cmidrules = []
    for i in range(n_scen):
        start = 2 + i * 3
        end = start + 2
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
    lines.append(" ".join(cmidrules))

    header2_parts = ["Method"]
    for _ in scenarios:
        header2_parts.extend(["TPR", "FPR", "Delay"])
    lines.append(" & ".join(header2_parts) + r" \\")
    lines.append(r"\midrule")

    best_tpr = {}
    for scen in scenarios:
        tprs = [results["results"][scen].get(m, {}).get("tpr", 0.0)
                for m in ALL_DETECTOR_NAMES]
        best_tpr[scen] = max(tprs)

    for method in ALL_DETECTOR_NAMES:
        row_parts = [method.replace("_", r"\_")]
        for scen in scenarios:
            s = results["results"][scen].get(method, {})
            tpr   = s.get("tpr",        float("nan"))
            fpr   = s.get("fpr",        float("nan"))
            delay = s.get("mean_delay", float("nan"))

            tpr_str = _fmt_pct(tpr)
            if not np.isnan(tpr) and abs(tpr - best_tpr[scen]) < 1e-6:
                tpr_str = rf"\textbf{{{tpr_str}}}"

            fpr_str = _fmt_pct(fpr)
            if not np.isnan(fpr) and fpr > alpha:
                fpr_str = rf"\underline{{{fpr_str}}}"

            row_parts.extend([tpr_str, fpr_str, _fmt_delay(delay)])

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


# ── LaTeX macros ──────────────────────────────────────────────────────────────

def generate_experiment_macros(results: dict, save_dir: Path) -> str:
    r"""Generate a LaTeX file of \newcommand macros for experiment params and results.

    Parameters
    ----------
    results : dict
    save_dir : Path

    Returns
    -------
    str
        Complete LaTeX macro file source.
    """
    cfg = results["config"]
    lines = [
        "% Auto-generated by plots.py — do not edit manually.",
        "% Experiment configuration macros",
        rf"\newcommand{{\expSeed}}{{{cfg['seed']}}}",
        rf"\newcommand{{\expEpochs}}{{{cfg['epochs']:,}}}".replace(",", "{,}"),
        rf"\newcommand{{\expLr}}{{3 \times 10^{{-4}}}}",
        rf"\newcommand{{\expNtrain}}{{{cfg['n_train']:,}}}".replace(",", "{,}"),
        rf"\newcommand{{\expNstable}}{{{cfg['n_stable']:,}}}".replace(",", "{,}"),
        rf"\newcommand{{\expNpost}}{{{cfg['n_post']:,}}}".replace(",", "{,}"),
        rf"\newcommand{{\expAlpha}}{{{cfg['alpha']}}}",
        rf"\newcommand{{\expNbins}}{{{cfg['n_bins']}}}",
        rf"\newcommand{{\expNtrials}}{{{cfg['n_trials']:,}}}".replace(",", "{,}"),
        rf"\newcommand{{\expBinaryThreshold}}{{{cfg['binary_threshold']:.4f}}}",
        "",
        "% Per-scenario result macros (PITMonitor)",
    ]

    res = results["results"]
    for scen_key, short in SCENARIO_SHORT.items():
        tag = short
        pm = res.get(scen_key, {}).get("PITMonitor", {})
        if not pm:
            continue
        tpr    = pm.get("tpr",        float("nan"))
        fpr    = pm.get("fpr",        float("nan"))
        delay  = pm.get("mean_delay", float("nan"))
        cp_err = pm.get("mean_cp_error", float("nan"))

        lines.append(f"% {tag}")
        lines.append(rf"\newcommand{{\resPM{tag}TPR}}{{{tpr:.1%}}}".replace("%", r"\%"))
        lines.append(rf"\newcommand{{\resPM{tag}FPR}}{{{fpr:.1%}}}".replace("%", r"\%"))
        if not np.isnan(delay):
            lines.append(rf"\newcommand{{\resPM{tag}Delay}}{{{delay:.0f}}}")
        if not np.isnan(cp_err):
            lines.append(rf"\newcommand{{\resPM{tag}CPErr}}{{{cp_err:.1f}}}")

    lines.append("")
    lines.append("% Per-scenario result macros (ADWIN)")
    for scen_key, short in SCENARIO_SHORT.items():
        tag = short
        ad = res.get(scen_key, {}).get("ADWIN", {})
        if not ad:
            continue
        tpr   = ad.get("tpr",        float("nan"))
        fpr   = ad.get("fpr",        float("nan"))
        delay = ad.get("mean_delay", float("nan"))
        lines.append(f"% {tag}")
        lines.append(rf"\newcommand{{\resAD{tag}TPR}}{{{tpr:.1%}}}".replace("%", r"\%"))
        lines.append(rf"\newcommand{{\resAD{tag}FPR}}{{{fpr:.1%}}}".replace("%", r"\%"))
        if not np.isnan(delay):
            lines.append(rf"\newcommand{{\resAD{tag}Delay}}{{{delay:.0f}}}")

    lines.append("")
    lines.append("% Other detectors mentioned in text")
    for det_name, prefix in [("DDM", "DDM"), ("HDDM_A", "HDDMA")]:
        for scen_key, short in SCENARIO_SHORT.items():
            d = res.get(scen_key, {}).get(det_name, {})
            if not d:
                continue
            tpr   = d.get("tpr",        float("nan"))
            fpr   = d.get("fpr",        float("nan"))
            delay = d.get("mean_delay", float("nan"))
            tag   = short
            lines.append(
                rf"\newcommand{{\res{prefix}{tag}TPR}}{{{tpr:.1%}}}".replace("%", r"\%")
            )
            lines.append(
                rf"\newcommand{{\res{prefix}{tag}FPR}}{{{fpr:.1%}}}".replace("%", r"\%")
            )
            if not np.isnan(delay):
                lines.append(rf"\newcommand{{\res{prefix}{tag}Delay}}{{{delay:.0f}}}")

    latex_src = "\n".join(lines) + "\n"
    out = save_dir / "experiment_macros.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex_src)
    print(f"  Saved {out}")
    return latex_src


# ── Master ────────────────────────────────────────────────────────────────────

def make_all_plots(results: dict, save_dir: Path) -> None:
    """Generate all publication figures and the LaTeX table from saved results."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating plots:")
    plot_detection_rates(results, save_dir)
    plot_delay_distributions(results, save_dir)
    plot_cp_error_distribution(results, save_dir)
    generate_latex_table(results, save_dir)
    generate_experiment_macros(results, save_dir)
    for scen_key, artifacts in results.get("single_runs", {}).items():
        out_path = save_dir / f"fig_single_run_{scen_key}.png"
        plot_single_run_panels(artifacts, save_path=out_path)
    print("Done.")
