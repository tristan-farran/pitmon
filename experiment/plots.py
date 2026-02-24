"""Plots for the drift detection comparison.

Generates three main figures:
    1. Summary table — TPR / FPR / median delay per method × scenario
    2. Detection rate bar chart — grouped bars (one group per scenario)
    3. Delay distributions — box plots per method × scenario
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from detectors import ALL_DETECTOR_NAMES


# ─── Style constants ─────────────────────────────────────────────────

SCENARIO_LABELS = {
    "gra_tw0": "Abrupt (GRA)",
    "gsg_tw500": "Gradual (GSG)",
    "lea_tw0": "Local (LEA)",
}

METHOD_COLORS = {
    "PITMonitor": "#2489cd",
    "ADWIN": "#9e7ade",
    "KSWIN": "#41c0c4",
    "PageHinkley": "#68f496",
    "DDM": "#fec229",
    "EDDM": "#d95f0e",
    "HDDM_A": "#f768a1",
    "HDDM_W": "#c51b8a",
}

_RC_OVERRIDES = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
}


def _apply_style():
    plt.rcParams.update(_RC_OVERRIDES)


# ─── Figure 1: Detection rate comparison (grouped bar) ───────────────


def plot_detection_rates(results, save_dir):
    """Clear horizontal bar charts: one row per scenario."""
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

        methods = []
        tprs = []
        fprs = []
        tpr_err = []
        fpr_err = []
        colors = []

        for method in ALL_DETECTOR_NAMES:
            s = results["results"][scen].get(method, {})
            methods.append(method)

            tpr = s.get("tpr", 0)
            fpr = s.get("fpr", 0)

            tprs.append(tpr)
            fprs.append(fpr)

            ci = s.get("tpr_ci", (tpr, tpr))
            tpr_err.append([tpr - ci[0], ci[1] - tpr])

            ci = s.get("fpr_ci", (fpr, fpr))
            fpr_err.append([fpr - ci[0], ci[1] - fpr])

            colors.append(METHOD_COLORS.get(method, "#999999"))

        y = np.arange(len(methods))

        # ── TPR ──
        ax = axes[i, 0]
        ax.barh(y, tprs, xerr=np.abs(np.array(tpr_err).T), color=colors, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(methods)
        ax.set_xlim(0, 1.05)
        ax.set_title(f"{scen_label} — TPR")
        ax.grid(axis="x", alpha=0.3)

        # ── FPR ──
        ax = axes[i, 1]
        ax.barh(y, fprs, xerr=np.abs(np.array(fpr_err).T), color=colors, alpha=0.9)
        ax.axvline(alpha, color="crimson", ls="--", lw=1.5)
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.set_xlim(0, max(0.15, alpha * 3))
        ax.set_title(f"{scen_label} — FPR")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Drift Detection Comparison on FriedmanDrift", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    out = save_dir / "fig_detection_rates.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─── Figure 2: Detection delay box plots ────────────────────────────


def plot_delay_distributions(results, save_dir):
    """Cleaner boxplots: only plot real delay data."""
    _apply_style()

    scenarios = [k for k in results["results"] if k in SCENARIO_LABELS]
    if not scenarios:
        scenarios = list(results["results"].keys())

    n_scenarios = len(scenarios)

    fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 5), sharey=True)

    if n_scenarios == 1:
        axes = [axes]

    for ax, scen in zip(axes, scenarios):
        scen_label = SCENARIO_LABELS.get(scen, scen)

        delay_data = []
        labels = []
        colors = []

        for method in ALL_DETECTOR_NAMES:
            delays = results["results"][scen].get(method, {}).get("delays", [])
            if len(delays) > 0:
                delay_data.append(delays)
                labels.append(method)
                colors.append(METHOD_COLORS.get(method, "#999999"))

        if not delay_data:
            ax.set_title(f"{scen_label}\n(no detections)")
            continue

        bp = ax.boxplot(
            delay_data,
            patch_artist=True,
            widths=0.6,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(scen_label)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Detection Delay (samples)")

    fig.suptitle(
        "Detection Delay Distributions (True Positives Only)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    out = save_dir / "fig_delay_distributions.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─── Figure 3: Summary table (rendered as a figure) ─────────────────


def plot_summary_table(results, save_dir):
    """Render a formatted summary table as a figure."""
    _apply_style()
    scenarios = [k for k in results["results"] if k in SCENARIO_LABELS]
    if not scenarios:
        scenarios = list(results["results"].keys())

    # Build table data: rows = methods, columns grouped by scenario
    col_headers = []
    for scen in scenarios:
        label = SCENARIO_LABELS.get(scen, scen)
        col_headers.extend([f"{label}\nTPR", f"{label}\nFPR", f"{label}\nDelay"])

    cell_text = []
    row_labels = []
    cell_colors = []

    for method in ALL_DETECTOR_NAMES:
        row = []
        row_clr = []
        for scen in scenarios:
            s = results["results"][scen].get(method, {})
            tpr = s.get("tpr", float("nan"))
            fpr = s.get("fpr", float("nan"))
            delay = s.get("median_delay", float("nan"))

            row.append(f"{tpr:.1%}" if not np.isnan(tpr) else "—")
            row.append(f"{fpr:.1%}" if not np.isnan(fpr) else "—")
            row.append(f"{delay:.0f}" if not np.isnan(delay) else "—")

            # Colour coding: green for high TPR, red for high FPR
            tpr_clr = (0.85, 0.95, 0.85, 1.0) if tpr > 0.8 else (1, 1, 1, 1)
            alpha_val = results["config"]["alpha"]
            fpr_clr = (1.0, 0.85, 0.85, 1.0) if fpr > alpha_val else (1, 1, 1, 1)
            delay_clr = (1, 1, 1, 1)
            row_clr.extend([tpr_clr, fpr_clr, delay_clr])

        cell_text.append(row)
        cell_colors.append(row_clr)
        row_labels.append(method)

    n_cols = len(col_headers)
    fig_width = 3.2 * len(scenarios)
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

    # Bold header row
    for j in range(n_cols):
        table[0, j].set_text_props(fontweight="bold")
    # Bold row labels
    for i in range(len(ALL_DETECTOR_NAMES)):
        table[i + 1, -1].set_text_props(fontweight="bold")

    fig.suptitle(
        f"Drift Detection Results — {results['config']['n_trials']} trials, "
        f"α = {results['config']['alpha']}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    out = save_dir / "fig_summary_table"
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─── Master plot function ────────────────────────────────────────────


def make_all_plots(results, save_dir):
    """Generate all publication figures."""
    save_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating plots:")
    plot_detection_rates(results, save_dir)
    plot_delay_distributions(results, save_dir)
    plot_summary_table(results, save_dir)
    print("Done.")
