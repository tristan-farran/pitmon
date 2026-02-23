from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pitmon import PITMonitor


def plot_single_run_panels(artifacts: dict, save_path: Path | None = None) -> None:
    single = artifacts["single_run"]
    cfg = artifacts["config"]

    true_labels = np.asarray(single["true_labels"])
    pred_labels = np.asarray(single["pred_labels"])
    pits = np.asarray(single["pits"], dtype=float)
    evidence_trace = np.asarray(single["evidence_trace"], dtype=float)
    true_shift_point = int(single["true_shift_point"])
    alpha = float(single["monitor_alpha"])
    t = np.arange(1, len(pits) + 1)

    # Determine changepoint: prefer stored value, fall back to recomputing
    changepoint = single.get("changepoint")
    if changepoint is None:
        try:
            monitor_tmp = PITMonitor(alpha=alpha, n_bins=int(cfg["n_bins"]))
            monitor_tmp.update_many(pits, stop_on_alarm=False)
            changepoint = monitor_tmp.changepoint()
        except Exception:
            changepoint = None

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"PITMonitor: CIFAR-10 → CIFAR-10-C ({cfg['corruption']}, severity={single['severity']})",
        fontsize=13,
        fontweight="bold",
    )

    # Panel 1: Predicted classes colored by correctness
    ax = axes[0, 0]
    is_correct = pred_labels == true_labels
    point_colors = np.where(is_correct, "darkgreen", "crimson")
    ax.scatter(
        t,
        pred_labels,
        s=9,
        alpha=0.5,
        c=point_colors,
        label="Prediction (green=correct, red=wrong)",
        zorder=2,
    )
    ax.axvline(
        true_shift_point,
        color="red",
        ls=":",
        lw=1.5,
        alpha=0.8,
        label=f"True shift (t={true_shift_point})",
    )
    if single["alarm_fired"] and single["alarm_time"] is not None:
        ax.axvline(
            int(single["alarm_time"]),
            color="orange",
            ls="--",
            lw=1.5,
            label=f"Alarm (t={single['alarm_time']})",
        )
    ax.set_ylabel("Predicted class", fontsize=11)
    ax.set_title("Predictions (Correct vs Wrong)", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.5, 9.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(left=0, right=len(t))

    # Panel 2: PIT scatter with rolling mean
    ax = axes[0, 1]
    colors = np.where(t <= true_shift_point, "steelblue", "crimson")
    ax.scatter(t, pits, s=5, alpha=0.4, c=colors)
    if len(pits) >= 30:
        rolling_pit = np.convolve(pits, np.ones(30) / 30, mode="valid")
        ax.plot(
            np.arange(30, len(pits) + 1),
            rolling_pit,
            color="black",
            lw=1.5,
            alpha=0.6,
            label="Rolling mean (w=30)",
        )
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.4)
    ax.axvline(true_shift_point, color="red", ls=":", lw=1.5, alpha=0.8)
    if single["alarm_fired"] and single["alarm_time"] is not None:
        ax.axvline(int(single["alarm_time"]), color="orange", ls="--", lw=1.5)
    ax.set_ylabel("PIT value", fontsize=11)
    ax.set_title("PIT Stream", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(left=0, right=len(t))

    # Panel 3: Log-evidence trace with alarm / true shift
    ax = axes[1, 0]
    ax.semilogy(t, np.maximum(evidence_trace, 1e-10), color="steelblue", lw=2)
    threshold = 1.0 / alpha
    ax.axhline(
        threshold,
        color="crimson",
        ls="--",
        lw=2,
        label=f"Threshold (1/α = {threshold:.0f})",
    )
    ax.axvline(
        true_shift_point, color="red", ls=":", lw=1.5, alpha=0.8, label="True shift"
    )
    if changepoint is not None:
        ax.axvline(
            int(changepoint),
            color="green",
            ls="--",
            lw=1.5,
            alpha=0.7,
            label=f"Est. changepoint (t≈{changepoint})",
        )
    if single["alarm_fired"] and single["alarm_time"] is not None:
        ax.axvline(
            int(single["alarm_time"]),
            color="orange",
            ls="--",
            lw=2,
            label=f"Alarm (t={single['alarm_time']})",
        )
    ax.set_ylabel("Log-evidence", fontsize=11)
    ax.set_xlabel("Observation", fontsize=11)
    ax.set_title("PITMonitor Evidence", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2, which="both")
    ax.set_xlim(left=0, right=len(t))

    # Panel 4: PIT distributions pre/post shift
    ax = axes[1, 1]
    hist_bins = np.linspace(0, 1, 21)
    ax.hist(
        pits[: true_shift_point - 1],
        bins=hist_bins,
        density=True,
        alpha=0.5,
        color="steelblue",
        edgecolor="white",
        label="Pre-shift",
    )
    ax.hist(
        pits[true_shift_point - 1 :],
        bins=hist_bins,
        density=True,
        alpha=0.5,
        color="crimson",
        edgecolor="white",
        label="Post-shift",
    )
    ax.axhline(1.0, color="black", ls="--", lw=1.5, label="Ideal (Uniform)")
    ax.set_xlabel("PIT value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("PIT Distributions", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_power_panels(artifacts: dict, save_path: Path | None = None) -> None:
    cfg = artifacts["config"]
    severity_levels = list(cfg["severity_levels"])
    h0 = artifacts["h0_results_by_severity"]
    h1 = artifacts["power_results_by_severity"]

    h0_fprs = [h0[level]["false_alarm_rate"] for level in severity_levels]
    h1_tprs = [h1[level]["tpr"] for level in severity_levels]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "PITMonitor Power Analysis: Detection Reliability Across Corruption Severities",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    ax = axes[0, 0]
    ax.plot(severity_levels, h1_tprs, "o-", color="steelblue", lw=2.5, markersize=8)
    ax.set(
        xlabel="CIFAR-10-C Severity",
        ylabel="Detection Rate (TPR)",
        title="Power Curve: Detection Rate vs. Corruption Severity",
    )
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.2)

    ax = axes[0, 1]
    selected = [level for level in [1, 3, 5] if level in h1]
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(selected)))
    for level, color in zip(selected, colors):
        delays = np.asarray(h1[level]["delays"], dtype=float)
        if delays.size == 0:
            continue
        delays_sorted = np.sort(delays)
        ecdf = np.arange(1, len(delays_sorted) + 1) / len(delays_sorted)
        ax.step(
            delays_sorted,
            ecdf,
            where="post",
            color=color,
            lw=2,
            alpha=0.85,
            label=f"Severity {level} (n={len(delays)})",
        )
    ax.set(
        xlabel="Detection delay (obs after shift)",
        ylabel="Cumulative probability",
        title="Detection Latency ECDF by Severity",
    )
    ax.set_ylim(0.0, 1.02)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)

    ax = axes[1, 0]
    ax.plot(
        severity_levels,
        h0_fprs,
        "s-",
        color="mediumpurple",
        lw=2.5,
        markersize=7,
        label="False Positive Rate",
    )
    ax.axhline(
        cfg["alpha_power"],
        color="crimson",
        ls="--",
        lw=1.5,
        label=f"Ville bound (α={cfg['alpha_power']:.2f})",
    )
    ax.fill_between(
        severity_levels,
        0,
        [cfg["alpha_power"]] * len(severity_levels),
        color="crimson",
        alpha=0.06,
    )
    ax.set(
        xlabel="CIFAR-10-C Severity",
        ylabel="False alarm rate (FPR)",
        title="Ville Bound Check: FPR Control",
    )
    ax.set_ylim(-0.01, min(0.15, cfg["alpha_power"] * 3.0))
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)

    ax = axes[1, 1]
    selected = [level for level in [1, 3, 5] if level in h1]
    colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(selected)))
    for level, color in zip(selected, colors):
        evidences = np.asarray(h1[level]["evidences"], dtype=float)
        if evidences.size == 0:
            continue
        ax.hist(
            evidences,
            bins=12,
            density=True,
            alpha=0.35,
            color=color,
            edgecolor="white",
            label=f"Severity {level} (n={len(evidences)})",
        )
    ax.set(
        xlabel="Final evidence",
        ylabel="Density",
        title="Final Evidence Distribution by Severity",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_comparison_panels(artifacts: dict, save_path: Path | None = None) -> None:
    rows = artifacts["comparison_rows"]
    cfg = artifacts["config"]
    severities = list(cfg["severity_levels"])
    methods = ["PITMonitor", "DDM", "EDDM", "ADWIN", "KSWIN"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for method in methods:
        ys_fpr, ys_tpr, ys_delay = [], [], []
        for severity in severities:
            row = next(
                row
                for row in rows
                if row["severity"] == severity and row["method"] == method
            )
            ys_fpr.append(row["false_alarm_rate"])
            ys_tpr.append(row["tpr"])
            ys_delay.append(row["median_delay"])

        linestyle = "-" if method == "PITMonitor" else "--"
        marker = "s" if method == "PITMonitor" else "o"
        linewidth = 2.5 if method == "PITMonitor" else 2
        axes[0].plot(
            severities, ys_fpr, marker=marker, lw=linewidth, ls=linestyle, label=method
        )
        axes[1].plot(
            severities, ys_tpr, marker=marker, lw=linewidth, ls=linestyle, label=method
        )
        axes[2].plot(
            severities,
            ys_delay,
            marker=marker,
            lw=linewidth,
            ls=linestyle,
            label=method,
        )

    axes[0].axhline(
        cfg["alpha_power"],
        color="crimson",
        ls=":",
        lw=1.5,
        label=f"Ville bound (α={cfg['alpha_power']:.2f})",
        alpha=0.6,
    )
    axes[0].set(
        xlabel="CIFAR-10-C severity",
        ylabel="False Alarm Rate",
        title="FPR Across Methods",
    )
    axes[0].set_ylim(-0.01, 0.15)
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(fontsize=9, loc="upper right")

    axes[1].set(
        xlabel="CIFAR-10-C severity",
        ylabel="True Positive Rate",
        title="TPR Across Methods",
    )
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(fontsize=9, loc="upper right")

    axes[2].set(
        xlabel="CIFAR-10-C severity",
        ylabel="Median Detection Delay (obs)",
        title="Delay Across Methods",
    )
    axes[2].set_ylim(bottom=0)
    axes[2].grid(True, alpha=0.2)
    axes[2].legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_baseline_h0_panels(artifacts: dict, save_path: Path | None = None) -> None:
    """Visualize baseline H0 false-alarm behavior of all methods.

    Uses clean→clean (no-shift) streams for the river baselines and the existing
    H0 simulations for PITMonitor aggregated across severities.
    """
    cfg = artifacts["config"]
    h0_by_severity = artifacts["h0_results_by_severity"]
    baseline = artifacts.get("baseline_h0_results_by_method", {})

    methods = ["PITMonitor", "DDM", "EDDM", "ADWIN", "KSWIN"]

    # Aggregate PITMonitor H0 FPR across severities.
    severities = list(cfg["severity_levels"])
    pit_fprs = [h0_by_severity[sev]["false_alarm_rate"] for sev in severities]
    pit_fpr_mean = float(np.mean(pit_fprs)) if pit_fprs else float("nan")

    fprs = []
    for method in methods:
        if method == "PITMonitor":
            fprs.append(pit_fpr_mean)
        else:
            stats = baseline.get(method)
            fprs.append(
                stats["false_alarm_rate"] if stats is not None else float("nan")
            )

    x = np.arange(len(methods))
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    bars = ax.bar(
        x, fprs, color=["tab:blue"] + ["tab:gray"] * (len(methods) - 1), alpha=0.85
    )
    ax.axhline(
        cfg["alpha_power"],
        color="crimson",
        ls="--",
        lw=1.5,
        label=f"Ville bound (α={cfg['alpha_power']:.2f})",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("False alarm rate under H₀")
    ax.set_ylim(0.0, max(0.2, np.nanmax(fprs) * 1.3 if fprs else 0.1))
    ax.set_title("Baseline H₀ Behavior: Clean → Clean (No Shift)")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(fontsize=9)

    # Annotate exact FPRs on top of bars for clarity.
    for rect, value in zip(bars, fprs):
        if np.isfinite(value):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height() + 0.005,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
