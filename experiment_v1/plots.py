from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_single_run_panels(artifacts: dict, save_path: Path | None = None) -> None:
    cfg = artifacts["config"]
    single = artifacts["single_run"]

    true_shift_point = int(single["true_shift_point"])
    y_all = np.asarray(single["true_labels"], dtype=float)
    preds = np.asarray(single["predictions"], dtype=float)
    pits = np.asarray(single["pits"], dtype=float)
    evidence = np.asarray(single["evidence_trace"], dtype=float)
    times = np.arange(1, len(y_all) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "PITMonitor: Delivery Regime Shift (single run)",
        fontsize=13,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.scatter(times, y_all, s=7, alpha=0.35, c="steelblue", label="Actual")
    ax.scatter(times, preds, s=7, alpha=0.35, c="darkorange", label="Predicted mean")
    ax.axvline(true_shift_point, color="red", ls=":", lw=1.5, alpha=0.8)
    if single["alarm_fired"] and single["alarm_time"] is not None:
        ax.axvline(
            int(single["alarm_time"]),
            color="orange",
            ls="--",
            lw=1.5,
            label=f"Alarm (t={single['alarm_time']})",
        )
    ax.set(xlabel="Shipment", ylabel="Delivery time", title="Predictions vs Reality")
    ax.legend(fontsize=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    ax = axes[0, 1]
    point_colors = np.where(times < true_shift_point, "steelblue", "crimson")
    ax.scatter(times, pits, s=5, alpha=0.45, c=point_colors, label="PIT value")
    if len(pits) >= 30:
        rolling = np.convolve(pits, np.ones(30) / 30, mode="valid")
        ax.plot(
            np.arange(30, len(pits) + 1),
            rolling,
            color="black",
            lw=1.3,
            label="Rolling mean (w=30)",
        )
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="Reference (0.5)")
    ax.axvline(
        true_shift_point, color="red", ls=":", lw=1.5, alpha=0.8, label="True shift"
    )
    if single["alarm_fired"] and single["alarm_time"] is not None:
        ax.axvline(
            int(single["alarm_time"]),
            color="orange",
            ls="--",
            lw=1.5,
            label=f"Alarm (t={single['alarm_time']})",
        )
    ax.set(xlabel="Shipment", ylabel="PIT", title="PIT stream")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    ax = axes[1, 0]
    ax.semilogy(
        times, np.maximum(evidence, 1e-10), color="steelblue", lw=1.8, label="E-process"
    )
    threshold = 1.0 / float(single["monitor_alpha"])
    ax.axhline(
        threshold,
        color="crimson",
        ls="--",
        lw=1.8,
        label=f"Threshold (1/α = {threshold:.0f})",
    )
    ax.axvline(
        true_shift_point, color="red", ls=":", lw=1.5, alpha=0.8, label="True shift"
    )
    if single.get("changepoint") is not None:
        ax.axvline(
            int(single["changepoint"]),
            color="green",
            ls="--",
            lw=1.3,
            alpha=0.8,
            label="Changepoint",
        )
    if single["alarm_fired"] and single["alarm_time"] is not None:
        ax.axvline(
            int(single["alarm_time"]),
            color="orange",
            ls="--",
            lw=1.8,
            label=f"Alarm (t={single['alarm_time']})",
        )
    ax.set(xlabel="Shipment", ylabel="Evidence (log)", title="E-process")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    ax = axes[1, 1]
    bins = np.linspace(0, 1, 21)
    ax.hist(
        pits[: true_shift_point - 1],
        bins=bins,
        density=True,
        alpha=0.5,
        color="steelblue",
        edgecolor="white",
        label="Pre-shift",
    )
    ax.hist(
        pits[true_shift_point - 1 :],
        bins=bins,
        density=True,
        alpha=0.5,
        color="crimson",
        edgecolor="white",
        label="Post-shift",
    )
    ax.axhline(1.0, color="black", ls="--", lw=1.3)
    ax.set(xlabel="PIT", ylabel="Density", title="PIT distributions")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_power_panels(artifacts: dict, save_path: Path | None = None) -> None:
    cfg = artifacts["config"]
    shifts = list(cfg["shift_levels"])
    results = artifacts["power_results_by_shift"]

    tprs = [results[s]["tpr"] for s in shifts]
    fprs = [results[s]["false_alarm_rate"] for s in shifts]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "PITMonitor Power Analysis: Delivery Shift", fontsize=13, fontweight="bold"
    )

    ax = axes[0, 0]
    ax.plot(
        shifts,
        tprs,
        "o-",
        color="steelblue",
        lw=2.4,
        markersize=7,
        label="Detection Rate (TPR)",
    )
    ax.set(xlabel="Shift magnitude", ylabel="Detection rate (TPR)", title="Power curve")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="upper right")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    ax = axes[0, 1]
    selected = [s for s in [0.2, 0.4, 0.6, 1.0] if s in results]
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(selected)))
    for shift, color in zip(selected, colors):
        delays = np.asarray(results[shift]["delays"], dtype=float)
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
            label=f"Shift {shift:.0%} (n={len(delays)})",
        )
    ax.set(
        xlabel="Detection delay", ylabel="Cumulative probability", title="Delay ECDF"
    )
    ax.set_ylim(0.0, 1.02)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    ax = axes[1, 0]
    ax.plot(
        shifts,
        fprs,
        "s-",
        color="mediumpurple",
        lw=2.4,
        markersize=6,
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
        shifts, 0, [cfg["alpha_power"]] * len(shifts), color="crimson", alpha=0.06
    )
    ax.set(
        xlabel="Shift magnitude",
        ylabel="False alarm rate",
        title="Ville-bound diagnostic",
    )
    ax.set_ylim(-0.01, min(0.2, cfg["alpha_power"] * 3.0))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    ax = axes[1, 1]
    alarm_selected = [s for s in [0.2, 0.4, 0.6, 1.0] if s in results]
    colors_alarm = plt.cm.plasma(np.linspace(0.2, 0.85, len(alarm_selected)))
    true_change = int(cfg["n_stable_power"]) + 1
    for shift, color in zip(alarm_selected, colors_alarm):
        delays = np.asarray(results[shift]["delays"], dtype=float)
        if delays.size == 0:
            continue
        alarm_times = cfg["n_stable_power"] + delays
        ax.hist(
            alarm_times,
            bins=16,
            density=True,
            alpha=0.35,
            color=color,
            edgecolor="white",
            label=f"Shift {shift:.0%} (n={len(alarm_times)})",
        )
    ax.axvline(true_change, color="black", ls="--", lw=1.5, label="True change point")
    ax.set(xlabel="Alarm time", ylabel="Density", title="Alarm-time distribution")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_comparison_panels(artifacts: dict, save_path: Path | None = None) -> None:
    cfg = artifacts["config"]
    rows = artifacts["comparison_rows"]
    shifts = list(cfg["compare_shift_levels"])
    methods = ["PITMonitor", "DDM", "EDDM", "ADWIN", "KSWIN"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for method in methods:
        ys_fpr, ys_tpr, ys_delay = [], [], []
        for shift in shifts:
            row = next(r for r in rows if r["shift"] == shift and r["method"] == method)
            ys_fpr.append(row["false_alarm_rate"])
            ys_tpr.append(row["tpr"])
            ys_delay.append(row["median_delay"])

        linestyle = "-" if method == "PITMonitor" else "--"
        marker = "s" if method == "PITMonitor" else "o"
        linewidth = 2.5 if method == "PITMonitor" else 2.0
        axes[0].plot(
            shifts, ys_fpr, marker=marker, ls=linestyle, lw=linewidth, label=method
        )
        axes[1].plot(
            shifts, ys_tpr, marker=marker, ls=linestyle, lw=linewidth, label=method
        )
        axes[2].plot(
            shifts, ys_delay, marker=marker, ls=linestyle, lw=linewidth, label=method
        )

    axes[0].axhline(
        cfg["alpha_power"],
        color="crimson",
        ls=":",
        lw=1.5,
        alpha=0.7,
        label=f"Ville bound (α={cfg['alpha_power']:.2f})",
    )
    axes[0].set(
        xlabel="Shift magnitude", ylabel="False alarm rate", title="FPR across methods"
    )
    axes[0].set_ylim(-0.01, 0.15)
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].set(
        xlabel="Shift magnitude",
        ylabel="True positive rate",
        title="TPR across methods",
    )
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(fontsize=8, loc="upper right")

    axes[2].set(
        xlabel="Shift magnitude",
        ylabel="Median detection delay",
        title="Delay across methods",
    )
    axes[2].set_ylim(bottom=0)
    axes[2].grid(True, alpha=0.2)
    axes[2].legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
