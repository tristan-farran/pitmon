
- Make training the model it's own independent file which saves the weights to avoid redoing it every time, so the model can be trained once and then tested

- Add full docstrings

- Ensure the README and the paper are aligned with the code

- Edit config and other files in experiment to take several n_bins values and run the experiment for each and create a plot showing the effectiveness at each, figure out the theory

- Improve crowded and unreadable plots featuring boxes that don't render properly, NaNs, and very tightly clustered results that appear as a line



Plotting
#############


def plot_pit_histogram(pits, title="PIT Histogram"):
    """Plots PIT density vs. theoretical uniform distribution[cite: 273]."""
    plt.figure(figsize=(8, 5))
    plt.hist(pits, bins=20, density=True, alpha=0.7, color="blue", edgecolor="black")
    plt.axhline(1.0, color="red", linestyle="--", label="Perfect Calibration")
    plt.xlabel("Probability Integral Transform (PIT)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_calibration_curve(pits):
    """Generates a Reliability Diagram comparing empirical CDF to Ideal[cite: 285]."""
    sorted_pits = np.sort(pits)
    empirical_cdf = np.arange(1, len(sorted_pits) + 1) / len(sorted_pits)

    plt.figure(figsize=(8, 5))
    plt.plot(sorted_pits, empirical_cdf, label="Empirical CDF")
    plt.plot([0, 1], [0, 1], "k--", label="Ideal")
    plt.xlabel("PIT Values")
    plt.ylabel("Cumulative Probability")
    plt.title("Reliability Diagram (PIT-based)")
    plt.legend()
    plt.show()


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
