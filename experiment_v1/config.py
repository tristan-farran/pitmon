from dataclasses import asdict, dataclass, field
from __future__ import annotations
from pathlib import Path


@dataclass
class DeliveryDemoConfig:
    seed: int = 0
    output_dir: Path = Path(__file__).parent / "out"
    artifact_name: str = "delivery_demo_artifacts.pkl"

    n_train: int = 2_000
    n_cal: int = 500
    n_y_bins: int = 50

    alpha: float = 0.05
    n_monitor_bins: int = 10
    n_stable: int = 300
    n_shifted: int = 200

    alpha_power: float = 0.05
    n_stable_power: int = 200
    n_shifted_power: int = 200
    n_trials: int = 1_000
    shift_levels: list[float] = field(
        default_factory=lambda: [
            0.0,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            1.0,
        ]
    )

    compare_shift_levels: list[float] = field(
        default_factory=lambda: [0.0, 0.30, 0.60, 1.0]
    )
    n_trials_compare: int = 1_000
    max_workers: int = 8

    def normalized(self) -> "DeliveryDemoConfig":
        cfg = DeliveryDemoConfig(**asdict(self))
        cfg.output_dir = Path(cfg.output_dir)
        cfg.max_workers = max(1, int(cfg.max_workers))
        cfg.n_trials = max(1, int(cfg.n_trials))
        cfg.n_trials_compare = max(1, int(cfg.n_trials_compare))
        cfg.n_y_bins = max(3, int(cfg.n_y_bins))
        cfg.n_monitor_bins = max(2, int(cfg.n_monitor_bins))
        return cfg

    def artifact_path(self) -> Path:
        return self.output_dir / self.artifact_name

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload
