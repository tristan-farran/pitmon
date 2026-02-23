from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class CifarDemoConfig:
    seed: int = 0
    data_dir: Path = Path(__file__).parent / "data"
    output_dir: Path = Path(__file__).parent / "out"

    corruption: str = "gaussian_noise"
    severity_demo: int = 3
    severity_levels: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])

    train_size: int = 15_000
    n_stable: int = 300
    n_shifted: int = 300

    alpha: float = 0.05
    alpha_power: float = 0.05
    n_bins: int = 10

    n_stable_power: int = 300
    n_shifted_power: int = 300
    n_trials: int = 1_000

    max_workers: int = 8

    artifact_name: str = "cifar_demo_artifacts.pkl"

    def normalized(self) -> "CifarDemoConfig":
        cfg = CifarDemoConfig(**asdict(self))
        cfg.data_dir = Path(cfg.data_dir)
        cfg.output_dir = Path(cfg.output_dir)
        cfg.max_workers = max(1, cfg.max_workers)
        return cfg

    def artifact_path(self) -> Path:
        return self.output_dir / self.artifact_name

    def to_dict(self) -> dict:
        data = asdict(self)
        data["data_dir"] = str(self.data_dir)
        data["output_dir"] = str(self.output_dir)
        return data
