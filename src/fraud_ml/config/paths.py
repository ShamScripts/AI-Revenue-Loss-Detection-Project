from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def find_project_root() -> Path:
    """Resolve repo root (folder containing ``DATASET_ieee-cis-elliptic``)."""
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "DATASET_ieee-cis-elliptic").is_dir():
            return p
    cwd = Path.cwd().resolve()
    if (cwd / "DATASET_ieee-cis-elliptic").is_dir():
        return cwd
    return cwd


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_root: Path
    ieee_dir: Path
    elliptic_dir: Path
    processed_dir: Path
    figures: Path  # pipeline PNG outputs (EDA, stages 1–5)

    @classmethod
    def from_root(cls, root: Path | None = None) -> ProjectPaths:
        r = (root or find_project_root()).resolve()
        data = r / "DATASET_ieee-cis-elliptic"
        ieee = data / "ieee-fraud-detection"
        elliptic = data / "elliptic-dataset" / "elliptic_bitcoin_dataset"
        processed = r / "processed_data"
        figs = r / "figures"
        return cls(
            root=r,
            data_root=data,
            ieee_dir=ieee,
            elliptic_dir=elliptic,
            processed_dir=processed,
            figures=figs,
        )


def get_paths(root: Path | None = None) -> ProjectPaths:
    return ProjectPaths.from_root(root)
