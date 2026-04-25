"""Path resolution and safe file loading for the dashboard."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Order for grouping figures on the EDA page (prefix match on filename stem).
FIGURE_STAGE_PREFIXES: tuple[str, ...] = ("stage01", "stage02", "stage03", "stage04", "stage05")


def get_project_root() -> Path:
    """Locate ML project root (contains ``DATASET_ieee-cis-elliptic``). ``processed_data`` may be empty until stage 1."""
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "DATASET_ieee-cis-elliptic").is_dir():
            return p
    cwd = Path.cwd().resolve()
    if (cwd / "DATASET_ieee-cis-elliptic").is_dir():
        return cwd
    # Running from app/ only
    for p in cwd.parents:
        if (p / "DATASET_ieee-cis-elliptic").is_dir():
            return p
    return cwd


def data_bundle_dir_present() -> bool:
    """True when ``DATASET_ieee-cis-elliptic`` exists under the resolved project root."""
    return (get_project_root() / "DATASET_ieee-cis-elliptic").is_dir()


def figure_stage_bucket(path: Path) -> str:
    stem = path.stem.lower()
    for prefix in FIGURE_STAGE_PREFIXES:
        if stem.startswith(prefix):
            return prefix
    return "other"


def figures_grouped_by_stage(root: Path) -> dict[str, list[Path]]:
    """Pipeline figures grouped by stage prefix; newest first within each group."""
    paths = list_images_recursive(root)
    buckets: dict[str, list[Path]] = {p: [] for p in (*FIGURE_STAGE_PREFIXES, "other")}
    for p in paths:
        buckets.setdefault(figure_stage_bucket(p), []).append(p)
    for key in list(buckets.keys()):
        items = buckets[key]
        items.sort(
            key=lambda x: x.stat().st_mtime if x.is_file() else 0.0,
            reverse=True,
        )
    return {k: buckets[k] for k in (*FIGURE_STAGE_PREFIXES, "other")}


def newest_core_artifact_mtime() -> datetime | None:
    """Latest modification time among expected ``processed_data`` artifacts (UTC)."""
    root = processed_dir()
    mtimes: list[float] = []
    for names in ARTIFACTS.values():
        for n in names:
            p = root / n
            if p.is_file():
                try:
                    mtimes.append(p.stat().st_mtime)
                except OSError:
                    continue
    if not mtimes:
        return None
    return datetime.fromtimestamp(max(mtimes), tz=timezone.utc)


def processed_dir() -> Path:
    return get_project_root() / "processed_data"


def figures_dir() -> Path:
    """Pipeline-generated PNGs (EDA, SHAP, stage plots)."""
    return get_project_root() / "figures"


def report_docs_dir() -> Path:
    """Markdown / LaTeX report sources (tracked in Git)."""
    return get_project_root() / "manuscript"


def lit_review_dir() -> Path:
    return get_project_root() / "Lit_Review"


def ref_dir() -> Path:
    return get_project_root() / "Ref"


def main_py_path() -> Path:
    return get_project_root() / "main.py"


def safe_read_csv(path: Path, nrows: int | None = 5000, **kwargs: Any) -> pd.DataFrame | None:
    try:
        if not path.is_file():
            return None
        return pd.read_csv(path, nrows=nrows, low_memory=False, **kwargs)
    except Exception:
        return None


def safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.is_file():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def safe_read_text(path: Path, max_chars: int = 50_000) -> str | None:
    try:
        if not path.is_file():
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[:max_chars] + ("…" if len(text) > max_chars else "")
    except Exception:
        return None


def list_images_recursive(root: Path, extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    try:
        for ext in extensions:
            out.extend(root.rglob(f"*{ext}"))
        return sorted(set(out))
    except Exception:
        return []


def list_pdfs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    try:
        return sorted(root.rglob("*.pdf"))
    except Exception:
        return []


def list_csvs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    try:
        return sorted(root.glob("*.csv"))
    except Exception:
        return []


# Expected artifact names (pipeline outputs)
ARTIFACTS = {
    "stage1": [
        "ieee_train_merged_cleaned.csv",
        "ieee_train_eda_ready.csv",
        "elliptic_transactions_cleaned.csv",
        "preprocessing_config.json",
        "ieee_missing_top20_summary.csv",
    ],
    "stage2": ["gbdt_preds.csv"],
    "stage3": ["hybrid_dnn_anomaly_preds.csv"],
    "stage4": [
        "final_hybrid_comparison_metrics.csv",
        "final_hybrid_scores.csv",
        "final_hybrid_threshold.txt",
    ],
    "stage5": [
        "elliptic_graph_experiments.csv",
    ],
}
