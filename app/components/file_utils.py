"""Path resolution and safe file loading for the dashboard."""

from __future__ import annotations

import json
import re
import shutil
import ssl
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

# Order for grouping figures on the EDA page (prefix match on filename stem).
FIGURE_STAGE_PREFIXES: tuple[str, ...] = ("stage01", "stage02", "stage03", "stage04", "stage05")

# Google Drive: ZIP with DATASET_ieee-cis-elliptic/, processed_data/, figures/
GDRIVE_ARTIFACT_ZIP_URL = (
    "https://drive.google.com/uc?export=download&id=1KdTWACLTNxpH5VrbSKdbLw1W10O9J29u"
)


def _git_repo_root() -> Path:
    """Clone root (``main.py`` + ``app/``), even when ``DATASET_ieee-cis-elliptic`` is absent."""
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "main.py").is_file() and (p / "app").is_dir():
            return p
    return Path.cwd().resolve()


def _local_bundle_complete(root: Path) -> bool:
    """True when repo-root layout has dataset, processed metrics, and at least one figure PNG."""
    ds = root / "DATASET_ieee-cis-elliptic"
    proc = root / "processed_data"
    fig = root / "figures"
    if not ds.is_dir() or not proc.is_dir() or not fig.is_dir():
        return False
    if not (proc / "final_hybrid_comparison_metrics.csv").is_file():
        return False
    try:
        return any(fig.rglob("*.png"))
    except OSError:
        return False


def _read_bundle_marker(runtime_dir: Path) -> Path | None:
    marker = runtime_dir / ".resolved_bundle_root.txt"
    if not marker.is_file():
        return None
    try:
        p = Path(marker.read_text(encoding="utf-8").strip()).resolve()
    except OSError:
        return None
    return p if p.is_dir() else None


def _find_bundle_under_staging(staging: Path) -> Path | None:
    for c in (staging, staging / "runtime_artifacts"):
        if _local_bundle_complete(c):
            return c
    try:
        for sub in staging.iterdir():
            if sub.is_dir() and _local_bundle_complete(sub):
                return sub
    except OSError:
        pass
    return None


def _download_gdrive_zip(dest_zip: Path) -> None:
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FraudDashboardArtifacts/1.0)"}

    def fetch(u: str) -> bytes:
        req = Request(u, headers=headers)
        with urlopen(req, context=ctx, timeout=600) as r:
            return r.read()

    data = fetch(GDRIVE_ARTIFACT_ZIP_URL)
    if len(data) < 100_000 and (
        data.strip().startswith(b"<")
        or b"virus scan" in data.lower()
        or b"download anyway" in data.lower()
    ):
        for pat in (rb"confirm=([\w-]+)", rb'href="[^"]*confirm=([\w-]+)'):
            m = re.search(pat, data)
            if m:
                token = m.group(1).decode("ascii", errors="ignore")
                joiner = "&" if "?" in GDRIVE_ARTIFACT_ZIP_URL else "?"
                data = fetch(f"{GDRIVE_ARTIFACT_ZIP_URL}{joiner}confirm={token}")
                break
    dest_zip.write_bytes(data)


@st.cache_resource(show_spinner="Fetching dashboard artifacts…")
def _cached_resolve_data_root(repo_root_str: str) -> str:
    repo = Path(repo_root_str)
    if _local_bundle_complete(repo):
        return str(repo.resolve())
    rt = repo / "runtime_artifacts"
    rt.mkdir(parents=True, exist_ok=True)
    cached = _read_bundle_marker(rt)
    if cached is not None and _local_bundle_complete(cached):
        return str(cached.resolve())
    zip_path = rt / "runtime_artifacts.zip"
    _download_gdrive_zip(zip_path)
    staging = rt / "_extract"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(staging)
    base = _find_bundle_under_staging(staging)
    if base is None or not _local_bundle_complete(base):
        raise RuntimeError(
            "Downloaded artifact ZIP is missing DATASET_ieee-cis-elliptic/, "
            "processed_data/, or figures/ with expected contents."
        )
    (rt / ".resolved_bundle_root.txt").write_text(str(base.resolve()), encoding="utf-8")
    return str(base.resolve())


def resolved_data_root() -> Path:
    """Root used for ``processed_data/``, ``figures/``, and the dataset bundle (repo or extracted ZIP)."""
    return Path(_cached_resolve_data_root(str(_git_repo_root().resolve())))


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
    """True when ``DATASET_ieee-cis-elliptic`` exists under the resolved data root (repo or runtime ZIP)."""
    return dataset_bundle_dir().is_dir()


def dataset_bundle_dir() -> Path:
    """Directory ``DATASET_ieee-cis-elliptic`` (under :func:`resolved_data_root`)."""
    return resolved_data_root() / "DATASET_ieee-cis-elliptic"


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
    return resolved_data_root() / "processed_data"


def figures_dir() -> Path:
    """Pipeline-generated PNGs (EDA, SHAP, stage plots)."""
    return resolved_data_root() / "figures"


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
