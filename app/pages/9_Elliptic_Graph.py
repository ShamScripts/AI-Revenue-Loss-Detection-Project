"""Elliptic graph experiments (Stage 5) — GCN + tabular baselines."""

import sys
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

import pandas as pd
import streamlit as st

from components import file_utils as fu
from components.cards import info_panel
from components.styling import inject_css, section_title
from components.sidebar import render_sidebar_stats
from components.tables import preview_df

inject_css()
render_sidebar_stats()

section_title("Elliptic graph · Stage 5")
st.markdown(
    "Two-layer **GCN** on the Elliptic transaction graph (requires **PyTorch**), plus **FraudGT-style** deep MLP and "
    "**LR / RF** on tabular features — **time-ordered** train/test split. Output: `elliptic_graph_experiments.csv`."
)

csv_path = fu.processed_dir() / "elliptic_graph_experiments.csv"
cfg_path = fu.processed_dir() / "stage05_experiment_config.json"
hist_path = fu.figures_dir() / "stage05_elliptic_gcn_score_hist.png"

if not csv_path.is_file():
    st.warning(
        "`elliptic_graph_experiments.csv` not found. Run **Stage 5** (`python main.py --stage 5`) after Stage 1. "
        "Install **torch** for GCN metrics: `pip install torch`."
    )
else:
    df = fu.safe_read_csv(csv_path, nrows=50)
    if df is not None:
        st.success(f"Loaded **{len(df)}** model rows.")
        gcn = df[df["model"].astype(str).str.contains("GCN", case=False, na=False)]
        if not gcn.empty:
            split_s = gcn["split"].astype(str).iloc[0] if "split" in gcn.columns else ""
            roc_s = gcn["roc_auc"].iloc[0] if "roc_auc" in gcn.columns else None
            if "skipped" in split_s.lower() or pd.isna(roc_s):
                st.info(
                    "**GCN row skipped or empty** — install PyTorch (`pip install torch`) from the project "
                    "venv, then re-run **Stage 5** to populate GCN metrics and `stage05_elliptic_gcn_score_hist.png`.",
                    icon="ℹ️",
                )
        preview_df(df, "Results", max_rows=20)

if cfg_path.is_file():
    st.caption("Config")
    st.code(fu.safe_read_text(cfg_path, max_chars=4000) or "", language="json")

if hist_path.is_file():
    st.markdown("**GCN score distribution (test)**")
    st.image(str(hist_path), use_container_width=True)

info_panel(
    "Report alignment",
    "TABLE II (Elliptic) in **Fusion & Final Results** can pull from this CSV when present. "
    "If PyTorch is missing, the GCN row is empty until you install torch and re-run Stage 5.",
)
