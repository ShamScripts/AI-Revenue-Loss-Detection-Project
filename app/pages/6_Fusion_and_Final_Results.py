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

section_title("Fusion & Final Results · Stage 4")
st.markdown(
    "Scores are **min–max normalized** per component, then fused with weights "
    "**0.45 GBDT + 0.40 DNN + 0.15 anomaly**. The hybrid threshold is tuned on **validation F1**; "
    "component rows in the comparison table use **0.5** on normalized scores unless noted."
)

proc = fu.processed_dir()
m_path = proc / "final_hybrid_comparison_metrics.csv"
s_path = proc / "final_hybrid_scores.csv"
t_path = proc / "final_hybrid_threshold.txt"

with st.expander("Run manifest (weights & experiment flags)"):
    st.markdown(
        "**Fusion weights (code defaults):** `0.45` GBDT · `0.40` DNN · `0.15` anomaly — see `stage04_fusion.py`."
    )
    for name, p in (
        ("Stage 4", proc / "stage04_experiment_config.json"),
        ("Stage 3", proc / "stage03_experiment_config.json"),
        ("Stage 2", proc / "stage02_experiment_config.json"),
    ):
        raw = fu.safe_read_text(p, max_chars=8000)
        if raw:
            st.markdown(f"**{name}** — `{p.name}`")
            st.code(raw, language="json")
        else:
            st.caption(f"{name}: `{p.name}` not found")

if t_path.is_file():
    thr = fu.safe_read_text(t_path, max_chars=200)
    st.metric("Selected threshold (validation F1)", thr.strip() if thr else "—")

_PALETTE = (
    "#2563eb",
    "#7c3aed",
    "#db2777",
    "#059669",
    "#d97706",
    "#64748b",
)


def _download_csv_button(path: Path, *, key: str) -> None:
    if not path.is_file():
        return
    try:
        data = path.read_bytes()
    except OSError:
        return
    st.download_button(
        label=f"Download `{path.name}`",
        data=data,
        file_name=path.name,
        mime="text/csv",
        key=key,
    )


metrics = fu.safe_read_csv(m_path, nrows=500)
if metrics is not None and not metrics.empty:
    section_title("Model comparison (test)")
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    _download_csv_button(m_path, key="dl_metrics")
    try:
        import plotly.express as px

        if "model" in metrics.columns and "roc_auc" in metrics.columns:
            fig = px.bar(
                metrics,
                x="model",
                y="roc_auc",
                title="ROC-AUC by model (threshold-free)",
                color="model",
                color_discrete_sequence=_PALETTE,
            )
            fig.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        if "model" in metrics.columns and "pr_auc" in metrics.columns:
            fig_pr = px.bar(
                metrics,
                x="model",
                y="pr_auc",
                title="PR-AUC by model (informative under imbalance)",
                color="model",
                color_discrete_sequence=_PALETTE,
            )
            fig_pr.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_pr, use_container_width=True)

        if "f1" in metrics.columns:
            fig2 = px.bar(
                metrics,
                x="model",
                y="f1",
                title="F1 at stated threshold (hybrid uses tuned τ)",
                color="model",
                color_discrete_sequence=_PALETTE,
            )
            fig2.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
    except Exception:
        pass
else:
    st.warning("`final_hybrid_comparison_metrics.csv` not found — run Stage 4.")

scores = fu.safe_read_csv(s_path, nrows=50_000)
if scores is not None and not scores.empty:
    section_title("Final scores (preview)")
    n_full = len(scores)
    st.caption(f"Loaded **{n_full:,}** rows (capped at 50k for dashboard). Showing head preview.")
    preview_df(scores, max_rows=25)
    if "hybrid_weighted_score" in scores.columns and "target" in scores.columns:
        try:
            import plotly.express as px

            pos = scores[scores["target"] == 1]
            neg = scores[scores["target"] == 0]
            parts: list[pd.DataFrame] = []
            if len(pos) > 0:
                parts.append(pos.sample(n=min(1200, len(pos)), random_state=42))
            if len(neg) > 0:
                parts.append(neg.sample(n=min(4000, len(neg)), random_state=42))
            if parts:
                sample = pd.concat(parts, axis=0)
                fig = px.scatter(
                    sample,
                    x="hybrid_weighted_score",
                    y=sample["target"].astype(str),
                    color="target",
                    title="Hybrid score vs label (stratified sample: fraud + random negatives)",
                    opacity=0.35,
                    color_discrete_sequence=("#dc2626", "#64748b"),
                )
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass
else:
    st.info("No `final_hybrid_scores.csv` yet.")

section_title("Report tables (for thesis / paper)")
st.caption(
    "Generated when you run **Stage 4** — same **test split** as fusion for IEEE rows; "
    "Elliptic baselines use a separate 80/20 split on licit/illicit when Stage 5 is absent. "
    "Empty cells are placeholders for skipped models (e.g. GCN without PyTorch)."
)

t1 = proc / "report_table_1_ieee_cis.csv"
t2 = proc / "report_table_2_elliptic.csv"
t3 = proc / "report_table_3_ablation.csv"

for label, path in (
    ("TABLE I — Performance comparison (IEEE-CIS)", t1),
    ("TABLE II — Performance comparison (Elliptic)", t2),
    ("TABLE III — Ablation study results", t3),
):
    df_r = fu.safe_read_csv(path, nrows=50)
    if df_r is not None and not df_r.empty:
        st.markdown(f"**{label}**")
        st.dataframe(df_r, use_container_width=True, hide_index=True)
        _download_csv_button(path, key=f"dl_{path.name}")
        st.caption(f"`{path.name}`")
    else:
        st.warning(f"`{path.name}` not found — run Stage 4 after Stages 2–3.")

info_panel(
    "Exports",
    "Use **Download** buttons for quick copies of report tables and metrics. "
    "Full `final_hybrid_scores.csv` may be large — pull from `processed_data/` for the complete join.",
)
