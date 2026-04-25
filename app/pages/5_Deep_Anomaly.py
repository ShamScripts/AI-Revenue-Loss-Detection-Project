import sys
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

import streamlit as st

from components import file_utils as fu
from components.cards import info_panel
from components.styling import inject_css, section_title
from components.sidebar import render_sidebar_stats
from components.tables import preview_df

inject_css()
render_sidebar_stats()

section_title("Deep Learning & Anomaly · Stage 3")
st.markdown(
    "**Branch A:** Attention DNN (TensorFlow) or **MLPClassifier** fallback when TF is unavailable. "
    "**Branch B:** Isolation Forest anomaly scores (normalized). **Hybrid:** weighted blend for downstream fusion."
)

path = fu.processed_dir() / "hybrid_dnn_anomaly_preds.csv"
if not path.is_file():
    st.error("`hybrid_dnn_anomaly_preds.csv` not found. Run **Stage 3** — requires Stage 1–2 outputs.")
else:
    df = fu.safe_read_csv(path, nrows=100_000)
    if df is not None:
        st.success(f"Loaded **{len(df):,}** rows.")
        cols = [c for c in ["dnn_pred_proba", "anomaly_score", "hybrid_score", "isFraud"] if c in df.columns]
        for c in cols:
            if c != "isFraud" and df[c].dtype != object:
                st.caption(c)
                try:
                    import plotly.express as px

                    fig = px.histogram(df[c].dropna(), nbins=50, title=c)
                    fig.update_layout(template="plotly_white", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass
        preview_df(df, max_rows=35)

base_path = fu.processed_dir() / "stage03_ieee_dnn_baselines.csv"
if base_path.is_file():
    st.markdown("**Attention vs plain MLP (validation ROC-AUC)** — `stage03_ieee_dnn_baselines.csv`")
    st.dataframe(fu.safe_read_csv(base_path, nrows=20), use_container_width=True, hide_index=True)

info_panel(
    "Scores",
    "• **dnn_pred_proba** — neural probability of fraud.  \n"
    "• **anomaly_score** — normalized IF score.  \n"
    "• **hybrid_score** — 0.7·DNN + 0.3·anomaly (stage 3 default).",
)
