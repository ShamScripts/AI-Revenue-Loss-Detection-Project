import sys
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

import streamlit as st

from components import file_utils as fu
from components.cards import info_panel, kpi_row, status_badge
from components.sidebar import render_sidebar_stats
from components.styling import inject_css, section_title
from components.tables import preview_df

inject_css()
render_sidebar_stats()

section_title("Dataset & Preprocessing · Stage 1")
st.markdown(
    "This stage loads **IEEE-CIS** (transactions + identity) and **Elliptic** (features, classes, edges), "
    "drops ultra-sparse columns, imputes values, engineers time/amount and graph features, and writes cleaned CSVs."
)

root = fu.get_project_root()
proc = fu.processed_dir()
miss_path = proc / "ieee_missing_top20_summary.csv"
cfg_path = proc / "preprocessing_config.json"
ieee_ready = proc / "ieee_train_eda_ready.csv"
elliptic_out = proc / "elliptic_transactions_cleaned.csv"

c1, c2, c3 = st.columns(3)
c1.markdown(f"**IEEE (EDA-ready)**  \n{status_badge(ieee_ready.is_file())}", unsafe_allow_html=True)
c2.markdown(f"**Elliptic cleaned**  \n{status_badge(elliptic_out.is_file())}", unsafe_allow_html=True)
c3.markdown(f"**preprocessing_config.json**  \n{status_badge(cfg_path.is_file())}", unsafe_allow_html=True)

cfg = fu.safe_read_json(cfg_path)
if cfg:
    kpi_row(
        [
            ("IEEE shape (saved)", str(cfg.get("ieee", {}).get("final_shape", "—")), "rows × cols"),
            ("Elliptic shape", str(cfg.get("elliptic", {}).get("final_shape", "—")), "rows × cols"),
            ("Dropped cols (IEEE)", str(cfg.get("ieee", {}).get("n_dropped_columns", "—")), "high-missing"),
        ]
    )
    with st.expander("preprocessing_config.json (summary)"):
        st.json(cfg)

miss = fu.safe_read_csv(miss_path, nrows=50)
if miss is not None:
    section_title("Missing-value summary (top 20)")
    preview_df(miss, "Top missing columns", max_rows=25)
    try:
        import plotly.express as px

        fig = px.bar(miss.head(15), x="pct_missing", y="column", orientation="h", title="Missing % (IEEE, pre-clean)")
        fig.update_layout(template="plotly_white", height=420)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass
else:
    st.warning("`ieee_missing_top20_summary.csv` not found — run **Stage 1** first.")

df = fu.safe_read_csv(ieee_ready, nrows=2000)
if df is not None and "isFraud" in df.columns:
    section_title("Class distribution (IEEE sample)")
    vc = df["isFraud"].value_counts()
    c1, c2 = st.columns(2)
    c1.metric("Rows (sample)", f"{len(df):,}")
    c2.metric("Fraud rate", f"{df['isFraud'].mean()*100:.2f}%")
    try:
        import plotly.express as px

        fig = px.pie(values=vc.values, names=vc.index.astype(str), title="isFraud balance (sample)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.bar_chart(vc)
else:
    st.info("Load `ieee_train_eda_ready.csv` after Stage 1 to see class distribution.")

if df is not None:
    section_title("Preview · IEEE EDA-ready")
    preview_df(df, max_rows=30)

info_panel(
    "Raw data locations",
    f"`{root / 'DATASET_ieee-cis-elliptic'}` — ieee-fraud-detection · elliptic-dataset/elliptic_bitcoin_dataset",
)
