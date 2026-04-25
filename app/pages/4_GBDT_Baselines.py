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

section_title("GBDT & Baselines · Stage 2")
st.markdown(
    "**Baselines:** Logistic Regression, Decision Tree, Random Forest (validation metrics printed to console during training). "
    "**Primary model:** LightGBM with XGBoost fallback — probabilities exported for fusion. "
    "Optional **SMOTE**, **SHAP** summary plot, and **GBDT tuning** are available via CLI (see **Run Pipeline**)."
)

pred_path = fu.processed_dir() / "gbdt_preds.csv"
if not pred_path.is_file():
    st.error("`gbdt_preds.csv` not found. Run **Stage 2** (`python main.py --stage 2`).")
else:
    df = fu.safe_read_csv(pred_path, nrows=100_000)
    if df is not None:
        st.success(f"Loaded **{len(df):,}** rows from `gbdt_preds.csv`.")
        if "gbdt_pred_proba" in df.columns:
            s = df["gbdt_pred_proba"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean prob", f"{s.mean():.4f}")
            c2.metric("Std", f"{s.std():.4f}")
            c3.metric("Min / Max", f"{s.min():.3f} / {s.max():.3f}")
            if "is_valid_split_row" in df.columns:
                c4.metric("Valid rows", f"{int(df['is_valid_split_row'].sum()):,}")
            try:
                import plotly.express as px

                fig = px.histogram(s, nbins=60, title="GBDT predicted probability distribution")
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.bar_chart(s.value_counts(bins=20))
        preview_df(df, "Preview", max_rows=40)

shap_path = fu.figures_dir() / "stage02_shap_summary.png"
if shap_path.is_file():
    st.markdown("**SHAP (GBDT)** — `figures/stage02_shap_summary.png`")
    st.image(str(shap_path), use_container_width=True)

info_panel(
    "Note",
    "Baseline comparison tables are printed to the console during Stage 2. "
    "This page shows **`gbdt_preds.csv`** and SHAP when Stage 2 was run with SHAP enabled. "
    "Experiment flags: `stage02_experiment_config.json`.",
)
