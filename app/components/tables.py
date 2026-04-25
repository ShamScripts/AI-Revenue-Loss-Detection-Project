"""DataFrame preview helpers."""

from __future__ import annotations

import streamlit as st
import pandas as pd


def preview_df(
    df: pd.DataFrame | None,
    title: str = "Preview",
    max_rows: int = 50,
    use_container_width: bool = True,
) -> None:
    if df is None or df.empty:
        st.warning("No tabular data to display.")
        return
    st.caption(f"Showing up to {max_rows} rows · {df.shape[1]} columns")
    st.dataframe(df.head(max_rows), use_container_width=use_container_width, hide_index=True)


def metrics_from_df_row(row: pd.Series, keys: list[str]) -> None:
    cols = st.columns(len(keys))
    for c, k in zip(cols, keys):
        if k in row.index:
            c.metric(k.replace("_", " ").title(), f"{row[k]:.4f}" if isinstance(row[k], float) else row[k])
