"""
Fraud Detection Dashboard — entry point.

Run from project root:
  streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP = Path(__file__).resolve().parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

import streamlit as st

from components.overview_content import render_dashboard_home
from components.styling import inject_css

st.set_page_config(
    page_title="Revenue Leakage Detection · Hybrid ML",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_css()
render_dashboard_home()
