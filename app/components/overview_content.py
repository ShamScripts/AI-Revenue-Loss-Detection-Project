"""Home dashboard: fintech shell + horizontal wizard + research steps."""

from __future__ import annotations

import streamlit as st

from components import dashboard_shell as ds
from components import research_presentation as rp


def render_dashboard_home() -> None:
    ds.inject_fintech_light_theme()
    ds.hide_default_sidebar_nav()
    st.markdown(
        "<style>section[data-testid='stSidebar']{display:none!important;}</style>",
        unsafe_allow_html=True,
    )
    ds.render_horizontal_stepper()
    rp.render_wizard_step(int(st.session_state.get("wiz", 1)))

    with st.expander("Technical appendix — artifact paths & pipeline CLI"):
        from components import file_utils as fu

        st.markdown(
            f"""
| Path | Purpose |
|------|---------|
| `{fu.get_project_root()}` | Project root |
| `{fu.processed_dir()}` | CSV + JSON artifacts |
| `{fu.figures_dir()}` | Saved figures (`figures/`) |

Run **`python main.py`** from the project root (Python **3.10–3.12** recommended).
"""
        )


def render_overview() -> None:
    """Entry used by `app.py` and `pages/1_Overview.py`."""
    render_dashboard_home()
