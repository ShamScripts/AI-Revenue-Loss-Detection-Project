"""Fintech shell: light theme, chevron stepper (image-style), optional sidebar hide."""

from __future__ import annotations

import streamlit as st

try:
    from streamlit.errors import StreamlitInvalidColumnGapError
except ImportError:  # pragma: no cover — very old Streamlit
    class StreamlitInvalidColumnGapError(Exception):
        """Placeholder if gap validation error type is missing."""

        pass

STEPPER_CONTAINER_KEY = "wiz_stepper_wedge"

WIZARD_STEPS: list[tuple[int, str]] = [
    (1, "Overview"),
    (2, "Pipeline"),
    (3, "Dataset & EDA"),
    (4, "Modeling"),
    (5, "Results"),
    (6, "Business impact"),
    (7, "Reports"),
]


def _sync_wiz_from_query_params() -> None:
    """Clickable chevrons use ?wiz=N — mirror into session_state."""
    raw: object | None = None
    if hasattr(st, "query_params"):
        qp = st.query_params
        if "wiz" in qp:
            raw = qp.get("wiz")
    else:
        try:
            legacy = st.experimental_get_query_params()
            if "wiz" in legacy and legacy["wiz"]:
                raw = legacy["wiz"][0]
        except Exception:
            return
    if raw is None:
        return
    s = raw[0] if isinstance(raw, list) else str(raw)
    try:
        w = int(s)
    except ValueError:
        return
    if 1 <= w <= len(WIZARD_STEPS):
        st.session_state.wiz = w


def _wiz_stepper_row_selector(*, scoped: bool) -> str:
    """DOM path to the stepper's stHorizontalBlock (keyed subtree; :has survives inner st-key wrapper)."""
    if scoped:
        # st-key class may sit on an inner node; horizontal row stays under same vertical block.
        return (
            f'[data-testid="stAppViewContainer"] [data-testid="stVerticalBlock"]:has([class*="{STEPPER_CONTAINER_KEY}"]) '
            f'[data-testid="stHorizontalBlock"]'
        )
    return '[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"]:first-of-type'


def _wiz_stepper_dynamic_css(wiz: int, *, scoped: bool) -> str:
    """Chevron-shaped Streamlit buttons: colors + clip-path + z-index (overrides BaseWeb defaults)."""
    hb = _wiz_stepper_row_selector(scoped=scoped)
    n = len(WIZARD_STEPS)
    rules: list[str] = [
        f"{hb} {{ gap: 0 !important; column-gap: 0 !important; }}",
        f"{hb} > div ~ div {{ margin-left: -10px !important; }}",
    ]
    for idx, (num, _) in enumerate(WIZARD_STEPS, start=1):
        if num < wiz:
            bg, fg = "#6d8b76", "#ffffff"
        elif num == wiz:
            bg, fg = "#d97706", "#ffffff"
        else:
            bg, fg = "#94a3b8", "#f8fafc"
        if idx == 1:
            clip = "polygon(0 0, calc(100% - 12px) 0, 100% 50%, calc(100% - 12px) 100%, 0 100%)"
            mx = "margin-left: 0 !important;"
            rad = "border-radius: 10px 0 0 10px !important;"
        elif idx == n:
            clip = "polygon(12px 0, 100% 0, 100% 100%, 12px 100%, 0 50%)"
            mx = "margin-left: -12px !important;"
            rad = "border-radius: 0 10px 10px 0 !important;"
        else:
            clip = "polygon(0 0, calc(100% - 12px) 0, 100% 50%, calc(100% - 12px) 100%, 0 100%, 12px 50%)"
            mx = "margin-left: -12px !important;"
            rad = ""
        col = f"{hb} > div:nth-child({idx})"
        rules.append(
            f"{col} {{ min-width: 0 !important; overflow: visible !important; z-index: {idx} !important; position: relative !important; }}"
            f"{col} .stButton {{ width: 100% !important; overflow: visible !important; }}"
            f"{col} .stButton button {{"
            f"width: 100% !important; min-height: 3.2rem !important;"
            f"padding: 10px 4px 10px 16px !important; white-space: pre-line !important; text-align: center !important;"
            f"line-height: 1.28 !important; border: none !important; box-shadow: none !important;"
            f"font-family: \"DM Sans\", \"Segoe UI\", system-ui, sans-serif !important;"
            f"font-size: clamp(0.62rem, 1.05vw, 0.78rem) !important; font-weight: 700 !important;"
            f"text-shadow: 0 1px 1px rgba(0,0,0,0.12) !important; transition: filter 0.15s ease !important;"
            f"background-color: {bg} !important; background-image: none !important; color: {fg} !important;"
            f"border-color: transparent !important; clip-path: {clip}; {mx} {rad}"
            f"}}"
        )
    rules.append(
        f"{hb} .stButton button:hover {{ filter: brightness(1.08) !important; }}"
        f"{hb} .stButton button:focus-visible {{ outline: 2px solid #fcd34d !important; outline-offset: 1px; }}"
    )
    return "<style>" + "".join(rules) + "</style>"


def inject_fintech_light_theme() -> None:
    """Light fintech shell + chevron stepper + presentation polish (UI only)."""
    st.markdown(
        """
<style>
[data-testid="stAppViewContainer"] > .main {
  background: linear-gradient(180deg, #faf8f5 0%, #f4faf4 45%, #fffbf5 100%) !important;
}
.block-container {
  max-width: min(1180px, 96vw) !important;
  padding-top: 0.75rem !important;
  padding-bottom: 3rem !important;
}
[data-testid="stHeader"] {
  background: rgba(255,255,255,0.97) !important;
  border-bottom: 1px solid rgba(15,23,42,0.06);
}

/* --- Chevron stepper: keyed vertical block (:has) or first horizontal row --- */
[data-testid="stAppViewContainer"] [data-testid="stVerticalBlock"]:has([class*="wiz_stepper_wedge"]) [data-testid="stHorizontalBlock"],
[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"]:first-of-type {
  margin: 0.25rem 0 0.85rem !important;
  filter: drop-shadow(0 2px 8px rgba(15,23,42,0.08));
  overflow: visible !important;
}
[data-testid="stAppViewContainer"] [data-testid="stVerticalBlock"]:has([class*="wiz_stepper_wedge"]) [data-testid="stHorizontalBlock"] > div,
[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"]:first-of-type > div {
  min-width: 0 !important;
  overflow: visible !important;
}
[data-testid="stAppViewContainer"] [data-testid="stVerticalBlock"]:has([class*="wiz_stepper_wedge"]) [data-testid="stHorizontalBlock"] .stButton,
[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"]:first-of-type .stButton {
  width: 100%;
  overflow: visible !important;
}

/* Premium KPI tiles */
.premium-kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.85rem;
  margin: 0.75rem 0 1.25rem;
}
@media (max-width: 900px) {
  .premium-kpi-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 520px) {
  .premium-kpi-grid { grid-template-columns: 1fr; }
}
.premium-kpi-tile {
  background: linear-gradient(165deg, #ffffff 0%, #f8fafc 100%);
  border: 1px solid rgba(15,23,42,0.07);
  border-radius: 14px;
  padding: 1rem 1.1rem 1rem 1rem;
  box-shadow: 0 2px 14px rgba(15,23,42,0.06);
  position: relative;
  overflow: hidden;
}
.premium-kpi-tile::before {
  content: "";
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 4px;
  border-radius: 4px 0 0 4px;
  background: linear-gradient(180deg, #166534, #d97706);
}
.premium-kpi-ico {
  font-size: 1.25rem;
  margin-bottom: 0.35rem;
  line-height: 1;
}
.premium-kpi-label {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: #64748b;
  margin-bottom: 0.35rem;
}
.premium-kpi-value {
  font-size: clamp(1.25rem, 2.5vw, 1.65rem);
  font-weight: 800;
  letter-spacing: -0.03em;
  color: #0f172a;
  line-height: 1.1;
}
.premium-kpi-hint {
  font-size: 0.78rem;
  color: #64748b;
  margin-top: 0.45rem;
  line-height: 1.35;
}

/* Smart insights board — amber + sage (readable, no cool blue wash) */
.insight-board {
  background: linear-gradient(135deg, #fffbeb 0%, #f0fdf4 52%, #ecfdf5 100%);
  border: 1px solid rgba(217, 119, 6, 0.22);
  border-radius: 16px;
  padding: 1.15rem 1.35rem 1.05rem;
  margin: 0.75rem 0 1.25rem;
  box-shadow: 0 4px 18px rgba(22, 101, 52, 0.08);
}
.insight-board h3 {
  margin: 0 0 0.65rem 0;
  font-size: 1rem;
  color: #92400e;
  font-weight: 700;
  letter-spacing: -0.02em;
}
.insight-board ul {
  margin: 0;
  padding-left: 1.2rem;
  color: #334155;
  font-size: 0.93rem;
  line-height: 1.65;
}

/* Table section ribbon */
.table-ribbon {
  display: inline-block;
  font-size: 0.7rem;
  font-weight: 800;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #fff;
  padding: 0.35rem 0.85rem;
  border-radius: 8px;
  margin-bottom: 0.45rem;
  box-shadow: 0 2px 8px rgba(15,23,42,0.12);
}
.tr-ieee { background: linear-gradient(90deg, #0d9488, #14b8a6); }
.tr-elliptic { background: linear-gradient(90deg, #2563eb, #6366f1); }
.tr-compare { background: linear-gradient(90deg, #7c3aed, #a855f7); }
.tr-ablate { background: linear-gradient(90deg, #ea580c, #f97316); }
.tr-fig { background: linear-gradient(90deg, #0f172a, #334155); }
.tr-artifacts { background: linear-gradient(90deg, #334155, #64748b); }
.tr-sec-ieee { background: linear-gradient(90deg, #0f766e, #2dd4bf); }
.tr-sec-elliptic { background: linear-gradient(90deg, #1e40af, #818cf8); }
.tr-sec-gbdt { background: linear-gradient(90deg, #166534, #4ade80); }
.tr-sec-deep { background: linear-gradient(90deg, #6b21a8, #c084fc); }
.tr-sec-fusion { background: linear-gradient(90deg, #0e7490, #22d3ee); }
.tr-sec-graph { background: linear-gradient(90deg, #9a3412, #fb923c); }
.tr-sec-other { background: linear-gradient(90deg, #475569, #94a3b8); }

/* Figure panel */
.figure-panel {
  background: #fff;
  border: 1px solid rgba(15,23,42,0.08);
  border-radius: 16px;
  padding: 1rem 1.15rem 1.15rem;
  margin: 0.85rem 0;
  box-shadow: 0 4px 18px rgba(15,23,42,0.05);
}
.figure-panel h5 {
  margin: 0 0 0.25rem 0 !important;
  font-size: 1rem !important;
}
.figure-panel .exp {
  color: #475569;
  font-size: 0.88rem;
  line-height: 1.5;
  margin-bottom: 0.65rem;
}

.story-card {
  background: #ffffff;
  border: 1px solid rgba(15,23,42,0.07);
  border-radius: 14px;
  padding: 1.15rem 1.35rem;
  margin: 0.65rem 0 1rem;
  box-shadow: 0 2px 12px rgba(15,23,42,0.04);
}
.story-card h4 {
  margin: 0 0 0.5rem 0;
  font-size: 0.95rem;
  color: #0f172a;
}
.story-rail {
  width: 4px;
  border-radius: 4px;
  min-height: 100%;
  margin-right: 0.75rem;
  flex-shrink: 0;
}
.story-flex {
  display: flex;
  align-items: flex-start;
  gap: 0;
}
.rail-data { background: linear-gradient(180deg,#2563eb,#60a5fa); }
.rail-pre { background: linear-gradient(180deg,#ea580c,#fb923c); }
.rail-feat { background: linear-gradient(180deg,#16a34a,#4ade80); }
.rail-model { background: linear-gradient(180deg,#7c3aed,#a78bfa); }
.rail-result { background: linear-gradient(180deg,#0d9488,#2dd4bf); }
.rail-risk { background: linear-gradient(180deg,#ca8a04,#facc15); }
.rail-out { background: linear-gradient(180deg,#dc2626,#f87171); }
.table-wrap {
  background: #fff;
  border-radius: 12px;
  border: 1px solid rgba(15,23,42,0.08);
  padding: 0.5rem 0.25rem;
  box-shadow: 0 1px 8px rgba(15,23,42,0.04);
  margin: 0.35rem 0 0.85rem;
}
</style>
""",
        unsafe_allow_html=True,
    )


def hide_default_sidebar_nav() -> None:
    st.markdown(
        """
<style>
nav[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="stSidebar"] > div:first-child { padding-top: 0.5rem; }
</style>
""",
        unsafe_allow_html=True,
    )


def render_horizontal_stepper() -> int:
    """Chevron-shaped buttons: same-tab navigation (no <a href> — avoids new-tab behavior)."""
    _sync_wiz_from_query_params()
    if "wiz" not in st.session_state:
        st.session_state.wiz = 1
    wiz = int(st.session_state.wiz)
    if wiz not in dict(WIZARD_STEPS):
        wiz = 1
        st.session_state.wiz = wiz
    scoped = True
    try:
        stepper_ctx = st.container(key=STEPPER_CONTAINER_KEY)
    except TypeError:
        stepper_ctx = st.container()
        scoped = False
    with stepper_ctx:
        st.markdown(_wiz_stepper_dynamic_css(wiz, scoped=scoped), unsafe_allow_html=True)
        try:
            cols = st.columns(len(WIZARD_STEPS), gap="xxsmall")
        except (TypeError, StreamlitInvalidColumnGapError):
            cols = st.columns(len(WIZARD_STEPS))
        for i, (num, short) in enumerate(WIZARD_STEPS):
            with cols[i]:
                sub = short if len(short) <= 20 else (short[:18] + "…")
                label = f"STEP {num}\n{sub}"
                if st.button(
                    label,
                    key=f"wiz_nav_{num}",
                    use_container_width=True,
                    help=f"Open: {short}",
                ):
                    # Session only — avoids query-param serialization cost on every click (bookmark ?wiz= still works on load).
                    st.session_state.wiz = num
    return int(st.session_state.wiz)


def _stage_page_link_grid() -> None:
    st.caption("Use these when you need per-stage drill-down (EDA, GBDT, fusion runner).")
    try:
        r1 = st.columns(3)
        with r1[0]:
            st.page_link("pages/3_EDA.py", label="EDA & figures gallery", icon="📊")
        with r1[1]:
            st.page_link("pages/4_GBDT_Baselines.py", label="GBDT & SHAP", icon="🌲")
        with r1[2]:
            st.page_link("pages/5_Deep_Anomaly.py", label="Deep + anomaly", icon="🧠")
        r2 = st.columns(3)
        with r2[0]:
            st.page_link("pages/6_Fusion_and_Final_Results.py", label="Fusion & downloads", icon="🔗")
        with r2[1]:
            st.page_link("pages/8_Run_Pipeline.py", label="Run pipeline", icon="▶")
        with r2[2]:
            st.page_link("pages/9_Elliptic_Graph.py", label="Elliptic graph", icon="🕸")
    except Exception:
        st.markdown(
            "Open from the project root: `streamlit run app/streamlit_app.py` (or `app/app.py`) and use the `pages/` menu if available."
        )


def render_results_stage_drilldown() -> None:
    """Results step (5): same links as Advanced menu, expanded by default + grid layout."""
    with st.expander(
        "Stage drill-down — open stage-level Streamlit pages",
        expanded=True,
    ):
        _stage_page_link_grid()


def render_advanced_page_links() -> None:
    """Optional collapsed copy (e.g. footer); prefer Results step embed."""
    with st.expander("Advanced — open stage-level Streamlit pages", expanded=False):
        _stage_page_link_grid()
