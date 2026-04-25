import sys
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

import streamlit as st

from components import file_utils as fu
from components.figure_captions import caption_for_figure
from components.sidebar import render_sidebar_stats
from components.styling import inject_css, section_title

inject_css()
render_sidebar_stats()

section_title("Exploratory Data Analysis")
st.markdown(
    "Figures are generated when you run the pipeline **without** `--no-plots`. "
    "They live under **`figures/`** (project root). "
    "Plots are **grouped by pipeline stage** (newest first within each group). "
    "Each plot includes a **short caption** for presentations and reports."
)

fig_root = fu.figures_dir()
if not fig_root.is_dir():
    fig_root.mkdir(parents=True, exist_ok=True)


@st.cache_data(ttl=90, show_spinner=False)
def _cached_figure_groups(root_str: str) -> dict[str, list[str]]:
    """Cache figure paths grouped by stage; invalidate ~90s or on root change."""
    root = Path(root_str)
    grouped = fu.figures_grouped_by_stage(root)
    return {k: [str(p) for p in v] for k, v in grouped.items()}


grouped_paths = _cached_figure_groups(str(fig_root.resolve()))
counts = {k: len(v) for k, v in grouped_paths.items()}
total = sum(counts.values())

filter_labels = ["All stages"] + [
    f"Stage {i} ({fu.FIGURE_STAGE_PREFIXES[i - 1]}) — {counts[fu.FIGURE_STAGE_PREFIXES[i - 1]]} fig(s)"
    for i in range(1, len(fu.FIGURE_STAGE_PREFIXES) + 1)
    if counts[fu.FIGURE_STAGE_PREFIXES[i - 1]] > 0
]
if counts["other"] > 0:
    filter_labels.append(f"Other — {counts['other']} fig(s)")

choice = st.selectbox("Filter", filter_labels, index=0)
per_section_cap = st.slider("Max figures per section", min_value=4, max_value=24, value=12, step=2)

STAGE_TITLES = {
    "stage01": "Stage 1 — Data & EDA (IEEE + Elliptic)",
    "stage02": "Stage 2 — GBDT & baselines",
    "stage03": "Stage 3 — Deep learning & anomaly",
    "stage04": "Stage 4 — Fusion & evaluation",
    "stage05": "Stage 5 — Elliptic graph",
    "other": "Other figures",
}


def render_one_image(img_path: Path) -> None:
    title, explain = caption_for_figure(img_path)
    st.markdown("---")
    st.markdown(f"##### {title}")
    st.caption(explain)
    try:
        st.image(str(img_path), use_container_width=True)
    except Exception as e:
        st.error(f"{img_path.name}: {e}")
    st.caption(f"`{img_path.name}` · `{img_path.parent.name}/`")


if total == 0:
    st.warning(
        f"No figures found under `{fu.figures_dir()}`. "
        "Run `python main.py` (with plots enabled) to generate PNGs."
    )
else:
    st.success(
        f"**{total}** figure(s) discovered · "
        f"Stage 1–5: {sum(counts[k] for k in fu.FIGURE_STAGE_PREFIXES)} · Other: {counts['other']}"
    )

    def should_show_section(stage_key: str) -> bool:
        if choice.startswith("All"):
            return True
        if choice.startswith("Other"):
            return stage_key == "other"
        for i, prefix in enumerate(fu.FIGURE_STAGE_PREFIXES, start=1):
            if choice.startswith(f"Stage {i}"):
                return stage_key == prefix
        return True

    shown = 0
    for stage_key in (*fu.FIGURE_STAGE_PREFIXES, "other"):
        if not should_show_section(stage_key):
            continue
        paths_str = grouped_paths.get(stage_key, [])
        if not paths_str:
            continue
        st.subheader(STAGE_TITLES.get(stage_key, stage_key))
        n = 0
        for ps in paths_str:
            if n >= per_section_cap:
                remaining = len(paths_str) - per_section_cap
                if remaining > 0:
                    st.caption(f"*…and {remaining} more in this section (raise “Max figures per section” or pick another filter).*")
                break
            render_one_image(Path(ps))
            n += 1
            shown += 1

    with st.expander("Figure naming & scan paths"):
        st.markdown(
            """
- **Stage 1** figures use the prefix `stage01_` (IEEE + Elliptic EDA).
- **Stages 2–5** use `stage02_` … `stage05_` for model and fusion plots.
- Unknown filenames appear under **Other** with a **generic** explanation; extend `app/components/figure_captions.py` for custom names.
- Figure list is **cached ~90s**; refresh the page to pick up new PNGs immediately after a run.
"""
        )
        st.code(str(fu.figures_dir()), language="text")
