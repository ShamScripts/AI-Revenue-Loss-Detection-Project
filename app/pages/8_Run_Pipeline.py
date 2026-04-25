import subprocess
import sys
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

import streamlit as st

from components import file_utils as fu
from components.sidebar import render_sidebar_stats
from components.styling import inject_css, section_title

inject_css()
render_sidebar_stats()

section_title("Run pipeline (optional)")
root = fu.get_project_root()
main_py = fu.main_py_path()

st.markdown(
    f"Executes the project **`main.py`** (same as CLI) with working directory:  \n`{root}`"
)

if not main_py.is_file():
    st.error(f"Cannot find main.py at `{main_py}`")
    st.stop()

_labels = [
    "All (1–5)",
    "1 — Data",
    "2 — GBDT",
    "3 — Deep + Anomaly",
    "4 — Fusion",
    "5 — Elliptic GCN + baselines",
]
_choice = st.selectbox("Stage", _labels, index=0)
_stage_val: int | None = {
    "All (1–5)": None,
    "1 — Data": 1,
    "2 — GBDT": 2,
    "3 — Deep + Anomaly": 3,
    "4 — Fusion": 4,
    "5 — Elliptic GCN + baselines": 5,
}[_choice]

col_a, col_b = st.columns(2)
with col_a:
    no_plots = st.checkbox("Skip saving figures (--no-plots)", value=False)
    skip_graph = st.checkbox("Skip Elliptic graph stage (--skip-elliptic-graph)", value=False)
with col_b:
    split_mode = st.selectbox("Split mode", ["random", "temporal"], index=0)
    use_smote = st.checkbox("SMOTE on IEEE train (--smote)", value=False)
    tune_gbdt = st.checkbox("Tune GBDT (--tune-gbdt)", value=False)

st.warning("**Long-running.** Training may take tens of minutes to hours. Keep this tab open.")

if st.button("▶ Run pipeline", type="primary"):
    cmd = [sys.executable, str(main_py)]
    if _stage_val is not None:
        cmd.extend(["--stage", str(_stage_val)])
    if no_plots:
        cmd.append("--no-plots")
    if skip_graph and _stage_val is None:
        cmd.append("--skip-elliptic-graph")
    if split_mode == "temporal":
        cmd.extend(["--split", "temporal"])
    if use_smote:
        cmd.append("--smote")
    if tune_gbdt:
        cmd.append("--tune-gbdt")
    with st.spinner("Running…"):
        try:
            r = subprocess.run(
                cmd,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=86400,
                shell=False,
            )
            st.code((r.stdout or "") + "\n" + (r.stderr or ""), language="text")
            if r.returncode == 0:
                st.success("Finished successfully.")
                try:
                    st.cache_data.clear()
                except Exception:
                    pass
                st.info("Caches cleared — refresh the EDA page to pick up new figures (or reload this page).")
            else:
                st.error(f"Exit code {r.returncode}")
        except subprocess.TimeoutExpired:
            st.error("Timed out.")
        except Exception as e:
            st.exception(e)

with st.expander("Equivalent shell command"):
    line = f'cd "{root}"\npython main.py'
    if _stage_val is not None:
        line += f" --stage {_stage_val}"
    if no_plots:
        line += " --no-plots"
    if skip_graph and _stage_val is None:
        line += " --skip-elliptic-graph"
    if split_mode == "temporal":
        line += " --split temporal"
    if use_smote:
        line += " --smote"
    if tune_gbdt:
        line += " --tune-gbdt"
    st.code(line, language="text")
