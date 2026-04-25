"""Stage / artifact status helpers."""

from __future__ import annotations

import streamlit as st

from components import file_utils as fu


@st.cache_data(show_spinner=False, ttl=45)
def artifact_status() -> dict[str, dict[str, bool]]:
    """Map stage -> filename -> exists."""
    root = fu.processed_dir()
    out: dict[str, dict[str, bool]] = {}
    for stage, names in fu.ARTIFACTS.items():
        out[stage] = {}
        for n in names:
            p = root / n
            out[stage][n] = p.is_file()
    return out


def stage_completion_pct() -> float:
    st = artifact_status()
    total = sum(len(v) for v in st.values())
    if total == 0:
        return 0.0
    ok = sum(1 for d in st.values() for v in d.values() if v)
    return 100.0 * ok / total


def count_artifacts_found() -> tuple[int, int]:
    st = artifact_status()
    total = sum(len(v) for v in st.values())
    ok = sum(1 for d in st.values() for v in d.values() if v)
    return ok, total
