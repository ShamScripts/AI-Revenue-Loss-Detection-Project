"""Inject premium dashboard CSS and presentation blocks."""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st


def _css_path() -> Path:
    return Path(__file__).resolve().parent.parent / "assets" / "custom.css"


@st.cache_data(show_spinner=False)
def _custom_css_text(_mtime: float) -> str:
    p = _css_path()
    if not p.is_file():
        return ""
    return p.read_text(encoding="utf-8")


def inject_css() -> None:
    p = _css_path()
    try:
        mt = p.stat().st_mtime if p.is_file() else 0.0
    except OSError:
        mt = 0.0
    css = _custom_css_text(mt)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def hero(title: str, subtitle: str, badge: str = "Research · Production pipeline") -> None:
    b = html.escape(badge)
    t = html.escape(title)
    s = html.escape(subtitle)
    st.markdown(
        f"""
<div class="hero-wrap">
  <div class="hero-inner">
    <div class="hero-badge">{b}</div>
    <h1 class="hero-title">{t}</h1>
    <p class="hero-sub">{s}</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def executive_summary(text: str) -> None:
    t = html.escape(text)
    st.markdown(
        f"""
<div class="exec-summary">
  <div class="exec-summary-label">Executive summary</div>
  <p class="exec-summary-text">{t}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def stage_timeline(steps: list[tuple[str, str]]) -> None:
    """steps: (short title, one-line description)"""
    parts = []
    for i, (label, desc) in enumerate(steps, start=1):
        l = html.escape(label)
        d = html.escape(desc)
        parts.append(
            f"""<div class="stage-node">
  <div class="stage-num">{i}</div>
  <div class="stage-label">{l}</div>
  <div class="stage-desc">{d}</div>
</div>"""
        )
    st.markdown(f'<div class="stage-rail">{"".join(parts)}</div>', unsafe_allow_html=True)


def dataset_cards(items: list[tuple[str, str, str]]) -> None:
    """(tag, title, body) — compact legacy cards."""
    cols = []
    for tag, title, body in items:
        cols.append(
            f"""<div class="ds-card">
  <span class="ds-card-tag">{html.escape(tag)}</span>
  <div class="ds-card-head">{html.escape(title)}</div>
  <div class="ds-card-body">{html.escape(body)}</div>
</div>"""
        )
    st.markdown(
        f'<div class="ds-grid">{"".join(cols)}</div>',
        unsafe_allow_html=True,
    )


def dataset_overview_cards(
    lead: str,
    items: list[tuple[str, str, str, str, str, list[tuple[str, str]]]],
) -> None:
    """Rich dataset cards: (variant ieee|elliptic, tag, subtitle, title, body, [(stat_label, stat_value), ...])."""
    v_safe = frozenset({"ieee", "elliptic"})
    cards: list[str] = []
    for variant, tag, subtitle, title, body, stats in items:
        v = variant if variant in v_safe else "ieee"
        pills = []
        for lab, val in stats:
            pills.append(
                f'<div class="ds-stat-pill"><span class="ds-stat-lab">{html.escape(lab)}</span>'
                f'<span class="ds-stat-val">{html.escape(val)}</span></div>'
            )
        stat_html = f'<div class="ds-stat-row">{"".join(pills)}</div>' if pills else ""
        cards.append(
            f'<div class="ds-card-v2 ds-card--{html.escape(v)}">'
            f'<span class="ds-card-tag">{html.escape(tag)}</span>'
            f'<div class="ds-card-sub">{html.escape(subtitle)}</div>'
            f'<div class="ds-card-head">{html.escape(title)}</div>'
            f'<div class="ds-card-body">{html.escape(body)}</div>'
            f"{stat_html}</div>"
        )
    lead_h = f'<p class="ds-overview-lead">{html.escape(lead)}</p>' if lead else ""
    st.markdown(
        f'<div class="ds-overview-wrap">{lead_h}<div class="ds-grid">{"".join(cards)}</div></div>',
        unsafe_allow_html=True,
    )


def dataset_insights_split(
    left_title: str,
    left_bullets: list[str],
    right_title: str,
    right_bullets: list[str],
) -> None:
    """Two-column takeaway panels under dataset cards."""

    def _ul(items: list[str]) -> str:
        lis = "".join(f"<li>{html.escape(t)}</li>" for t in items)
        return f"<ul>{lis}</ul>"

    st.markdown(
        f'<div class="ds-insight-split">'
        f'<div class="ds-insight-panel"><h4>{html.escape(left_title)}</h4>{_ul(left_bullets)}</div>'
        f'<div class="ds-insight-panel"><h4>{html.escape(right_title)}</h4>{_ul(right_bullets)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def highlight_list_plain(items: list[tuple[int, str]]) -> None:
    """Numbered highlight rows; text is plain (escaped)."""
    rows = []
    for num, text in items:
        safe = html.escape(text)
        rows.append(
            f'<div class="hl-item"><div class="hl-num">{num}</div>'
            f'<div class="hl-content">{safe}</div></div>'
        )
    st.markdown(f'<div class="hl-grid">{"".join(rows)}</div>', unsafe_allow_html=True)


def presentation_footer(line: str) -> None:
    st.markdown(f'<div class="pres-footer">{html.escape(line)}</div>', unsafe_allow_html=True)


def section_title(text: str, icon: str = "") -> None:
    prefix = f"{icon} " if icon else ""
    t = html.escape(prefix + text)
    st.markdown(f'<div class="section-title">{t}</div>', unsafe_allow_html=True)
