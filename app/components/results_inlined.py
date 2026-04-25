"""Stage-scoped artifacts and categorized figures (wizard distributes sections across steps)."""

from __future__ import annotations

import html
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import streamlit as st

from components import file_utils as fu
from components import figure_story as fst
from components.cards import info_panel
from components.figure_captions import caption_for_figure
from components.tables import preview_df

_RIBBON_ALLOW = frozenset(
    {
        "tr-sec-ieee",
        "tr-sec-elliptic",
        "tr-sec-gbdt",
        "tr-sec-deep",
        "tr-sec-fusion",
        "tr-sec-graph",
        "tr-sec-other",
        "tr-artifacts",
    }
)


def _ribbon(css_class: str, text: str) -> None:
    cls = css_class if css_class in _RIBBON_ALLOW else "tr-sec-other"
    st.markdown(
        f'<div class="table-ribbon {cls}">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def _figure_bucket(path: Path) -> str:
    s = path.stem.lower()
    if s.startswith("stage05"):
        return "graph"
    if s.startswith("stage04"):
        return "fusion"
    if s.startswith("stage03"):
        return "deep"
    if s.startswith("stage02"):
        return "gbdt"
    if s.startswith("stage01") and "elliptic" in s:
        return "elliptic_eda"
    if s.startswith("stage01"):
        return "ieee_eda"
    return "other"


@st.cache_data(show_spinner=False, ttl=90)
def _all_png_paths_cached(root_resolved: str) -> tuple[str, ...]:
    root = Path(root_resolved)
    if not root.is_dir():
        return ()
    return tuple(sorted({str(p) for p in root.rglob("*.png") if p.is_file()}, key=lambda x: x.lower()))


def _all_png_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    try:
        resolved = str(root.resolve())
    except OSError:
        resolved = str(root)
    return [Path(p) for p in _all_png_paths_cached(resolved)]


def _bucket_figures(root: Path) -> dict[str, list[Path]]:
    keys = ("ieee_eda", "elliptic_eda", "gbdt", "deep", "fusion", "graph", "other")
    out: dict[str, list[Path]] = {k: [] for k in keys}
    for p in _all_png_paths(root):
        b = _figure_bucket(p)
        if b not in out:
            b = "other"
        out[b].append(p)
    return out


def render_figure_with_narrative(path: Path) -> None:
    if not path.is_file():
        return
    stem = path.stem
    title, explanation = caption_for_figure(path)
    inf, biz = fst.inference_and_business_for_stem(stem)
    with st.container(border=True):
        st.markdown(
            f'<div class="figure-panel" style="border:none;box-shadow:none;padding:0;margin:0">'
            f"<h5>{html.escape(title)}</h5>"
            f'<p class="exp"><strong>Explanation.</strong> {html.escape(explanation)}</p>'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.image(str(path), use_container_width=True)
        st.markdown(
            f'<div class="inference-callout"><strong>Inference.</strong> {html.escape(inf)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="story-card" style="margin-top:0.75rem"><h4>Business meaning</h4>'
            f'<p style="margin:0;color:#475569;font-size:0.92rem;line-height:1.6">{html.escape(biz)}</p></div>',
            unsafe_allow_html=True,
        )
    st.divider()


FIGURE_SECTIONS: tuple[tuple[str, str, str, str], ...] = (
    (
        "ieee_eda",
        "tr-sec-ieee",
        "IEEE · data & distributions",
        "Class balance, correlations, transaction histograms, and temporal views for IEEE-CIS.",
    ),
    (
        "elliptic_eda",
        "tr-sec-elliptic",
        "Elliptic · data & graph context",
        "Elliptic class mix, timestep activity, feature KDEs, and network-oriented EDA plots.",
    ),
    (
        "gbdt",
        "tr-sec-gbdt",
        "Model diagnostics · GBDT",
        "Feature importance, SHAP, and other Stage-2 tree diagnostics.",
    ),
    (
        "deep",
        "tr-sec-deep",
        "Model diagnostics · deep & anomaly",
        "DNN training curves, confusion views, and Isolation Forest score densities.",
    ),
    (
        "fusion",
        "tr-sec-fusion",
        "Fusion · performance & calibration",
        "ROC/PR, fusion confusion, and other Stage-4 evaluation figures.",
    ),
    (
        "graph",
        "tr-sec-graph",
        "Graph · Elliptic GCN",
        "GCN score histograms and other Stage-5 graph experiment plots.",
    ),
    (
        "other",
        "tr-sec-other",
        "Other pipeline figures",
        "Any additional PNG exports not classified above (still part of the run).",
    ),
)


def render_categorized_figure_gallery(
    fig_dir: Path,
    *,
    bucket_keys: Sequence[str] | None = None,
    section_title: str | None = None,
) -> None:
    """If bucket_keys is set, only those buckets (in FIGURE_SECTIONS order) are shown."""
    buckets = _bucket_figures(fig_dir)
    if bucket_keys is None:
        keys_order = [s[0] for s in FIGURE_SECTIONS]
        sections = list(FIGURE_SECTIONS)
        total = sum(len(v) for v in buckets.values())
    else:
        want = frozenset(bucket_keys)
        keys_order = [s[0] for s in FIGURE_SECTIONS if s[0] in want]
        sections = [s for s in FIGURE_SECTIONS if s[0] in want]
        total = sum(len(buckets.get(k, [])) for k in keys_order)
    if section_title:
        st.markdown(f"### {section_title}")
    st.caption(f"**{total}** PNG(s) in this view — under `figures/` on disk.")
    for key, css, title, blurb in sections:
        paths = buckets.get(key, [])
        _ribbon(css, title.replace(" · ", " · ").upper())
        st.markdown(f"##### {title}")
        st.caption(blurb)
        if not paths:
            st.info(f"No figures in **{title}** yet — run the pipeline with plots enabled.")
            continue
        for p in paths:
            render_figure_with_narrative(p)


def _download(path: Path, key: str) -> None:
    if not path.is_file():
        return
    try:
        data = path.read_bytes()
    except OSError:
        return
    st.download_button(
        label=f"Download `{path.name}`",
        data=data,
        file_name=path.name,
        mime="text/csv",
        key=key,
    )


def render_rerun_and_configs(proc: Path) -> None:
    """CLI re-run recipe + saved experiment JSON (Pipeline step)."""
    with st.expander("Re-run stages & experiment flags", expanded=False):
        st.markdown("Run from project root (Python **3.10–3.12** recommended):")
        st.code(
            "python main.py --stage 1   # data\n"
            "python main.py --stage 2   # GBDT\n"
            "python main.py --stage 3   # deep + anomaly\n"
            "python main.py --stage 4   # fusion + report tables\n"
            "python main.py --stage 5   # Elliptic graph",
            language="bash",
        )
        for name, p in (
            ("Stage 4", proc / "stage04_experiment_config.json"),
            ("Stage 3", proc / "stage03_experiment_config.json"),
            ("Stage 2", proc / "stage02_experiment_config.json"),
            ("Stage 5", proc / "stage05_experiment_config.json"),
        ):
            raw = fu.safe_read_text(p, max_chars=6000)
            if raw:
                st.markdown(f"**{name}** — `{p.name}`")
                st.code(raw, language="json")


def render_stage2_gbdt_section(proc: Path, *, download_key: str = "dl_gbdt_preds_modeling") -> None:
    with st.expander("Stage 2 · GBDT scores & distribution", expanded=True):
        pred_path = proc / "gbdt_preds.csv"
        if not pred_path.is_file():
            st.warning("`gbdt_preds.csv` not found — run Stage 2.")
        else:
            df = fu.safe_read_csv(pred_path, nrows=100_000)
            if df is not None:
                st.success(f"**{len(df):,}** rows in `gbdt_preds.csv`.")
                if "gbdt_pred_proba" in df.columns:
                    s = df["gbdt_pred_proba"]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Mean prob", f"{float(s.mean()):.4f}")
                    c2.metric("Std", f"{float(s.std()):.4f}")
                    c3.metric("Min / Max", f"{float(s.min()):.3f} / {float(s.max()):.3f}")
                    try:
                        import plotly.express as px

                        fig = px.histogram(s, nbins=60, title="GBDT predicted probability distribution")
                        fig.update_layout(template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.bar_chart(s.value_counts(bins=20))
                preview_df(df, "Preview", max_rows=40)
                _download(pred_path, key=download_key)


def render_stage3_deep_section(proc: Path, *, download_key_csv: str = "dl_hybrid_dnn_modeling", download_key_base: str = "dl_stage03_baselines_modeling") -> None:
    with st.expander("Stage 3 · Deep learning & anomaly", expanded=True):
        p3 = proc / "hybrid_dnn_anomaly_preds.csv"
        if not p3.is_file():
            st.warning("`hybrid_dnn_anomaly_preds.csv` not found — run Stage 3.")
        else:
            df = fu.safe_read_csv(p3, nrows=100_000)
            if df is not None:
                st.success(f"**{len(df):,}** rows.")
                cols = [c for c in ("dnn_pred_proba", "anomaly_score", "hybrid_score", "isFraud") if c in df.columns]
                for c in cols:
                    if c != "isFraud" and str(df[c].dtype) != "object":
                        try:
                            import plotly.express as px

                            fig = px.histogram(df[c].dropna(), nbins=50, title=c)
                            fig.update_layout(template="plotly_white", showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass
                preview_df(df, "Preview", max_rows=35)
                _download(p3, key=download_key_csv)
        base = proc / "stage03_ieee_dnn_baselines.csv"
        if base.is_file():
            st.markdown("**Attention vs plain MLP (validation)** — `stage03_ieee_dnn_baselines.csv`")
            st.dataframe(fu.safe_read_csv(base, nrows=20), use_container_width=True, hide_index=True)
            _download(base, key=download_key_base)
        info_panel(
            "Score columns",
            "**dnn_pred_proba** — neural fraud probability. **anomaly_score** — normalized IF score. "
            "**hybrid_score** — Stage-3 blend before final fusion weights.",
        )


def render_stage4_fusion_section(
    proc: Path,
    *,
    download_key_metrics: str = "dl_metrics_modeling",
    download_key_scores: str = "dl_scores_modeling",
) -> None:
    with st.expander("Stage 4 · Fusion outputs & charts", expanded=True):
        t_path = proc / "final_hybrid_threshold.txt"
        if t_path.is_file():
            thr = fu.safe_read_text(t_path, max_chars=200)
            st.metric("Selected threshold (validation F1)", thr.strip() if thr else "—")
        m_path = proc / "final_hybrid_comparison_metrics.csv"
        s_path = proc / "final_hybrid_scores.csv"
        metrics = fu.safe_read_csv(m_path, nrows=500)
        if metrics is not None and not metrics.empty:
            st.markdown("**Component comparison (same file as TABLE III source)**")
            st.dataframe(metrics, use_container_width=True, hide_index=True)
            _download(m_path, key=download_key_metrics)
            try:
                import plotly.express as px

                pal = ("#2563eb", "#7c3aed", "#db2777", "#059669", "#d97706", "#64748b")
                if "model" in metrics.columns and "roc_auc" in metrics.columns:
                    fig = px.bar(
                        metrics, x="model", y="roc_auc", title="ROC-AUC by model", color="model", color_discrete_sequence=pal
                    )
                    fig.update_layout(template="plotly_white", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                if "model" in metrics.columns and "pr_auc" in metrics.columns:
                    fig2 = px.bar(
                        metrics, x="model", y="pr_auc", title="PR-AUC by model", color="model", color_discrete_sequence=pal
                    )
                    fig2.update_layout(template="plotly_white", showlegend=False)
                    st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                pass
        scores = fu.safe_read_csv(s_path, nrows=25_000)
        if scores is not None and not scores.empty:
            st.markdown("**`final_hybrid_scores.csv`** (preview)")
            preview_df(scores, max_rows=25)
            _download(s_path, key=download_key_scores)


def render_stage5_elliptic_graph_section(proc: Path, *, download_key: str = "dl_elliptic_graph_modeling") -> None:
    with st.expander("Stage 5 · Elliptic graph experiments", expanded=True):
        eg = proc / "elliptic_graph_experiments.csv"
        if eg.is_file():
            df = fu.safe_read_csv(eg, nrows=100)
            st.dataframe(df, use_container_width=True, hide_index=True)
            _download(eg, key=download_key)
        else:
            st.info("`elliptic_graph_experiments.csv` not found — run Stage 5 with Elliptic data + torch.")


def render_modeling_pipeline_sections(proc: Path) -> None:
    """Stages 2–5 artifacts and previews (Modeling wizard step)."""
    st.markdown("### Pipeline outputs by stage")
    st.caption("Same artifacts as under `processed_data/`; standalone pages remain under `app/pages/`.")
    render_stage2_gbdt_section(proc)
    render_stage3_deep_section(proc)
    render_stage4_fusion_section(proc)
    render_stage5_elliptic_graph_section(proc)


def render_inlined_pipeline_sections(proc: Path) -> None:
    """Compose full inlined stack (e.g. legacy Results view). Prefer granular render_* calls from the wizard."""
    st.markdown("### Pipeline detail (inlined)")
    st.caption("Everything below also exists as standalone pages under `app/pages/` for deep dives.")
    render_rerun_and_configs(proc)
    render_modeling_pipeline_sections(proc)


def render_supplementary_result_tables(
    proc: Path,
    *,
    only_filenames: frozenset[str] | None = None,
) -> None:
    """CSV outputs that support the report but are not TABLE I–IV.

    If only_filenames is set, only those basenames are listed (e.g. fusion-only on Results).
    """
    _ribbon("tr-artifacts", "ARTIFACT TABLES · SUPPORTING CSVs")
    st.markdown("##### Raw outputs used by fusion & report generation")
    st.caption("Same files on disk as the pipeline writes — nothing synthesized here.")

    spec: tuple[tuple[str, str, int], ...] = (
        ("final_hybrid_scores.csv", "Per-record hybrid and component scores (large file).", 15),
        ("gbdt_preds.csv", "GBDT probabilities merged to the modeling cohort.", 12),
        ("hybrid_dnn_anomaly_preds.csv", "DNN + anomaly + Stage-3 hybrid columns.", 12),
        ("stage03_ieee_dnn_baselines.csv", "Attention vs plain MLP validation ROC-AUC.", 20),
        ("elliptic_graph_experiments.csv", "Elliptic temporal / GCN benchmark rows.", 50),
    )
    rows: list[tuple[str, Path, int, str]] = []
    for name, blurb, cap in spec:
        if only_filenames is not None and name not in only_filenames:
            continue
        p = proc / name
        if p.is_file():
            rows.append((name, p, cap, blurb))

    if not rows:
        st.warning("No supplementary CSVs found yet — run the pipeline.")
        return

    for name, p, cap, blurb in rows:
        st.markdown(f"**`{name}`** — {blurb} (_up to {cap} rows_)")
        df = fu.safe_read_csv(p, nrows=cap + 5)
        if df is not None:
            st.dataframe(df.head(cap), use_container_width=True, hide_index=True)
