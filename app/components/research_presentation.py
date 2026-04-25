"""Research dashboard content by wizard step (tables/figures unchanged — UI + narrative only)."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from components import file_utils as fu
from components import results_inlined as rin
from components import stage_utils as su
from components.cards import info_panel, premium_kpi_row
from components.styling import (
    dataset_insights_split,
    dataset_overview_cards,
    executive_summary,
    hero,
    highlight_list_plain,
    presentation_footer,
    section_title,
)

FUSION_W_GBDT = 0.45
FUSION_W_DNN = 0.40
FUSION_W_ANOM = 0.15

_PLOTLY_CHART_CONFIG: dict = {"displayModeBar": False, "responsive": True}

PIPELINE_STEPS: list[dict[str, str]] = [
    {
        "title": "1. Data Collection",
        "method": "IEEE-CIS transactional records and identity tables; Elliptic Bitcoin features, classes, and transaction graph edge list.",
        "models": "— (ingestion)",
        "output": "`ieee_train_merged_cleaned.csv`, Elliptic raw → cleaned tables under `processed_data/`.",
        "why": "Establishes a single source of truth for supervised labels and graph topology used downstream.",
        "rail": "data",
    },
    {
        "title": "2. Data Preprocessing",
        "method": "Sparse-column removal, median imputation, type coercion, merge on keys; Elliptic licit/illicit labeling and degree features.",
        "models": "—",
        "output": "`ieee_train_eda_ready.csv`, `elliptic_transactions_cleaned.csv`, `preprocessing_config.json`.",
        "why": "Reduces noise and missingness so GBDT/DNN training is stable and comparable across runs.",
        "rail": "pre",
    },
    {
        "title": "3. Feature Engineering",
        "method": "Time/amount signals, categorical handling, graph-derived degrees and ratios on Elliptic; GBDT probability injected as a feature for the deep channel.",
        "models": "Feature pipelines before Stage 2–3.",
        "output": "Engineered columns in EDA-ready frames; `gbdt_preds.csv` feeds hybrid deep+anomaly inputs.",
        "why": "Captures both raw transactional risk and relational structure aligned with the report’s methodology.",
        "rail": "feat",
    },
    {
        "title": "4. Hybrid Modeling",
        "method": "Parallel channels: GBDT (LightGBM/XGBoost-style stack), attention DNN + optional sklearn MLP fallback, Isolation Forest anomaly scores; min–max normalization per channel.",
        "models": "GBDT · Attention DNN / MLP · Isolation Forest",
        "output": "`gbdt_preds.csv`, `hybrid_dnn_anomaly_preds.csv`.",
        "why": "Trees excel at tabular interactions; deep models capture non-linear patterns; IF highlights outliers missed by supervised heads.",
        "rail": "model",
    },
    {
        "title": "5. Expected vs Actual Revenue Analysis",
        "method": "Score distributions compared to realized fraud labels on the held-out test cohort; operating point from validation F1.",
        "models": "Fusion layer + threshold file",
        "output": "`final_hybrid_scores.csv`, `final_hybrid_comparison_metrics.csv`, `final_hybrid_threshold.txt`.",
        "why": "Connects model scores to observed fraud incidence—proxy for expected vs realized loss when amounts are joined.",
        "rail": "result",
    },
    {
        "title": "6. Revenue Leakage Estimation",
        "method": "Flag high hybrid scores; join transaction amounts where available to quantify flagged monetary exposure.",
        "models": "Hybrid score + business parameters",
        "output": "Dashboard KPIs; optional export from filtered score table.",
        "why": "Translates ranking quality into a finance-facing narrative: missed positives ≈ leakage risk.",
        "rail": "risk",
    },
    {
        "title": "7. Risk Scoring and Ranking",
        "method": "Weighted fusion P = w₁P_GBDT + w₂P_DL + w₃P_AD with tuned threshold τ on validation.",
        "models": "Hybrid (GBDT + DNN + anomaly)",
        "output": "`hybrid_weighted_score`, ranked `record_id` list.",
        "why": "Single interpretable risk score for investigators and for monitoring rules.",
        "rail": "risk",
    },
    {
        "title": "8. Final Output",
        "method": "Thresholded alerts, report tables (IEEE, Elliptic, ablation), and manuscript-ready figures.",
        "models": "Full stack",
        "output": "`report_table_*.csv`, `figures/stage04_*.png`, etc.",
        "why": "Deliverables for academic review and executive readouts in one reproducible bundle.",
        "rail": "out",
    },
]

_RAIL_CLASS = {
    "data": "rail-data",
    "pre": "rail-pre",
    "feat": "rail-feat",
    "model": "rail-model",
    "result": "rail-result",
    "risk": "rail-risk",
    "out": "rail-out",
}


def _proc() -> Path:
    return fu.processed_dir()


def _fig() -> Path:
    return fu.figures_dir()


_HERO_SCORES_CAP = 50_000


@st.cache_data(show_spinner=False)
def _load_scores_for_hero() -> pd.DataFrame | None:
    """Capped cohort for KPIs / sliders — full file not needed for dashboard metrics."""
    p = _proc() / "final_hybrid_scores.csv"
    if not p.is_file():
        return None
    want = [
        "record_id",
        "target",
        "hybrid_weighted_score",
        "gbdt_score",
        "dnn_score",
        "anomaly_score",
    ]
    try:
        head = pd.read_csv(p, nrows=0)
        cols = [c for c in want if c in head.columns]
        if cols:
            return pd.read_csv(p, usecols=cols, nrows=_HERO_SCORES_CAP)
        return pd.read_csv(p, nrows=_HERO_SCORES_CAP)
    except Exception:
        try:
            return pd.read_csv(p, nrows=_HERO_SCORES_CAP)
        except Exception:
            return None


_AMTS_JOIN_CAP = 400_000


@st.cache_data(show_spinner=False)
def _load_amts_for_leakage() -> pd.DataFrame | None:
    """Capped read — KPI join only needs overlap with scored record_ids."""
    p = _proc() / "ieee_train_eda_ready.csv"
    if not p.is_file():
        return None
    try:
        return pd.read_csv(p, usecols=["TransactionID", "TransactionAmt"], nrows=_AMTS_JOIN_CAP)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _ieee_fraud_counts_sample(n: int = 250_000) -> tuple[int, int] | None:
    p = _proc() / "ieee_train_eda_ready.csv"
    if not p.is_file():
        return None
    try:
        df = pd.read_csv(p, usecols=["isFraud"], nrows=n)
        vc = df["isFraud"].value_counts()
        neg = int(vc.get(0, 0))
        pos = int(vc.get(1, 0))
        return neg, pos
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _ieee_eda_quick_stats() -> tuple[int | None, float | None, int | None]:
    """Rows read for class sample, fraud rate in that sample, column count from header."""
    proc = _proc()
    p = proc / "ieee_train_eda_ready.csv"
    ncols: int | None = None
    if p.is_file():
        try:
            ncols = len(pd.read_csv(p, nrows=0).columns)
        except Exception:
            pass
    samp = _ieee_fraud_counts_sample(300_000)
    if samp is None:
        return None, None, ncols
    neg, pos = samp
    n = neg + pos
    if n == 0:
        return None, None, ncols
    return n, float(pos) / float(n), ncols


@st.cache_data(show_spinner=False)
def _read_threshold_cached(file_mtime: float) -> float:
    p = _proc() / "final_hybrid_threshold.txt"
    raw = fu.safe_read_text(p, max_chars=50)
    if not raw:
        return 0.56
    try:
        return float(raw.strip())
    except ValueError:
        return 0.56


def _read_threshold() -> float:
    p = _proc() / "final_hybrid_threshold.txt"
    try:
        mt = p.stat().st_mtime
    except OSError:
        mt = 0.0
    return _read_threshold_cached(mt)


def _rename_comparison_models(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "model" not in out.columns:
        return out
    m = {
        "Hybrid_Weighted": "Proposed Hybrid Model",
        "Hybrid_From_Stage03": "Intermediate Hybrid",
        "DNN": "Deep Neural Network",
        "GBDT": "GBDT (XGBoost stack)",
        "Anomaly": "Isolation Forest",
    }
    out["model"] = out["model"].astype(str).map(lambda x: m.get(x, x))
    return out


def _safe_df(path: Path) -> pd.DataFrame | None:
    return fu.safe_read_csv(path, nrows=500)


def _table_shell(df: pd.DataFrame | None) -> None:
    st.markdown('<div class="table-wrap">', unsafe_allow_html=True)
    if df is not None and not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True, height=min(460, 42 * (len(df) + 2)))
    else:
        st.caption("No rows to display.")
    st.markdown("</div>", unsafe_allow_html=True)


def _table_ribbon(css_class: str, text: str) -> None:
    allowed = {"tr-ieee", "tr-elliptic", "tr-compare", "tr-ablate", "tr-fig"}
    cls = css_class if css_class in allowed else "tr-ieee"
    st.markdown(
        f'<div class="table-ribbon {cls}">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def _narrative_block(title: str, rail: str, bullets: list[str]) -> None:
    rc = _RAIL_CLASS.get(rail, "rail-result")
    items = "".join(f"<li>{html.escape(b)}</li>" for b in bullets)
    st.markdown(
        f'<div class="story-flex"><div class="story-rail {rc}"></div><div class="story-card" style="flex:1">'
        f"<h4>{html.escape(title)}</h4><ul style='margin:0;padding-left:1.1rem;color:#475569;font-size:0.92rem;line-height:1.55'>{items}</ul></div></div>",
        unsafe_allow_html=True,
    )


def _kpi_items(scores: pd.DataFrame | None, thr_default: float) -> list[tuple[str, str, str]]:
    fraud_rate_txt = "—"
    n_tx = "—"
    high_risk_txt = "—"
    leak_txt = "—"
    if scores is not None and len(scores) > 0:
        n = len(scores)
        n_tx = f"{n:,}"
        fr = float(scores["target"].mean())
        fraud_rate_txt = f"{100 * fr:.2f}%"
        t0 = thr_default
        flagged = scores["hybrid_weighted_score"] >= t0
        high_risk_txt = f"{100 * float(flagged.mean()):.2f}%"
        amts = _load_amts_for_leakage()
        if amts is not None and "hybrid_weighted_score" in scores.columns:
            sc2 = scores[["record_id", "hybrid_weighted_score"]].copy()
            sc2["flag"] = sc2["hybrid_weighted_score"] >= t0
            j = sc2.merge(amts.rename(columns={"TransactionID": "record_id"}), on="record_id", how="inner")
            jf = j[j["flag"]]
            if len(jf) > 0:
                leak_est = float(jf["TransactionAmt"].sum())
                leak_txt = f"≈ ${leak_est:,.0f} flagged transaction amount (sum)"
            else:
                leak_txt = "— (no rows above τ)"
        else:
            leak_txt = "Join amounts: ensure `ieee_train_eda_ready.csv` exists"
    base = [
        ("Total transactions (IEEE cohort)", n_tx, "Rows in `final_hybrid_scores.csv` when present"),
        ("Fraud rate (test / scored cohort)", fraud_rate_txt, "Share of positive labels — severe imbalance"),
        ("Estimated leakage signal", leak_txt, "Flagged hybrid scores × transaction amounts where joined"),
        ("High-risk share @ default τ", high_risk_txt, f"Hybrid score ≥ {thr_default:.3f} (from validation F1)"),
    ]
    icons = ("📋", "⚖️", "💸", "🎯")
    return [(icons[i], base[i][0], base[i][1], base[i][2]) for i in range(len(base))]


def render_wizard_step(step: int) -> None:
    proc = _proc()
    fig_dir = _fig()
    # Avoid loading large score CSV on every step — only Overview + Business impact need it.
    scores = _load_scores_for_hero() if step in (1, 6) else None
    thr_default = _read_threshold()

    if step == 1:
        hero(
            "AI-Powered Revenue Leakage Detection System",
            "Hybrid ML + Deep Learning + Anomaly Detection Framework — proactive detection of hidden revenue loss points "
            "using weighted fusion aligned with the project report.",
            badge="Research · Fintech presentation",
        )
        st.caption(
            "KPIs use the first **50k** scored rows and a capped amount join for speed; export `final_hybrid_scores.csv` for the full cohort."
        )
        premium_kpi_row(_kpi_items(scores, thr_default))
        executive_summary(
            "This system integrates GBDT, attention-based deep learning, and anomaly detection to proactively detect "
            "hidden revenue loss points. Channels are fused as P_final = w₁·P_GBDT + w₂·P_DL + w₃·P_AD with weights "
            f"{FUSION_W_GBDT}, {FUSION_W_DNN}, {FUSION_W_ANOM} and a tuned threshold, then evaluated on the IEEE-CIS "
            "held-out set and benchmarked against Elliptic graph baselines in the report."
        )
        section_title("Problem → data → model → results (story arc)")
        with st.container(border=True):
            st.markdown(
                "Use the **horizontal steps** above: **Pipeline** (flow + how to re-run), **Dataset & EDA** (sources, imbalance, "
                "EDA figures), **Modeling** (per-stage CSVs, plots, GBDT/deep diagnostics), **Results** (TABLE I–IV plus "
                "fusion/graph figures), **Business impact**, then **Reports**."
            )
        section_title("Smart insight panel (auto)")
        t1 = _safe_df(proc / "report_table_1_ieee_cis.csv")
        mcomp = _safe_df(proc / "final_hybrid_comparison_metrics.csv")
        insights: list[str] = [
            "Dataset is highly imbalanced → recall and PR-AUC should be prioritized alongside ROC-AUC.",
            "Hybrid model improves F1-score vs single channels when thresholds are tuned for operations.",
            "GBDT has high recall but low precision → more false alerts and investigation cost.",
            "Anomaly detection helps find unknown fraud patterns but is weak alone at 0.5 on normalized scores.",
            "Combining models improves robustness vs any single score (see ablation in TABLE IV).",
        ]
        if t1 is not None and not t1.empty and "Proposed Hybrid Model" in t1["Model"].values:
            hr = t1[t1["Model"] == "Proposed Hybrid Model"].iloc[0]
            insights.insert(
                0,
                f"TABLE I — Proposed Hybrid F1 **{float(hr['F1-Score']):.4f}** vs baselines on the IEEE-CIS cohort.",
            )
        if mcomp is not None and "model" in mcomp.columns:
            gb = mcomp[mcomp["model"].astype(str).str.contains("GBDT", na=False)]
            if not gb.empty and "precision" in gb.columns:
                insights.append(
                    f"GBDT precision **{float(gb.iloc[0]['precision']):.3f}** at 0.5 on normalized scores — noisy alert stream if used alone."
                )
        li = "".join(f"<li>{html.escape(s)}</li>" for s in insights)
        st.markdown(
            f'<div class="insight-board"><h3>Smart insights</h3><ul>{li}</ul></div>',
            unsafe_allow_html=True,
        )
        ok, tot = su.count_artifacts_found()
        pct = int(100 * ok / tot) if tot else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Pipeline artifacts", f"{ok} / {tot}", delta=f"{pct}% found" if tot else None)
        c2.metric("Project root", fu.get_project_root().name)
        ts = fu.newest_core_artifact_mtime()
        c3.metric("Newest core artifact (UTC)", ts.strftime("%Y-%m-%d %H:%M") if ts else "—")

    elif step == 2:
        section_title("Pipeline (report Figure 1 narrative)")
        st.caption("Color key: Data · Preprocessing · Features · Modeling · Results · Risk · Output")
        for ps in PIPELINE_STEPS:
            with st.expander(ps["title"], expanded=False):
                st.markdown(f"**What happens:** {ps['method']}")
                st.markdown(f"**Models:** {ps['models']}")
                st.markdown(f"**Outputs:** {ps['output']}")
                st.markdown(f"**Why it matters:** {ps['why']}")
        st.divider()
        rin.render_rerun_and_configs(proc)

    elif step == 3:
        section_title("Datasets (Section IV)")
        st.caption("Two complementary benchmarks: real-world card-not-present fraud (tabular) vs money-flow graphs on-chain.")
        st5 = fu.safe_read_json(proc / "stage05_experiment_config.json")
        elliptic_nodes = int(st5["n_nodes"]) if isinstance(st5, dict) and "n_nodes" in st5 else None
        elliptic_edges = int(st5["n_edges_used"]) if isinstance(st5, dict) and "n_edges_used" in st5 else None
        ieee_n, ieee_fr, ieee_cols = _ieee_eda_quick_stats()
        ieee_stats: list[tuple[str, str]] = []
        if ieee_n is not None:
            ieee_stats.append(("Class-balance sample", f"{ieee_n:,} rows"))
        if ieee_fr is not None:
            ieee_stats.append(("Fraud share (sample)", f"{100 * ieee_fr:.2f}%"))
        if ieee_cols is not None:
            ieee_stats.append(("Columns (schema width)", f"{ieee_cols:,}"))
        if not ieee_stats:
            ieee_stats.append(("Status", "Run Stage 1 for EDA file"))

        elliptic_stats: list[tuple[str, str]] = [
            ("Node features (source)", "~166"),
        ]
        if elliptic_nodes and elliptic_edges:
            elliptic_stats.insert(0, ("GCN subgraph nodes", f"{elliptic_nodes:,}"))
            elliptic_stats.insert(1, ("Undirected edge pairs", f"{elliptic_edges:,}"))
        else:
            elliptic_stats.insert(0, ("Graph run", "Run Stage 5 for subgraph stats"))

        dataset_overview_cards(
            "Use the stats below as at-a-glance scale signals; modeling cohorts may differ after filtering and splits.",
            [
                (
                    "ieee",
                    "IEEE-CIS",
                    "Payments · supervised tabular",
                    "Tabular fraud classification",
                    "Identity-linked transactions with a sparse, wide schema and extreme class imbalance. "
                    "Primary channel for the GBDT + attention DNN + anomaly fusion stack in this project.",
                    ieee_stats,
                ),
                (
                    "elliptic",
                    "Elliptic",
                    "On-chain · graph & time",
                    "Graph-structured flows",
                    "Transactions as nodes, capital flows as directed edges, and rich per-node attributes. "
                    "Temporal train/val/test avoids leakage; GCN and tabular baselines are reported side by side.",
                    elliptic_stats,
                ),
            ],
        )
        dataset_insights_split(
            "Why IEEE-CIS here",
            [
                "High-dimensional tabular geometry suits gradient boosting and learned embeddings.",
                "Severe imbalance forces PR-AUC and recall-aware thresholds—not accuracy alone.",
                "Maps cleanly to revenue-at-risk when amounts and labels align on transaction IDs.",
            ],
            "Why Elliptic here",
            [
                "Relational topology exposes rings, peels, and flow concentration missed by row-only models.",
                "Benchmarks graph-aware methods (e.g. GCN) against strong tree and MLP baselines.",
                "Supports a second narrative: hidden leakage paths through coordinated wallets, not isolated charges.",
            ],
        )
        section_title("EDA & preprocessing")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Class imbalance (IEEE sample)")
            samp = _ieee_fraud_counts_sample(300_000)
            if samp:
                neg, pos = samp
                try:
                    import plotly.graph_objects as go

                    fig = go.Figure(
                        data=[
                            go.Pie(
                                labels=["Non-fraud", "Fraud"],
                                values=[neg, pos],
                                hole=0.45,
                                marker=dict(colors=["#94a3b8", "#dc2626"]),
                            )
                        ]
                    )
                    fig.update_layout(template="plotly_white", height=320, showlegend=True, margin=dict(t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CHART_CONFIG)
                    st.caption("Shows relative class frequency in a capped sample for responsiveness.")
                    st.markdown(
                        "<div class=\"inference-callout\"><strong>Inference.</strong> "
                        "Extreme imbalance makes recall and PR-AUC primary quality indicators.</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        '<div class="story-card"><h4>Business meaning</h4><p style="margin:0;color:#475569">'
                        "Imbalance implies most revenue loss comes from rare events — models must not optimize accuracy alone.</p></div>",
                        unsafe_allow_html=True,
                    )
                except Exception:
                    st.bar_chart(pd.Series({"Non-fraud": neg, "Fraud": pos}))
            else:
                st.info("Train EDA file not found — run Stage 1.")
        with c2:
            st.markdown("##### Missing values (top drivers)")
            miss = _safe_df(proc / "ieee_missing_top20_summary.csv")
            if miss is not None and not miss.empty and "column" in miss.columns:
                pct_col = "missing_pct" if "missing_pct" in miss.columns else "pct_missing"
                try:
                    import plotly.express as px

                    top = miss.sort_values(pct_col, ascending=False).head(12)
                    figm = px.bar(
                        top,
                        x=pct_col,
                        y="column",
                        orientation="h",
                        title="Top columns by missing %",
                        color_discrete_sequence=["#2563eb"],
                    )
                    figm.update_layout(template="plotly_white", height=360, yaxis=dict(autorange="reversed"))
                    st.plotly_chart(figm, use_container_width=True, config=_PLOTLY_CHART_CONFIG)
                except Exception:
                    st.dataframe(miss.head(12), hide_index=True)
                st.markdown(
                    "<div class=\"inference-callout\"><strong>Inference.</strong> "
                    "Sparse fields are dropped or imputed before modeling — stabilizes tree splits and neural batches.</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="story-card"><h4>Business meaning</h4><p style="margin:0;color:#475569">'
                    "Clean inputs reduce silent failures where models learn artifacts instead of fraud.</p></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("`ieee_missing_top20_summary.csv` not found — run Stage 1.")
        st.divider()
        section_title("EDA figures — IEEE & Elliptic")
        st.caption("Stage-01 PNG exports: distributions, Elliptic graph context, and related exploratory plots.")
        rin.render_categorized_figure_gallery(fig_dir, bucket_keys=("ieee_eda", "elliptic_eda"))

    elif step == 4:
        section_title("Hybrid modeling (core)")
        st.markdown(
            "Three calibrated channels are **min–max normalized**, then fused; threshold **τ** is chosen on **validation F1**."
        )
        st.latex(
            r"P_{\mathrm{final}} = w_1 P_{\mathrm{GBDT}} + w_2 P_{\mathrm{DL}} + w_3 P_{\mathrm{AD}}"
        )
        st.markdown(
            f"**Weights (code defaults):** w₁ = **{FUSION_W_GBDT}** (GBDT), w₂ = **{FUSION_W_DNN}** (deep), w₃ = **{FUSION_W_ANOM}** (anomaly)."
        )
        info_panel(
            "Why hybrid?",
            "Combines strengths: trees handle heterogeneous tabular features; deep nets improve ranking; anomaly scores capture tail risk. "
            "Ablation in TABLE IV shows integration beats individual channels on F1.",
        )
        st.divider()
        rin.render_modeling_pipeline_sections(proc)
        st.divider()
        section_title("Model diagnostics figures")
        st.caption("GBDT / SHAP / importance and deep-channel training & anomaly diagnostics (Stage 2–3 PNGs).")
        rin.render_categorized_figure_gallery(fig_dir, bucket_keys=("gbdt", "deep"))

    elif step == 5:
        section_title("Results — tables & figures")
        st.caption("Values come directly from saved CSVs and PNGs under `processed_data/` and `figures/`.")
        st.divider()

        _table_ribbon("tr-ieee", "TABLE I · IEEE-CIS")
        st.markdown("##### Performance comparison (IEEE-CIS)")
        t1 = _safe_df(proc / "report_table_1_ieee_cis.csv")
        _table_shell(t1)
        _narrative_block(
            "Inference",
            "result",
            [
                "Logistic Regression → high precision but fails to detect many frauds (low recall).",
                "Random Forest → detects more fraud but produces many false positives.",
                "XGBoost (GBDT) → highest recall among singles but poor precision.",
                "DNN → strong AUC / ranking ability.",
                "Hybrid model → **best balance** (highest F1) for operational use.",
            ],
        )
        _narrative_block(
            "Business meaning",
            "risk",
            [
                "High recall → fewer missed fraud cases → lower direct revenue loss.",
                "Low precision → higher investigation cost and analyst fatigue.",
                "Hybrid balances both → more practical for real fraud operations.",
            ],
        )

        _table_ribbon("tr-elliptic", "TABLE II · ELLIPTIC")
        st.markdown("##### Performance comparison (Elliptic)")
        t2 = _safe_df(proc / "report_table_2_elliptic.csv")
        _table_shell(t2)
        _narrative_block(
            "Inference",
            "result",
            [
                "Models generally perform strongly on Elliptic under the temporal / graph-aware protocol.",
                "Random Forest often leads **AUC** among tabular baselines in this benchmark.",
                "FraudGT-style MLP can achieve very high **recall** on Elliptic features.",
                "Graph-based methods (GCN row when present) capture transaction relationships.",
            ],
        )
        _narrative_block(
            "Business meaning",
            "risk",
            [
                "Graph structure helps detect coordinated fraud patterns.",
                "Relational signal supports discovery of hidden revenue-leakage **networks**, not only isolated transactions.",
            ],
        )

        _table_ribbon("tr-compare", "TABLE III · MODEL COMPARISON")
        st.markdown("##### Fusion cohort (threshold, ROC-AUC, PR-AUC, …)")
        m3 = _safe_df(proc / "final_hybrid_comparison_metrics.csv")
        if m3 is not None:
            disp = _rename_comparison_models(m3.copy())
            for c in disp.columns:
                if c != "model":
                    disp[c] = pd.to_numeric(disp[c], errors="coerce").round(4)
            paper_order = [
                "Proposed Hybrid Model",
                "Deep Neural Network",
                "Intermediate Hybrid",
                "GBDT (XGBoost stack)",
                "Isolation Forest",
            ]
            if "model" in disp.columns:
                seen = set(disp["model"].astype(str))
                ordered = [m for m in paper_order if m in seen]
                extras = [m for m in disp["model"].astype(str).unique().tolist() if m not in ordered]
                disp = disp.set_index("model").loc[ordered + extras].reset_index()
            _table_shell(disp)
        else:
            st.warning("Run Stage 4 for `final_hybrid_comparison_metrics.csv`.")
        _narrative_block(
            "Inference",
            "result",
            [
                "DNN → often best ROC-AUC and PR-AUC (ranking under imbalance).",
                "Hybrid → best F1-score at the tuned threshold.",
                "GBDT → high recall but extremely low precision at 0.5 on normalized scores.",
                "Intermediate hybrid → can be more conservative than the fully tuned hybrid.",
            ],
        )
        _narrative_block(
            "Business meaning",
            "risk",
            [
                "Ranking ability ≠ real-world alert performance.",
                "Hybrid model is the most practical default for fraud systems balancing leakage vs cost.",
            ],
        )

        _table_ribbon("tr-ablate", "TABLE IV · ABLATION")
        st.markdown("##### Component combinations (IEEE test)")
        t3 = _safe_df(proc / "report_table_3_ablation.csv")
        _table_shell(t3)
        _narrative_block(
            "Inference",
            "result",
            [
                "GBDT → maximizes recall but remains noisy.",
                "DNN → high precision but can miss fraud vs hybrid at τ.",
                "Anomaly detection → weak alone; adds value inside fusion.",
                "Hybrid → combines strengths → best overall performance.",
            ],
        )
        _narrative_block(
            "Business meaning",
            "out",
            [
                "No single model is sufficient for minimizing revenue leakage risk.",
                "Combining models reduces both missed fraud and uncontrolled false-alert rates at tuned operating points.",
            ],
        )

        st.divider()
        st.caption(
            "Per-stage CSV previews, thresholds, and downloads for GBDT, deep+anomaly, fusion scores, and Elliptic graph runs "
            "live under **Modeling** (Stages 2–5)."
        )
        section_title("Evaluation & graph figures")
        st.caption("Fusion ROC/PR, confusion views, Elliptic GCN outputs, and any extra pipeline PNGs.")
        rin.render_categorized_figure_gallery(fig_dir, bucket_keys=("fusion", "graph", "other"))

    elif step == 6:
        section_title("Business impact")
        st.markdown(
            """
- **Missed fraud** = direct **revenue loss** and customer harm.
- **False positives** = **operational cost**, slower investigations, and policy friction.
- **Hybrid model** targets fewer misses **and** a more controlled alert stream vs single channels at tuned τ.
"""
        )
        st.markdown(
            '<p class="quote-accent">Early detection of small anomalies prevents large-scale revenue leakage.</p>',
            unsafe_allow_html=True,
        )
        section_title("Final output — explore risk")
        ds = st.radio("Dataset selector", ["IEEE-CIS (fusion scores)", "Elliptic (Stage 5 experiments)"], horizontal=True)
        if ds.startswith("IEEE"):
            if scores is None:
                st.info("Load `final_hybrid_scores.csv` by running Stage 4.")
            else:
                thr = st.slider("Threshold τ (hybrid)", 0.05, 0.95, float(thr_default), 0.005)
                risky = scores["hybrid_weighted_score"] >= thr
                st.metric("Flagged transactions", f"{int(risky.sum()):,}", delta=f"{100 * risky.mean():.2f}% of cohort")
                col_a, col_b = st.columns(2)
                if risky.any():
                    prec_t = float(scores.loc[risky, "target"].mean())
                    pos = scores["target"] == 1
                    rec_t = float((scores.loc[pos, "hybrid_weighted_score"] >= thr).mean()) if pos.any() else 0.0
                    col_a.metric("Precision @ τ (alerts)", f"{100 * prec_t:.2f}%")
                    col_b.metric("Recall @ τ (fraud caught)", f"{100 * rec_t:.2f}%")
                st.markdown("**Risk scores (flagged, preview)**")
                st.dataframe(scores.loc[risky].head(200), use_container_width=True, hide_index=True)
                st.caption("Estimated leakage uses amount join on the Overview KPIs; full audit: export `final_hybrid_scores.csv`.")
        else:
            eg = _safe_df(proc / "elliptic_graph_experiments.csv")
            _table_shell(eg)
            st.caption("Elliptic uses the Stage 5 temporal / graph protocol — separate from IEEE fusion scores.")

        section_title("Conclusion & future scope")
        highlight_list_plain(
            [
                (1, "Hybrid approach is modular and scalable across tabular and graph benchmarks."),
                (2, "Handles imbalance and evolving fraud by blending supervised and unsupervised signals."),
                (3, "Future: real-time scoring, richer graph models, and stronger explainability for investigators."),
            ]
        )

    elif step == 7:
        section_title("Reports & documents")
        rd = fu.report_docs_dir()
        st.markdown(f"Manuscript / report folder: `{rd}`")
        for pat in ("*.md", "*.tex"):
            for p in sorted(rd.glob(pat))[:25]:
                st.markdown(f"- `{p.name}`")
        st.markdown(
            "**Generated tables (CSV)** — same values as in **Results (Step 5)** and **Modeling** previews; "
            "files live under `processed_data/`."
        )
        for name in (
            "report_table_1_ieee_cis.csv",
            "report_table_2_elliptic.csv",
            "report_table_3_ablation.csv",
            "final_hybrid_comparison_metrics.csv",
        ):
            p = proc / name
            st.caption(f"`{name}` — {'present' if p.is_file() else 'missing'}")

    presentation_footer(
        "AI-Powered Proactive Detection of Hidden Revenue Loss Points using Hybrid Models · Read-only artifacts"
    )


def render_research_dashboard() -> None:
    """Backward-compatible name; prefer wizard + stepper from app shell."""
    render_wizard_step(int(st.session_state.get("wiz", 1)))
