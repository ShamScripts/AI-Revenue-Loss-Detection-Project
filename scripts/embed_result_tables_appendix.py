"""One-off / repeatable: rebuild Appendix section in docs/STAGE_1_TO_5_REPORTS_TABLES_GRAPHS_INFERENCES.md.

Uses ~~~~ fences (no PowerShell backtick issues). Run from repo root:
  python scripts/embed_result_tables_appendix.py
"""

from __future__ import annotations

from collections import deque
from pathlib import Path


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    p_doc = base / "docs" / "STAGE_1_TO_5_REPORTS_TABLES_GRAPHS_INFERENCES.md"
    p_proc = base / "processed_data"
    text = p_doc.read_text(encoding="utf-8")
    marker_cross = "\n\n## Cross-stage interpretation checklist"
    if marker_cross not in text:
        raise SystemExit("Could not find Cross-stage heading")
    head, tail = text.split(marker_cross, 1)
    if "\n\n## Appendix:" in head:
        head = head.split("\n\n## Appendix:")[0].rstrip()
    head = head.rstrip()
    if not head.endswith("---"):
        head = head + "\n\n---"

    def rf(name: str) -> str:
        return (p_proc / name).read_text(encoding="utf-8").rstrip("\n")

    scores_path = p_proc / "final_hybrid_scores.csv"
    with scores_path.open("r", encoding="utf-8") as f:
        hdr = f.readline().rstrip("\n")
        first5 = [f.readline().rstrip("\n") for _ in range(5)]
    with scores_path.open("r", encoding="utf-8") as f:
        last5 = [ln.rstrip("\n") for ln in deque(f, maxlen=5)]

    thr = (p_proc / "final_hybrid_threshold.txt").read_text(encoding="utf-8").rstrip()

    parts: list[str] = []
    parts.append("\n\n## Appendix: Full result tables (verbatim `processed_data/`)\n\n")
    parts.append(
        "Exact CSV/JSON copies from the current local `processed_data/` folder. "
        "Re-run the pipeline to refresh. "
        "`final_hybrid_scores.csv` is summarized (590,540 rows) at the end; all other tables are complete.\n\n"
    )

    def block(heading: str, fence: str, body: str) -> None:
        parts.append(f"### {heading}\n\n")
        parts.append(f"~~~~{fence}\n")
        parts.append(body)
        parts.append("\n~~~~\n\n")

    block("`report_table_1_ieee_cis.csv`", "csv", rf("report_table_1_ieee_cis.csv"))
    block("`report_table_2_elliptic.csv`", "csv", rf("report_table_2_elliptic.csv"))
    block("`report_table_3_ablation.csv`", "csv", rf("report_table_3_ablation.csv"))
    block("`final_hybrid_comparison_metrics.csv`", "csv", rf("final_hybrid_comparison_metrics.csv"))
    block("`final_hybrid_threshold.txt`", "text", thr)
    block("`stage03_ieee_dnn_baselines.csv`", "csv", rf("stage03_ieee_dnn_baselines.csv"))
    block("`elliptic_graph_experiments.csv`", "csv", rf("elliptic_graph_experiments.csv"))
    block("`ieee_missing_top20_summary.csv`", "csv", rf("ieee_missing_top20_summary.csv"))
    block("`preprocessing_config.json`", "json", rf("preprocessing_config.json"))
    block("`fusion_test_record_ids.csv` (all rows)", "csv", rf("fusion_test_record_ids.csv"))
    block("`stage02_experiment_config.json`", "json", rf("stage02_experiment_config.json"))
    block("`stage03_experiment_config.json`", "json", rf("stage03_experiment_config.json"))
    block("`stage04_experiment_config.json`", "json", rf("stage04_experiment_config.json"))
    block("`stage05_experiment_config.json`", "json", rf("stage05_experiment_config.json"))

    parts.append("### `final_hybrid_scores.csv` (per-record; not fully inlined)\n\n")
    parts.append("| Property | Value |\n|----------|-------|\n")
    parts.append("| Path | `processed_data/final_hybrid_scores.csv` |\n")
    parts.append("| Data rows | 590,540 |\n")
    parts.append("| Columns | 13 |\n\n")
    parts.append("**Header + first 5 data rows:**\n\n")
    parts.append("~~~~csv\n")
    parts.append(hdr + "\n" + "\n".join(first5))
    parts.append("\n~~~~\n\n")
    parts.append("**Last 5 data rows:**\n\n")
    parts.append("~~~~csv\n")
    parts.append("\n".join(last5))
    parts.append(
        "\n~~~~\n\n"
        "*Full file (~112 MB): open in DuckDB, pandas, or a spreadsheet engine; "
        "inlining all rows would make this document impractical to edit.*\n"
    )

    appendix = "".join(parts)
    out = head + appendix + marker_cross + tail
    p_doc.write_text(out, encoding="utf-8")
    print("Wrote", p_doc, "chars", len(out))


if __name__ == "__main__":
    main()
