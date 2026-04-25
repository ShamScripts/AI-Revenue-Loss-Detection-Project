"""Run pipeline stages 1–5 sequentially or a single stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from fraud_ml.pipeline.split_utils import SplitMode


def run_stages(
    project_root: Path | None = None,
    stage: int | None = None,
    save_plots: bool = True,
    *,
    split_mode: SplitMode = "random",
    use_smote: bool = False,
    tune_gbdt: bool = False,
    skip_elliptic_graph: bool = False,
) -> int:
    stages = [stage] if stage else [1, 2, 3, 4] + ([] if skip_elliptic_graph else [5])
    for s in stages:
        if s == 1:
            from fraud_ml.pipeline.stage01_data import run as run1

            run1(project_root=project_root, save_plots=save_plots)
        elif s == 2:
            from fraud_ml.pipeline.stage02_gbdt import run as run2

            run2(
                project_root=project_root,
                save_plots=save_plots,
                split_mode=split_mode,
                use_smote=use_smote,
                tune_gbdt=tune_gbdt,
            )
        elif s == 3:
            from fraud_ml.pipeline.stage03_deep_anomaly import run as run3

            run3(project_root=project_root, save_plots=save_plots, split_mode=split_mode)
        elif s == 4:
            from fraud_ml.pipeline.stage04_fusion import run as run4

            run4(project_root=project_root, save_plots=save_plots, split_mode=split_mode)
        elif s == 5:
            from fraud_ml.pipeline.stage05_elliptic_graph import run as run5

            run5(project_root=project_root, save_plots=save_plots)
        else:
            raise ValueError(f"Unknown stage: {s}")

    print("\n[DONE] Pipeline stage(s) completed. Artifacts under processed_data/")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fraud_ml pipeline stages.")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help="Project root (folder with DATASET_ieee-cis-elliptic). Default: auto-detect / cwd.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Run only this stage (1–5). Default: run 1–4 and Elliptic graph (5).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving figures under figures/ (faster, headless).",
    )
    parser.add_argument(
        "--split",
        choices=["random", "temporal"],
        default="random",
        help="IEEE + fusion split: random stratified (default) or temporal (TransactionDT / time_step).",
    )
    parser.add_argument(
        "--smote",
        action="store_true",
        help="Apply SMOTE to IEEE training rows in stage 2 (after median imputation).",
    )
    parser.add_argument(
        "--tune-gbdt",
        action="store_true",
        help="Run RandomizedSearchCV for LightGBM in stage 2 (slower).",
    )
    parser.add_argument(
        "--skip-elliptic-graph",
        action="store_true",
        help="When running all stages, skip stage 5 (Elliptic GCN + baselines).",
    )
    args = parser.parse_args()
    root = args.project_dir.resolve() if args.project_dir else None
    save_plots = not args.no_plots
    return run_stages(
        project_root=root,
        stage=args.stage,
        save_plots=save_plots,
        split_mode=args.split,
        use_smote=args.smote,
        tune_gbdt=args.tune_gbdt,
        skip_elliptic_graph=args.skip_elliptic_graph,
    )


if __name__ == "__main__":
    raise SystemExit(main())
