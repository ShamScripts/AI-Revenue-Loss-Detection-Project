"""Project entry: run the Python pipeline (stages 1–5)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fraud_ml pipeline (stages 1–5).")
    parser.add_argument(
        "--project-dir",
        default=".",
        type=Path,
        help="Project root (default: current directory).",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Run only pipeline stage 1–5. Default: run all (including Elliptic graph unless skipped).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving figures under figures/.",
    )
    parser.add_argument(
        "--split",
        choices=["random", "temporal"],
        default="random",
        help="IEEE + fusion: random stratified (default) or temporal split.",
    )
    parser.add_argument(
        "--smote",
        action="store_true",
        help="Apply SMOTE to IEEE training data in stage 2.",
    )
    parser.add_argument(
        "--tune-gbdt",
        action="store_true",
        help="Hyperparameter search for GBDT in stage 2.",
    )
    parser.add_argument(
        "--skip-elliptic-graph",
        action="store_true",
        help="Skip stage 5 when running the full pipeline.",
    )
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()

    from fraud_ml.pipeline.run_all import run_stages

    return run_stages(
        project_root=project_dir,
        stage=args.stage,
        save_plots=not args.no_plots,
        split_mode=args.split,
        use_smote=args.smote,
        tune_gbdt=args.tune_gbdt,
        skip_elliptic_graph=args.skip_elliptic_graph,
    )


if __name__ == "__main__":
    raise SystemExit(main())
