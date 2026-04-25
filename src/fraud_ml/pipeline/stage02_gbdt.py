"""Stage 2: baselines + GBDT; write processed_data/gbdt_preds.csv."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from fraud_ml.config.paths import get_paths
from fraud_ml.pipeline.split_utils import SplitMode, ieee_train_valid_arrays

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
TARGET_COL = "isFraud"


def evaluate_model(name, y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "Model": name,
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }


def run(
    project_root: Path | None = None,
    save_plots: bool = True,
    *,
    split_mode: SplitMode = "random",
    use_smote: bool = False,
    tune_gbdt: bool = False,
    shap_sample_size: int = 2500,
) -> None:
    paths = get_paths(project_root)
    data_root = paths.data_root
    processed_dir = paths.processed_dir
    figures_dir = paths.figures
    data_path = processed_dir / "ieee_train_eda_ready.csv"

    if not data_root.exists():
        raise FileNotFoundError(f"Missing dataset root folder: {data_root}")
    if not data_path.exists():
        raise FileNotFoundError(
            "Missing processed_data/ieee_train_eda_ready.csv. Run stage 1 first."
        )

    print(f"DATA_ROOT: {data_root}")
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(df.shape)

    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in dataset.")

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].select_dtypes(include=[np.number]).copy()
    y = df[TARGET_COL].astype(int)
    dropped_non_numeric = len(feature_cols) - X.shape[1]
    print(f"Using {X.shape[1]} numeric features. Dropped {dropped_non_numeric} non-numeric features.")
    print("Class distribution:")
    print(y.value_counts(normalize=True).rename("ratio"))

    X_train, X_valid, y_train, y_valid, _, _ = ieee_train_valid_arrays(
        df, X, y, split_mode=split_mode, test_size=0.2
    )
    print(f"Split mode: {split_mode} | Train shape:", X_train.shape, "Valid shape:", X_valid.shape)

    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE

            imp_pre = SimpleImputer(strategy="median")
            X_tr_np = imp_pre.fit_transform(X_train)
            sm = SMOTE(random_state=RANDOM_STATE)
            X_tr_np, y_train = sm.fit_resample(X_tr_np, y_train)
            X_train = pd.DataFrame(X_tr_np, columns=X_train.columns, index=None)
            print("After SMOTE — train shape:", X_train.shape, "fraud rate:", float(np.mean(y_train)))
        except Exception as e:
            print(f"[warn] SMOTE skipped: {e}")

    results = []

    log_reg = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    log_reg.fit(X_train, y_train)
    lr_proba = log_reg.predict_proba(X_valid)[:, 1]
    results.append(evaluate_model("Logistic Regression", y_valid, lr_proba))

    decision_tree = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=RANDOM_STATE)),
        ]
    )
    decision_tree.fit(X_train, y_train)
    dt_proba = decision_tree.predict_proba(X_valid)[:, 1]
    results.append(evaluate_model("Decision Tree", y_valid, dt_proba))

    random_forest = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_leaf=10,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )
    random_forest.fit(X_train, y_train)
    rf_proba = random_forest.predict_proba(X_valid)[:, 1]
    results.append(evaluate_model("Random Forest", y_valid, rf_proba))

    X_train_gbdt = X_train.copy()
    X_valid_gbdt = X_valid.copy()
    for col in X_train_gbdt.columns:
        median_val = X_train_gbdt[col].median()
        X_train_gbdt[col] = X_train_gbdt[col].fillna(median_val)
        X_valid_gbdt[col] = X_valid_gbdt[col].fillna(median_val)

    gbdt_name = None
    gbdt_model = None

    try:
        import lightgbm as lgb

        gbdt_name = "LightGBM"
        lgb_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "class_weight": "balanced",
        }
        gbdt_model = lgb.LGBMClassifier(**lgb_params)
        if tune_gbdt:
            param_dist = {
                "n_estimators": [200, 400, 600],
                "learning_rate": [0.03, 0.05, 0.08],
                "num_leaves": [31, 63, 127],
                "subsample": [0.7, 0.85],
            }
            search = RandomizedSearchCV(
                gbdt_model,
                param_distributions=param_dist,
                n_iter=12,
                scoring="roc_auc",
                cv=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=1,
            )
            search.fit(X_train_gbdt, y_train)
            gbdt_model = search.best_estimator_
            print("Best GBDT params (RandomizedSearchCV):", search.best_params_)
        else:
            gbdt_model.fit(X_train_gbdt, y_train)
        gbdt_proba = gbdt_model.predict_proba(X_valid_gbdt)[:, 1]
    except Exception:
        from xgboost import XGBClassifier

        gbdt_name = "XGBoost"
        xgb_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        gbdt_model = XGBClassifier(**xgb_params)
        gbdt_model.fit(X_train_gbdt, y_train)
        gbdt_proba = gbdt_model.predict_proba(X_valid_gbdt)[:, 1]

    results.append(evaluate_model(gbdt_name, y_valid, gbdt_proba))
    print(f"Trained main GBDT model: {gbdt_name}")

    comparison_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    print(comparison_df.to_string())

    gbdt_row = comparison_df[comparison_df["Model"] == gbdt_name]
    if not gbdt_row.empty:
        print(
            f"{gbdt_name} -> ROC-AUC: {gbdt_row['ROC-AUC'].iloc[0]:.4f}, "
            f"Precision: {gbdt_row['Precision'].iloc[0]:.4f}, "
            f"Recall: {gbdt_row['Recall'].iloc[0]:.4f}, "
            f"F1: {gbdt_row['F1-Score'].iloc[0]:.4f}"
        )

    if hasattr(gbdt_model, "feature_importances_"):
        importances = pd.Series(gbdt_model.feature_importances_, index=X_train_gbdt.columns)
        top_n = 25
        top_importances = importances.sort_values(ascending=False).head(top_n)
        plt.figure(figsize=(10, 8))
        top_importances.sort_values().plot(kind="barh")
        plt.title(f"Top {top_n} Feature Importances ({gbdt_name})")
        plt.xlabel("Importance")
        plt.tight_layout()
        if save_plots:
            figures_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(figures_dir / "stage02_gbdt_importance.png", dpi=120, bbox_inches="tight")
        plt.close()
    else:
        print("This GBDT implementation does not expose feature_importances_.")

    try:
        import shap

        shap_n = min(shap_sample_size, len(X_train_gbdt))
        X_shap = X_train_gbdt.sample(n=shap_n, random_state=RANDOM_STATE) if len(X_train_gbdt) > shap_n else X_train_gbdt
        explainer = shap.TreeExplainer(gbdt_model)
        sv = explainer.shap_values(X_shap)
        if isinstance(sv, list):
            sv_plot = sv[1] if len(sv) > 1 else sv[0]
        else:
            sv_plot = sv
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv_plot, X_shap, show=False, max_display=20)
        if save_plots:
            figures_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(figures_dir / "stage02_shap_summary.png", dpi=120, bbox_inches="tight")
        plt.close()
        print("Saved SHAP summary:", figures_dir / "stage02_shap_summary.png")
    except Exception as e:
        print(f"[warn] SHAP plot skipped: {e}")

    cfg: dict[str, Any] = {
        "split_mode": split_mode,
        "use_smote": use_smote,
        "tune_gbdt": tune_gbdt,
        "stage": 2,
    }
    (processed_dir / "stage02_experiment_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    X_all_gbdt = X.copy()
    for col in X_all_gbdt.columns:
        median_val = X_train[col].median()
        X_all_gbdt[col] = X_all_gbdt[col].fillna(median_val)
    gbdt_proba_all = gbdt_model.predict_proba(X_all_gbdt)[:, 1]

    preds_out = pd.DataFrame({"gbdt_pred_proba": gbdt_proba_all, "is_valid_split_row": 0})
    preds_out.loc[X_valid.index, "is_valid_split_row"] = 1
    if "TransactionID" in df.columns:
        preds_out.insert(0, "TransactionID", df["TransactionID"].values)

    preds_path = processed_dir / "gbdt_preds.csv"
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds_out.to_csv(preds_path, index=False)
    print(f"Saved dataset-wide GBDT probabilities to: {preds_path}")
    print(preds_out.head())


def main() -> None:
    run()


if __name__ == "__main__":
    main()
