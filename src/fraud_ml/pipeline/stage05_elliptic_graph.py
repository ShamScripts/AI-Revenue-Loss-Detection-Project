"""Stage 5: Elliptic graph — 2-layer GCN + FraudGT-style tabular MLP; time-based split."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy import sparse as sp

from fraud_ml.config.paths import get_paths

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


def _prf1_auc(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thr).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def _normalize_adjacency(adj: sp.csr_matrix) -> sp.csr_matrix:
    """Symmetrically normalize adjacency (GCN-style)."""
    adj = adj + sp.eye(adj.shape[0], format="csr")
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    return d_mat.dot(adj).dot(d_mat).astype(np.float32)


def run(project_root: Path | None = None, save_plots: bool = True) -> None:
    paths = get_paths(project_root)
    processed_dir = paths.processed_dir
    figures_dir = paths.figures
    elliptic_path = processed_dir / "elliptic_transactions_cleaned.csv"
    edges_path = paths.elliptic_dir / "elliptic_txs_edgelist.csv"

    if not elliptic_path.is_file():
        print("Stage 5 skipped: elliptic_transactions_cleaned.csv not found (run stage 1).")
        return
    if not edges_path.is_file():
        print("Stage 5 skipped: elliptic edgelist not found.")
        return

    df = pd.read_csv(elliptic_path)
    sub = df[df["class_label"].isin(["licit", "illicit"])].copy()
    if len(sub) < 500:
        print("Stage 5 skipped: too few labeled Elliptic rows.")
        return

    sub = sub.sort_values("time_step").reset_index(drop=True)
    sub["y"] = (sub["class_label"] == "illicit").astype(np.int64)
    feat_cols = [c for c in sub.columns if c.startswith("feature_")]
    feat_cols += [c for c in ("time_step", "in_degree", "out_degree", "total_degree", "degree_in_out_ratio") if c in sub.columns]
    feat_cols = [c for c in feat_cols if c in sub.columns]
    X = sub[feat_cols].select_dtypes(include=[np.number]).fillna(0.0).values.astype(np.float32)
    y = sub["y"].values
    tx_ids = sub["txId"].values.astype(np.int64)

    # Time-ordered split: last 20% of rows = test (no shuffle)
    n = len(sub)
    cut = int(n * 0.8)
    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[:cut] = True
    test_mask[cut:] = True

    id_to_idx = {int(t): i for i, t in enumerate(tx_ids)}
    edges = pd.read_csv(edges_path)
    rows_e, cols_e, data = [], [], []
    for _, r in edges.iterrows():
        a, b = int(r["txId1"]), int(r["txId2"])
        if a in id_to_idx and b in id_to_idx:
            ia, ib = id_to_idx[a], id_to_idx[b]
            rows_e += [ia, ib]
            cols_e += [ib, ia]
            data += [1.0, 1.0]
    adj = sp.csr_matrix((data, (rows_e, cols_e)), shape=(n, n), dtype=np.float32)
    adj_norm = _normalize_adjacency(adj)

    rows: list[dict] = []

    # --- FraudGT-style: deep MLP on tabular features only (no graph) ---
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    fraud_gt = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        alpha=1e-4,
        batch_size=min(512, len(X_train)),
        learning_rate_init=1e-3,
        max_iter=80,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    )
    fraud_gt.fit(X_train, y_train)
    p_fg = fraud_gt.predict_proba(X_test)[:, 1]
    rows.append(
        {
            "model": "FraudGT-style MLP (tabular encoder)",
            "split": "temporal (time_step order, last 20% test)",
            **_prf1_auc(y_test, p_fg),
        }
    )

    lr = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500, random_state=RANDOM_STATE))]
    )
    lr.fit(X_train, y_train)
    p_lr = lr.predict_proba(X_test)[:, 1]
    rows.append(
        {
            "model": "Logistic Regression (tabular)",
            "split": "temporal",
            **_prf1_auc(y_test, p_lr),
        }
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    p_rf = rf.predict_proba(X_test)[:, 1]
    rows.append(
        {
            "model": "Random Forest (tabular)",
            "split": "temporal",
            **_prf1_auc(y_test, p_rf),
        }
    )

    gcn_auc = None
    try:
        import torch
        import torch.nn as nn

        def sparse_to_torch(adj_csr: sp.csr_matrix):
            coo = adj_csr.tocoo()
            idx = torch.LongTensor(np.vstack([coo.row, coo.col]))
            val = torch.FloatTensor(coo.data)
            return torch.sparse_coo_tensor(idx, val, coo.shape).coalesce()

        class GCN(nn.Module):
            def __init__(self, nfeat: int, nhid: int, dropout: float = 0.3):
                super().__init__()
                self.lin1 = nn.Linear(nfeat, nhid)
                self.lin2 = nn.Linear(nhid, 1)
                self.dropout = dropout

            def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
                h = torch.relu(self.lin1(torch.sparse.mm(adj, x)))
                h = nn.functional.dropout(h, p=self.dropout, training=self.training)
                return self.lin2(torch.sparse.mm(adj, h)).squeeze(-1)

        device = torch.device("cpu")
        nfeat = X.shape[1]
        x_t = torch.FloatTensor(X).to(device)
        adj_t = sparse_to_torch(adj_norm).to(device)
        y_t = torch.FloatTensor(y).to(device)
        train_m = torch.BoolTensor(train_mask).to(device)
        test_m = torch.BoolTensor(test_mask).to(device)

        model = GCN(nfeat, nhid=64, dropout=0.3).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        pos = float(y_t[train_m].sum().clamp(min=1.0))
        neg = float((1.0 - y_t[train_m]).sum().clamp(min=1.0))
        pos_weight = torch.tensor([neg / pos], device=device)

        model.train()
        for _ in range(150):
            opt.zero_grad()
            logits = model(x_t, adj_t)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[train_m], y_t[train_m], pos_weight=pos_weight
            )
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(x_t, adj_t)
            prob = torch.sigmoid(logits).cpu().numpy()
        m_gcn = _prf1_auc(y[test_mask], prob[test_mask])
        rows.insert(
            0,
            {
                "model": "GNN (2-layer GCN, Elliptic graph)",
                "split": "temporal (transductive; loss on train nodes only)",
                **m_gcn,
            },
        )

        if save_plots:
            figures_dir.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(6, 4))
            plt.hist(prob[test_mask][y[test_mask] == 0], bins=40, alpha=0.6, label="licit", density=True)
            plt.hist(prob[test_mask][y[test_mask] == 1], bins=40, alpha=0.6, label="illicit", density=True)
            plt.xlabel("GCN fraud probability")
            plt.title("Elliptic GCN — test score distribution")
            plt.legend()
            plt.tight_layout()
            plt.savefig(figures_dir / "stage05_elliptic_gcn_score_hist.png", dpi=120, bbox_inches="tight")
            plt.close()

    except Exception as e:
        print(f"[warn] PyTorch GCN skipped ({e}). Install torch for Elliptic GCN: pip install torch")
        rows.insert(
            0,
            {
                "model": "GNN (2-layer GCN, Elliptic graph)",
                "split": "skipped (install torch: pip install torch)",
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "roc_auc": np.nan,
                "pr_auc": np.nan,
            },
        )

    out_df = pd.DataFrame(rows)
    out_path = processed_dir / "elliptic_graph_experiments.csv"
    out_df.to_csv(out_path, index=False)
    meta = {
        "stage": 5,
        "n_nodes": int(n),
        "n_edges_used": int(len(rows_e) // 2),
        "note": "GCN uses full graph (transductive); test metrics on time-held-out nodes.",
    }
    (processed_dir / "stage05_experiment_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved:", out_path)
    print(out_df.to_string())


def main() -> None:
    run()


if __name__ == "__main__":
    main()
