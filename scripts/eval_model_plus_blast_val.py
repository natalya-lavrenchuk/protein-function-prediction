import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
import yaml
import json
import time

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier

from src.data.load_embeddings import load_embeddings_h5
from src.data.load_labels import load_train_labels
from src.data.build_matrices import build_label_space, build_Y


def compute_fmax(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray):
    y_true = (y_true > 0).astype(np.int8)

    best_f, best_t = 0.0, float(thresholds[0])
    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int8)

        tp = int((y_pred & y_true).sum())
        fp = int((y_pred & (1 - y_true)).sum())
        fn = int(((1 - y_pred) & y_true).sum())

        if tp == 0:
            continue

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            continue

        f = 2 * precision * recall / (precision + recall)
        if f > best_f:
            best_f, best_t = f, float(t)

    return best_f, best_t


def load_ids(path: str) -> list[str]:
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


def blast_transfer_probs(
    blast_tsv: Path,
    query_ids: list[str],
    train_ids_set: set[str],
    asp_labels: dict[str, list[str]],
    term2idx: dict[str, int],
    top_k: int = 50,
) -> np.ndarray:
    """
    Build P_blast for query_ids using val->train BLAST hits.
    Weighted vote by bitscore normalized by max bitscore per query.
    """
    df = pd.read_csv(blast_tsv, sep="\t", header=None, names=["query", "target", "bits", "evalue"])
    df = df[df["target"].isin(train_ids_set)]

    q2row = {q: i for i, q in enumerate(query_ids)}
    n_q = len(query_ids)
    n_terms = len(term2idx)
    P = np.zeros((n_q, n_terms), dtype=np.float32)

    for q, g in df.groupby("query"):
        if q not in q2row:
            continue
        hits = g.sort_values("bits", ascending=False).head(top_k)
        bits = hits["bits"].to_numpy(dtype=np.float32)
        if bits.size == 0:
            continue

        w = bits / bits.max()
        w_sum = float(w.sum())
        if w_sum <= 0:
            continue

        row = q2row[q]
        for target, weight in zip(hits["target"].tolist(), w.tolist()):
            for go in asp_labels.get(target, []):
                j = term2idx.get(go)
                if j is not None:
                    P[row, j] += weight

        P[row, :] /= w_sum

    return P


def make_estimator(seed: int) -> OneVsRestClassifier:
    base = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        max_iter=50,
        tol=1e-3,
        random_state=seed,
        n_jobs=-1,
    )
    return OneVsRestClassifier(base, n_jobs=-1)


def main():
    cfg = yaml.safe_load(open("configs/config.yaml", "r", encoding="utf-8"))

    train_ids = load_ids("data/splits/train_ids_no_val.txt")
    val_ids = load_ids("data/splits/val_ids.txt")
    train_set = set(train_ids)

    print("train_no_val:", len(train_ids), "val:", len(val_ids))

    seed = int(cfg["run"]["seed"])
    aspects = cfg["run"]["aspects"]

    # Load embeddings for train_no_val and val from the SAME train_embeddings.h5
    t0 = time.time()
    X_train = load_embeddings_h5(cfg["data"]["train_embeddings"], train_ids)
    X_val = load_embeddings_h5(cfg["data"]["train_embeddings"], val_ids)
    X_train = sp.csr_matrix(np.asarray(X_train, dtype=np.float32))
    X_val = sp.csr_matrix(np.asarray(X_val, dtype=np.float32))
    print("Loaded embeddings in %.2fs" % (time.time() - t0))
    print("X_train:", X_train.shape, "X_val:", X_val.shape)

    labels = load_train_labels(cfg["data"]["train_labels"])

    thresholds = np.linspace(0.001, 0.5, 200)
    lambdas = np.linspace(0.0, 1.0, 11)  # 0..1 step 0.1

    blast_path = Path("data/splits/blast_val_results.tsv")

    out = {}

    for aspect in aspects:
        print("\n=== Aspect:", aspect, "===")
        asp_labels = labels[aspect]

        # Build label space based ONLY on train_no_val to avoid leakage
        asp_labels_train = {pid: asp_labels.get(pid, []) for pid in train_ids}
        go_terms, _ = build_label_space(asp_labels_train)
        term2idx = {t: i for i, t in enumerate(go_terms)}

        # Ground truth for val (as a multi-hot matrix)
        Y_val = build_Y(val_ids, asp_labels, term2idx)
        Y_val = Y_val.toarray() if sp.issparse(Y_val) else np.asarray(Y_val)

        # Training labels for train_no_val
        Y_train = build_Y(train_ids, asp_labels, term2idx)
        Y_train = Y_train.toarray() if sp.issparse(Y_train) else np.asarray(Y_train)

        # Filter out unlabeled rows (optional but helps)
        train_mask = (Y_train.sum(axis=1) > 0)
        val_mask = (Y_val.sum(axis=1) > 0)

        X_tr = X_train[train_mask]
        Y_tr = Y_train[train_mask]
        X_va = X_val[val_mask]
        Y_va = Y_val[val_mask]

        print("Labeled train rows:", X_tr.shape[0], " / ", X_train.shape[0])
        print("Labeled val rows:", X_va.shape[0], " / ", X_val.shape[0])
        print("Terms:", len(go_terms))

        # 1) Fit embeddings-only model
        est = make_estimator(seed)
        t0 = time.time()
        est.fit(X_tr, Y_tr)
        print("Fit embeddings model in %.2fs" % (time.time() - t0))

        P_model = est.predict_proba(X_va)

        f_m, t_m = compute_fmax(Y_va, P_model, thresholds)
        print(f"Embeddings-only: Fmax={f_m:.4f} @ t={t_m:.3f}")

        # 2) BLAST transfer on val (same filtered val ids)
        val_ids_lab = [pid for pid, ok in zip(val_ids, val_mask) if ok]
        P_blast = blast_transfer_probs(
            blast_path,
            val_ids_lab,
            train_set,
            asp_labels,
            term2idx,
            top_k=50,
        )

        f_b, t_b = compute_fmax(Y_va, P_blast, thresholds)
        print(f"BLAST-only:      Fmax={f_b:.4f} @ t={t_b:.3f}")

        # 3) Blend
        best = (-1.0, None, None, None)  # f, lambda, threshold, which
        for lam in lambdas:
            P_mix = lam * P_model + (1.0 - lam) * P_blast
            f, t = compute_fmax(Y_va, P_mix, thresholds)
            if f > best[0]:
                best = (float(f), float(lam), float(t), "model+blast")

        print(f"BEST blend:      Fmax={best[0]:.4f} @ t={best[2]:.3f} (lambda={best[1]:.2f})")

        out[aspect] = {
            "fmax_model": float(f_m),
            "t_model": float(t_m),
            "fmax_blast": float(f_b),
            "t_blast": float(t_b),
            "fmax_best_blend": float(best[0]),
            "t_best_blend": float(best[2]),
            "lambda_best": float(best[1]),
            "n_terms": int(len(go_terms)),
            "n_train_labeled": int(X_tr.shape[0]),
            "n_val_labeled": int(X_va.shape[0]),
        }

    out_path = Path("metrics") / "metrics_model_plus_blast_val.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
