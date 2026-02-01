import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
import yaml
import json

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


def main():
    cfg = yaml.safe_load(open("configs/config.yaml", "r", encoding="utf-8"))

    train_ids = load_ids("data/splits/train_ids_no_val.txt")
    val_ids = load_ids("data/splits/val_ids.txt")

    # Load labels for all train proteins (from original train_set.tsv)
    labels = load_train_labels(cfg["data"]["train_labels"])
    aspects = cfg["run"]["aspects"]

    # Load BLAST hits val->train
    blast_path = Path("data/splits/blast_val_results.tsv")
    df = pd.read_csv(blast_path, sep="\t", header=None, names=["query", "target", "bits", "evalue"])
    # Keep only targets that are in train_ids (should be, but safe)
    train_set = set(train_ids)
    df = df[df["target"].isin(train_set)]

    thresholds = np.linspace(0.001, 0.5, 200)

    results = {}
    for aspect in aspects:
        asp_labels = labels[aspect]

        # Restrict label space to TRAIN (no val leakage)
        asp_labels_train = {pid: asp_labels.get(pid, []) for pid in train_ids}

        go_terms, _ = build_label_space(asp_labels_train)
        term2idx = {t: i for i, t in enumerate(go_terms)}

        # Ground truth for VAL
        Y_val = build_Y(val_ids, asp_labels, term2idx)
        Y_val = Y_val.toarray() if sp.issparse(Y_val) else np.asarray(Y_val)

        # Build BLAST probability matrix for VAL
        n_val = len(val_ids)
        n_terms = len(go_terms)
        pid2row = {pid: i for i, pid in enumerate(val_ids)}

        P = np.zeros((n_val, n_terms), dtype=np.float32)

        # For each query, normalize weights by max bitscore (top hits)
        for q, group in df.groupby("query"):
            if q not in pid2row:
                continue
            hits = group.sort_values("bits", ascending=False).head(50)

            bits = hits["bits"].to_numpy(dtype=np.float32)
            if bits.size == 0:
                continue
            w = bits / bits.max()
            w_sum = float(w.sum())
            if w_sum <= 0:
                continue

            row = pid2row[q]
            for (target, weight) in zip(hits["target"].tolist(), w.tolist()):
                for go in asp_labels.get(target, []):
                    j = term2idx.get(go)
                    if j is not None:
                        P[row, j] += weight

            P[row, :] /= w_sum

        # Evaluate
        f, t = compute_fmax(Y_val, P, thresholds)
        results[aspect] = {"fmax_blast": float(f), "t_blast": float(t), "n_terms": int(n_terms)}

        print(f"[{aspect}] BLAST transfer Fmax={f:.4f} @ t={t:.3f} (terms={n_terms})")

    out = Path("metrics") / "metrics_blast_val.json"
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(results, indent=2))
    print("Saved:", out)


if __name__ == "__main__":
    main()
