import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import yaml
import numpy as np
import scipy.sparse as sp
import joblib

from src.data.load_embeddings import load_embeddings_h5
from src.data.load_labels import load_train_labels  # needed for BLAST transfer labels

# --- InterPro hashing (match training conventions) ---
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer


def read_id_list(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_protein2ipr(dat_path: str) -> dict[str, list[str]]:
    df = pd.read_csv(dat_path, sep="\t", header=None, usecols=[0, 1], dtype=str, comment="#")
    out: dict[str, list[str]] = {}
    for pid, ipr in zip(df.iloc[:, 0], df.iloc[:, 1]):
        pid = str(pid).strip()
        ipr = str(ipr).strip()
        if pid and ipr:
            out.setdefault(pid, []).append(ipr)
    return out


def build_ipr_hash_matrix(protein_ids: list[str], pid2ipr: dict[str, list[str]], n_features: int) -> sp.csr_matrix:
    docs = [" ".join(sorted(set(pid2ipr.get(pid, [])))) for pid in protein_ids]
    hv = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm=None,
        lowercase=False,
        tokenizer=str.split,
        token_pattern=None,
    )
    return hv.transform(docs).tocsr()


def write_cafa_submission(
    out_path: Path,
    prot_ids: list[str],
    go_terms: list[str],
    probs: np.ndarray,
    max_terms_per_protein: int = 1500,
    eps: float = 1e-12,
):
    """
    Writes lines: ProteinID GO:XXXXXXX score
    Enforces <= max_terms_per_protein per protein and score > 0.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, pid in enumerate(prot_ids):
            row = probs[i]
            if row.size == 0:
                continue

            k = min(max_terms_per_protein, row.size)
            idx = np.argsort(row)[::-1][:k]

            for j in idx:
                s = float(row[j])
                if s <= eps:
                    break
                f.write(f"{pid} {go_terms[j]} {s:.6g}\n")


# -------------------------
# BLAST transfer helpers
# -------------------------
def load_blast_hits(blast_path: str) -> pd.DataFrame:
    """
    Expects a TSV with a header and at least: query, target, bits (or bitscore).
    Your file has: query target bits evalue score
    """
    df = pd.read_csv(blast_path, sep="\t", header=0)
    df.columns = [c.lower() for c in df.columns]

    if "bitscore" in df.columns and "bits" not in df.columns:
        df = df.rename(columns={"bitscore": "bits"})

    for col in ["query", "target", "bits"]:
        if col not in df.columns:
            raise ValueError(f"BLAST file missing required column '{col}'. Found: {df.columns.tolist()}")

    # Keep only what we need
    df = df[["query", "target", "bits"]].copy()
    df["query"] = df["query"].astype(str)
    df["target"] = df["target"].astype(str)
    df["bits"] = df["bits"].astype(float)
    return df


def blast_transfer_probs(
    blast_df: pd.DataFrame,
    query_ids: list[str],
    asp_labels: dict[str, list[str]],
    go_terms: list[str],
    top_k: int = 50,
) -> np.ndarray:
    """
    Build BLAST-transfer probability matrix aligned to go_terms.
    Weighted vote: w_i = bits_i / max(bits for query), normalize by sum(w).
    """
    term2idx = {t: i for i, t in enumerate(go_terms)}
    q2row = {q: i for i, q in enumerate(query_ids)}

    P = np.zeros((len(query_ids), len(go_terms)), dtype=np.float32)

    for q, g in blast_df.groupby("query"):
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


def main():
    cfg = yaml.safe_load(open("configs/config.yaml", "r", encoding="utf-8"))

    # --- validated BLAST blending weights (lambda * v4 + (1-lambda) * blast) ---
    LAMBDA_BY_ASPECT = {
        "molecular_function": 0.30,
        "biological_process": 0.40,
        "cellular_component": 0.50,
    }

    test_ids = read_id_list(cfg["data"]["test_ids"])

    # Embeddings
    X_emb = load_embeddings_h5(cfg["data"]["test_embeddings"], test_ids)
    X_emb = sp.csr_matrix(np.asarray(X_emb, dtype=np.float32))

    # InterPro features
    pid2ipr = load_protein2ipr(cfg["data"]["test_ipr"])
    ipr_hash_dim = int(cfg.get("features", {}).get("ipr_hash_dim", 2**18))
    X_ipr = build_ipr_hash_matrix(test_ids, pid2ipr, n_features=ipr_hash_dim)

    # Load trained model bundles (contain alpha for emb+ipr ensemble)
    aspects = cfg["run"]["aspects"]
    bundles = {}
    for asp in aspects:
        path = Path("models") / f"model_{asp}.pkl"
        bundles[asp] = joblib.load(path)

    # Load train labels for BLAST transfer (targets are train proteins)
    labels = load_train_labels(cfg["data"]["train_labels"])

    # Load BLAST hits for test->train
    blast_path = cfg["data"].get(
        "blast_test_results",
        "data/raw/biological_data_pfp/test/blast_test_results.tsv",
    )
    blast_df = load_blast_hits(blast_path)

    # Output dir (new one, so you don’t overwrite old v4-only outputs)
    out_dir = Path("outputs") / "submission_v4_blast"
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_lines = []

    for asp in aspects:
        b = bundles[asp]
        model_emb = b["model_emb"]
        model_ipr = b["model_ipr"]
        alpha = float(b["alpha"])
        go_terms = b["go_terms"]

        # --- v4 prediction (trained emb+ipr ensemble) ---
        P_emb = model_emb.predict_proba(X_emb)
        P_ipr = model_ipr.predict_proba(X_ipr)
        P_v4 = alpha * P_emb + (1.0 - alpha) * P_ipr

        # --- BLAST transfer, aligned to same go_terms ---
        asp_labels = labels[asp]
        P_blast = blast_transfer_probs(
            blast_df=blast_df,
            query_ids=test_ids,
            asp_labels=asp_labels,
            go_terms=go_terms,
            top_k=50,
        )

        # --- Final blend using validated lambda ---
        lam = float(LAMBDA_BY_ASPECT[asp])
        P_final = lam * P_v4 + (1.0 - lam) * P_blast

        print(
            f"[{asp}] alpha(v4)={alpha:.2f} lambda(v4+blast)={lam:.2f} "
            f"shapes v4={P_v4.shape} blast={P_blast.shape}"
        )

        # Write per-aspect file
        out_file = out_dir / f"pred_{asp}.tsv"
        write_cafa_submission(out_file, test_ids, go_terms, P_final, max_terms_per_protein=1500)
        print(f"Wrote {out_file}")

        # Also build combined list for final <=1500 total terms/protein
        for i, pid in enumerate(test_ids):
            row = P_final[i]
            k = min(1500, row.size)
            idx = np.argsort(row)[::-1][:k]
            for j in idx:
                s = float(row[j])
                if s <= 1e-12:
                    break
                combined_lines.append((pid, go_terms[j], s))

    # Enforce <=1500 TOTAL terms per protein across all aspects
    final_path = out_dir / "submission_final.tsv"
    final_path.parent.mkdir(parents=True, exist_ok=True)

    from collections import defaultdict
    per_pid = defaultdict(list)
    for pid, go, s in combined_lines:
        per_pid[pid].append((go, s))

    with final_path.open("w", encoding="utf-8") as f:
        for pid in test_ids:
            items = per_pid.get(pid, [])
            items.sort(key=lambda x: x[1], reverse=True)
            for go, s in items[:1500]:
                if s <= 1e-12:
                    break
                f.write(f"{pid} {go} {float(s):.6g}\n")

    print(f"Wrote FINAL submission: {final_path}")


if __name__ == "__main__":
    main()

