import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import logging
import yaml
import numpy as np
import scipy.sparse as sp
import pandas as pd
import time
import warnings

from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import HashingVectorizer
from src.data.load_embeddings import load_embeddings_h5
from src.data.load_labels import load_train_labels
from src.data.build_matrices import build_label_space, build_Y
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold, cross_val_predict

warnings.filterwarnings("ignore", category=ConvergenceWarning)

#Set up logging
LOGGER = logging.getLogger("model_train_v2")
def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    lvl = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(lvl)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(lvl)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(lvl)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    LOGGER.setLevel(lvl)
    LOGGER.propagate = True
    LOGGER.info("Logging initialized (level=%s)", level.upper())

#Check file pathways
def validate_paths(cfg: dict) -> None:
    missing = []
    def _check(path_str: str | None, label: str, required: bool = True):
        if not path_str:
            if required:
                missing.append(f"{label} (empty)")
            return
        p = Path(path_str)
        if not p.exists():
            missing.append(f"{label}: {p}")
        else:
            LOGGER.info("Path OK: %s -> %s", label, p)

    # Required
    _check(cfg["data"].get("train_ids"), "data.train_ids", required=True)
    _check(cfg["data"].get("train_labels"), "data.train_labels", required=True)
    _check(cfg["data"].get("train_embeddings"), "data.train_embeddings", required=True)

    # Optional
    _check(cfg.get("data", {}).get("train_ipr"), "data.train_ipr", required=False)
    _check(cfg.get("data", {}).get("train_blast"), "data.train_blast", required=False)

    if missing:
        msg = "Missing required input paths:\n" + "\n".join(f"  - {m}" for m in missing)
        LOGGER.error(msg)
        raise FileNotFoundError(msg)

#Read protein ID list 
def read_id_list(path: str) -> list[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

#Import embeddings
def log_embedding_stats(X_emb: np.ndarray) -> None:
    arr = np.asarray(X_emb)
    LOGGER.info("Embeddings: shape=%s dtype=%s", arr.shape, arr.dtype)

    # NaNs / inf
    n_nan = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
    n_inf = int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
    LOGGER.info("Embeddings: n_nan=%d n_inf=%d", n_nan, n_inf)

    # min/max (nan-safe)
    if arr.size > 0 and np.issubdtype(arr.dtype, np.floating):
        try:
            mn = float(np.nanmin(arr))
            mx = float(np.nanmax(arr))
            LOGGER.info("Embeddings: min=%.6g max=%.6g", mn, mx)
        except ValueError:
            LOGGER.warning("Embeddings: could not compute min/max (all-NaN?)")

#Import IPR data
def load_protein2ipr(dat_path: str) -> dict[str, list[str]]:
    p = Path(dat_path)
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(
    p,
    sep="\t",
    header=None,
    usecols=[0, 1],   # only protein + IPR
    dtype=str,
    comment="#")

    if df.shape[1] < 2:
        raise ValueError(f"Expected >=2 columns in {p}, got {df.shape[1]}")

    out: dict[str, list[str]] = {}
    for pid, ipr in zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)):
        pid = pid.strip()
        ipr = ipr.strip()
        if not pid or not ipr:
            continue
        out.setdefault(pid, []).append(ipr)
    return out

#Fix IPR hash matrix
def build_ipr_hash_matrix(
    protein_ids: list[str],
    pid2ipr: dict[str, list[str]],
    n_features: int,
) -> sp.csr_matrix:
    docs = [" ".join(pid2ipr.get(pid, [])) for pid in protein_ids]
    hv = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm=None,
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
    )
    return hv.transform(docs)

#Build full feature matrix
def build_X(cfg: dict, protein_ids: list[str]) -> sp.csr_matrix:

    # ---- embeddings ----
    embedding_key = cfg.get("features", {}).get("embedding_key", None)
    X_emb = load_embeddings_h5(cfg["data"]["train_embeddings"], protein_ids, key=embedding_key)
    X_emb = sp.csr_matrix(np.asarray(X_emb, dtype=np.float32))
    blocks = [X_emb]
    LOGGER.info("Embeddings block: shape=%s nnz=%d", X_emb.shape, X_emb.nnz)

    # ---- InterPro (optional) ----
    ipr_path = cfg.get("data", {}).get("train_ipr", None)
    if ipr_path:
        n_hash = int(cfg.get("features", {}).get("ipr_hash_dim", 2**18))
        pid2ipr = load_protein2ipr(ipr_path)
        X_ipr = build_ipr_hash_matrix(protein_ids, pid2ipr, n_features=n_hash)
        blocks.append(X_ipr)
        LOGGER.info("InterPro block: shape=%s nnz=%d (hash_dim=%d)", X_ipr.shape, X_ipr.nnz, n_hash)
    else:
        LOGGER.info("InterPro block: skipped (data.train_ipr not set)")

    # ---- stack ----
    X = sp.hstack(blocks, format="csr")
    LOGGER.info("Final X: shape=%s nnz=%d (csr=%s)", X.shape, X.nnz, sp.isspmatrix_csr(X))
    return X

#Subset only filter for training data
def filter_terms_present_in_subset(Y, go_terms: list[str], min_pos: int = 1):
    col_counts = np.asarray(Y.sum(axis=0)).ravel()
    keep = col_counts >= float(min_pos)

    if keep.sum() == 0:
        raise RuntimeError(f"No GO terms survive min_pos={min_pos} in this subset.")

    Yf = Y[:, keep] if sp.issparse(Y) else Y[:, keep]
    go_terms_f = [t for t, k in zip(go_terms, keep) if k]
    return Yf, go_terms_f, keep

#Compute fmax
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

# CAFA evaluation
def write_cafa_preds(out_file: Path, prot_ids: list[str], go_terms: list[str], probs: np.ndarray,
                     top_k: int = 500, eps: float = 1e-12) -> None:
    """
    Writes CAFA-format predictions:
      protein_id  GO:xxxxxxx  score
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8") as f:
        for i, pid in enumerate(prot_ids):
            row = probs[i]
            if row.size == 0:
                continue

            # take top_k by score
            idx = np.argsort(row)[::-1][:top_k]
            for j in idx:
                s = float(row[j])
                if s <= eps:
                    break
                f.write(f"{pid} {go_terms[j]} {s:.6g}\n")

#CAFA evaluation
def write_cafa_gt(out_file: Path, prot_ids: list[str], asp_labels: dict[str, list[str]]) -> int:
    """
    Writes CAFA-format ground truth:
      protein_id  GO:xxxxxxx
    Returns number of lines written.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)

    wrote = 0
    with out_file.open("w", encoding="utf-8") as f:
        for pid in prot_ids:
            terms = asp_labels.get(pid, None)
            if not terms:
                continue
            for go in terms:
                f.write(f"{pid} {go}\n")
                wrote += 1
    return wrote

#Main
def main():
    start_time = datetime.now()
    cfg = yaml.safe_load(open("configs/config.yaml"))

    log_cfg = cfg.get("logging", {})
    setup_logging(
        level=str(log_cfg.get("level", "INFO")),
        log_file=log_cfg.get("file", None),
    )

    #Check input paths
    LOGGER.info("Validating input paths...")
    validate_paths(cfg)
    LOGGER.info("Path validation complete.")

    #Next load training IDs
    LOGGER.info("Loading training IDs...")
    all_ids = read_id_list(cfg["data"]["train_ids"])

    if not all_ids:
        raise ValueError("No training IDs loaded (empty list).")

    LOGGER.info("Train proteins: %d", len(all_ids))
    LOGGER.info("First 3 train IDs: %s", all_ids[:3])

    #Subsample training set
    max_n = cfg.get("train", {}).get("max_n", None)
    if max_n is None:
        train_ids = all_ids
    else:
        max_n = int(max_n)
        if max_n > len(all_ids):
            raise ValueError(
                f"train.max_n={max_n} exceeds available IDs ({len(all_ids)})"
            )
        rng = np.random.default_rng(
            int(cfg.get("train", {}).get("subsample_seed", cfg["run"]["seed"]))
        )
        train_ids = rng.choice(all_ids, size=max_n, replace=False).tolist()
    LOGGER.info("Train proteins (after max_n): %d", len(train_ids))

    #Load embeddings for subset only
    LOGGER.info("Loading embeddings...")
    X_emb = load_embeddings_h5(cfg["data"]["train_embeddings"], train_ids)
    log_embedding_stats(X_emb)

    #Load training IDs and labels
    LOGGER.info("Loading labels...")
    labels = load_train_labels(cfg["data"]["train_labels"])

    aspects = cfg["run"]["aspects"]
    LOGGER.info("Aspects: %s", aspects)
    train_set = set(train_ids)
    for aspect in aspects:
        LOGGER.info("=== Aspect: %s ===", aspect)
        asp_labels = labels[aspect]  # dict[protein_id] -> list[GO]

        # overlap check (ON TRAIN SUBSET)
        label_set = set(asp_labels.keys())
        n_overlap = len(train_set & label_set)
        n_missing = len(train_set - label_set)

        LOGGER.info("Label proteins (in file): %d", len(label_set))
        LOGGER.info("Overlap with train_ids: %d / %d", n_overlap, len(train_ids))
        LOGGER.info("train_ids missing labels: %d", n_missing)

        # build Y (ON TRAIN SUBSET)
        go_terms, _ = build_label_space(asp_labels)
        term2idx = {t: i for i, t in enumerate(go_terms)}
        Y = build_Y(train_ids, asp_labels, term2idx)

        # log Y shape + sparsity
        nnz = int(Y.nnz) if hasattr(Y, "nnz") else int(np.count_nonzero(Y))
        total = Y.shape[0] * Y.shape[1]
        density = nnz / total if total else 0.0

        LOGGER.info("GO terms: %d", len(go_terms))
        LOGGER.info("Y.shape = %s  nnz=%d  density=%.6g", Y.shape, nnz, density)

        # average labels per protein
        avg_labels = float(np.asarray(Y.sum(axis=1)).mean())
        LOGGER.info("Avg labels per protein = %.4f", avg_labels)

        if nnz == 0:
            raise RuntimeError(
                f"Y is all zeros for aspect={aspect}. Likely ID mismatch between train_ids and label dict keys."
            )

    #Build full X
    LOGGER.info("Build full X for subset (N=%d)...", len(train_ids))
    X = build_X(cfg, train_ids)

    aspects = cfg["run"]["aspects"]

    #Model training!
    LOGGER.info("Training aspects: %s", aspects)

    min_pos = int(cfg.get("train", {}).get("min_pos_in_subset", 1))  # optional; defaults to 1
    solver = cfg.get("model", {}).get("solver_final", "saga")
    max_iter = int(cfg.get("model", {}).get("max_iter_final", 120))
    seed = int(cfg["run"]["seed"])

    for aspect in aspects:
        LOGGER.info("Aspect=%s ===", aspect)

        asp_labels = labels[aspect]

        # Build label space + Y on THIS subset
        go_terms, _ = build_label_space(asp_labels)
        term2idx = {t: i for i, t in enumerate(go_terms)}
        Y = build_Y(train_ids, asp_labels, term2idx)

        LOGGER.info("Initial Y.shape = %s (terms=%d)", Y.shape, len(go_terms))

        # Drop terms not present in subset (or require >=min_pos)
        Yf, go_terms_f, keep = filter_terms_present_in_subset(Y, go_terms, min_pos=min_pos)

        # Convert Y to dense for sklearn
        Y_train = Yf.toarray() if sp.issparse(Yf) else np.asarray(Yf)
        LOGGER.info("Classes kept (pos>=%d): %d / %d", min_pos, Y_train.shape[1], len(go_terms))

        # Cross-validated predict_proba
        cv_folds = int(cfg["run"].get("cv_folds", 3))
        seed = int(cfg["run"]["seed"])
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

        LOGGER.info("Running CV: folds=%d method=predict_proba", cv_folds)

        # Use SGD OVR (same as before)
        base = SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            max_iter=20,
            tol=1e-3,
            random_state=seed,
            n_jobs=-1,)

        est = OneVsRestClassifier(base, n_jobs=-1)

        t0 = time.time()
        Y_prob = cross_val_predict(
            est,
            X,
            Y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,          # safer; avoids nested parallelism chaos
        )
        LOGGER.info("CV predict_proba complete in %.2f s", time.time() - t0)

        # Fmax over thresholds
        thresholds = np.linspace(0.01, 0.50, 50)
        fmax, t_best = compute_fmax(Y_train, Y_prob, thresholds)

        LOGGER.info("Fmax=%.4f at threshold=%.3f", fmax, t_best)

        # ---- CAFA inputs (for local cafaeval) ----
        cafa_root = Path("results") / "cafa_inputs" / aspect
        pred_dir = cafa_root / "preds"
        pred_file = pred_dir / "pred.tsv"
        gt_file = cafa_root / "gt.tsv"

        pred_dir.mkdir(parents=True, exist_ok=True)
        cafa_root.mkdir(parents=True, exist_ok=True)

        # Clean old preds (avoid mixing runs)
        for old in pred_dir.glob("*"):
            old.unlink()

        write_cafa_preds(pred_file, train_ids, go_terms_f, Y_prob, top_k=500)
        gt_lines = write_cafa_gt(gt_file, train_ids, asp_labels)

        LOGGER.info("[CAFA inputs] preds: %s", pred_file)
        LOGGER.info("[CAFA inputs] gt:    %s (lines=%d)", gt_file, gt_lines)

        # Quick sanity: print first few lines counts/overlap hints
        try:
            with pred_file.open("r", encoding="utf-8") as f:
                head = [next(f).strip() for _ in range(3)]
            LOGGER.info("[CAFA inputs] preds head: %s", head)
        except Exception:
            LOGGER.warning("[CAFA inputs] could not read preds head")

    #Total time
    LOGGER.info("Elapsed time: %s", datetime.now() - start_time)

if __name__ == "__main__":
    main()
