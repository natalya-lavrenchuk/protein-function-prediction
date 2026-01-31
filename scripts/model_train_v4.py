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
import json
import joblib

from datetime import datetime
from sklearn.feature_extraction.text import HashingVectorizer
from src.data.load_embeddings import load_embeddings_h5
from src.data.load_labels import load_train_labels
from src.data.build_matrices import build_label_space, build_Y
from sklearn.multiclass import OneVsRestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

LOGGER = logging.getLogger("model_train_v4")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


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


# Set up paths
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

    _check(cfg["data"].get("train_ids"), "data.train_ids", required=True)
    _check(cfg["data"].get("train_labels"), "data.train_labels", required=True)
    _check(cfg["data"].get("train_embeddings"), "data.train_embeddings", required=True)
    _check(cfg.get("data", {}).get("train_ipr"), "data.train_ipr", required=True)

    if missing:
        msg = "Missing required input paths:\n" + "\n".join(f"  - {m}" for m in missing)
        LOGGER.error(msg)
        raise FileNotFoundError(msg)


def read_id_list(path: str) -> list[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def maybe_subsample_ids(ids: list[str], max_n, seed: int) -> list[str]:
    """
    Returns full list if max_n is None/"null", else reproducible random subset of size max_n.
    """
    if max_n is None:
        return ids
    if isinstance(max_n, str) and max_n.lower() == "null":
        return ids

    max_n = int(max_n)
    if max_n <= 0:
        raise ValueError(f"max_n must be positive or null/None, got {max_n}")
    if max_n > len(ids):
        raise ValueError(f"max_n={max_n} exceeds available IDs ({len(ids)})")

    rng = np.random.default_rng(seed)
    return rng.choice(ids, size=max_n, replace=False).tolist()


def log_embedding_stats(X_emb: np.ndarray) -> None:
    arr = np.asarray(X_emb)
    LOGGER.info("Embeddings: shape=%s dtype=%s", arr.shape, arr.dtype)

    n_nan = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
    n_inf = int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
    LOGGER.info("Embeddings: n_nan=%d n_inf=%d", n_nan, n_inf)

    if arr.size > 0 and np.issubdtype(arr.dtype, np.floating):
        try:
            mn = float(np.nanmin(arr))
            mx = float(np.nanmax(arr))
            LOGGER.info("Embeddings: min=%.6g max=%.6g", mn, mx)
        except ValueError:
            LOGGER.warning("Embeddings: could not compute min/max (all-NaN?)")


def load_protein2ipr(dat_path: str) -> dict[str, list[str]]:
    p = Path(dat_path)
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(
        p,
        sep="\t",
        header=None,
        usecols=[0, 1],
        dtype=str,
        comment="#",
    )
    if df.shape[1] < 2:
        raise ValueError(f"Expected >=2 columns in {p}, got {df.shape[1]}")

    out: dict[str, list[str]] = {}
    for pid, ipr in zip(df.iloc[:, 0], df.iloc[:, 1]):
        pid = str(pid).strip()
        ipr = str(ipr).strip()
        if not pid or not ipr:
            continue
        out.setdefault(pid, []).append(ipr)
    return out


def build_ipr_hash_matrix(
    protein_ids: list[str],
    pid2ipr: dict[str, list[str]],
    n_features: int,
) -> sp.csr_matrix:
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


def log_ipr_coverage(protein_ids: list[str], pid2ipr: dict[str, list[str]], X_ipr: sp.csr_matrix) -> None:
    covered = sum(pid in pid2ipr for pid in protein_ids)
    lens = [len(pid2ipr.get(pid, [])) for pid in protein_ids]
    row_nnz = np.diff(X_ipr.indptr)

    LOGGER.info(
        "InterPro coverage: %d/%d (%.2f%%) proteins have >=1 IPR",
        covered,
        len(protein_ids),
        100.0 * covered / max(1, len(protein_ids)),
    )
    LOGGER.info(
        "InterPro per-protein: mean=%.2f median=%d max=%d",
        float(np.mean(lens)),
        int(np.median(lens)),
        int(np.max(lens) if len(lens) else 0),
    )
    LOGGER.info(
        "X_ipr: shape=%s nnz=%d rows_nnz>0=%d/%d",
        X_ipr.shape,
        X_ipr.nnz,
        int((row_nnz > 0).sum()),
        X_ipr.shape[0],
    )
    LOGGER.info(
        "X_ipr row nnz: mean=%.2f median=%d max=%d",
        float(row_nnz.mean()) if row_nnz.size else 0.0,
        int(np.median(row_nnz)) if row_nnz.size else 0,
        int(row_nnz.max()) if row_nnz.size else 0,
    )


def filter_terms_present_in_subset(Y, go_terms: list[str], min_pos: int = 1):
    col_counts = np.asarray(Y.sum(axis=0)).ravel()
    keep = col_counts >= float(min_pos)

    if keep.sum() == 0:
        raise RuntimeError(f"No GO terms survive min_pos={min_pos} in this subset.")

    Yf = Y[:, keep] if sp.issparse(Y) else Y[:, keep]
    go_terms_f = [t for t, k in zip(go_terms, keep) if k]
    return Yf, go_terms_f, keep


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


def write_cafa_preds(
    out_file: Path,
    prot_ids: list[str],
    go_terms: list[str],
    probs: np.ndarray,
    top_k: int = 500,
    eps: float = 1e-12,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for i, pid in enumerate(prot_ids):
            row = probs[i]
            if row.size == 0:
                continue

            idx = np.argsort(row)[::-1][:top_k]
            for j in idx:
                s = float(row[j])
                if s <= eps:
                    break
                f.write(f"{pid} {go_terms[j]} {s:.6g}\n")


def write_cafa_gt(out_file: Path, prot_ids: list[str], asp_labels: dict[str, list[str]]) -> int:
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


def make_estimator(seed: int) -> OneVsRestClassifier:
    base = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        max_iter=50,
        early_stopping=False,
        tol=1e-3,
        random_state=seed,
        n_jobs=-1,
    )
    return OneVsRestClassifier(base, n_jobs=-1)


def tune_alpha_and_threshold(
    Y_true: np.ndarray,
    P_emb: np.ndarray,
    P_ipr: np.ndarray,
    thresholds: np.ndarray,
    alphas: np.ndarray,
) -> tuple[float, float, float]:
    best_f, best_a, best_t = -1.0, float(alphas[0]), float(thresholds[0])
    for a in alphas:
        P_mix = a * P_emb + (1.0 - a) * P_ipr
        f, t = compute_fmax(Y_true, P_mix, thresholds)
        if f > best_f:
            best_f, best_a, best_t = float(f), float(a), float(t)
    return best_f, best_a, best_t


def main():
    start_time = datetime.now()
    cfg = yaml.safe_load(open("configs/config.yaml"))

    log_cfg = cfg.get("logging", {})
    setup_logging(
        level=str(log_cfg.get("level", "INFO")),
        log_file=log_cfg.get("file", None),
    )

    LOGGER.info("Validating input paths...")
    validate_paths(cfg)
    LOGGER.info("Path validation complete.")

    LOGGER.info("Loading training IDs...")
    all_ids = read_id_list(cfg["data"]["train_ids"])
    if not all_ids:
        raise ValueError("No training IDs loaded (empty list).")

    LOGGER.info("Train proteins (all): %d", len(all_ids))
    LOGGER.info("First 3 train IDs: %s", all_ids[:3])

    # ---- Two-stage sizing: hyperparameter subset vs final training subset ----
    seed = int(cfg["run"]["seed"])
    subsample_seed = int(cfg.get("train", {}).get("subsample_seed", seed))

    max_n_hp = cfg.get("train", {}).get("max_n_hyperparameters", None)
    max_n_final = cfg.get("train", {}).get("max_n_final", None)

    train_ids_hp = maybe_subsample_ids(all_ids, max_n_hp, subsample_seed)
    train_ids_final = maybe_subsample_ids(all_ids, max_n_final, subsample_seed)

    LOGGER.info("Train proteins (hyperparameter subset): %d", len(train_ids_hp))
    LOGGER.info("Train proteins (final subset): %d", len(train_ids_final))

    # Output dirs
    models_dir = Path("models")
    artifacts_dir = Path("artifacts")
    metrics_dir = Path("metrics")
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ===== Build FEATURES for HP subset =====
    LOGGER.info("Loading embeddings (HP subset)...")
    X_emb_hp_dense = load_embeddings_h5(cfg["data"]["train_embeddings"], train_ids_hp)
    log_embedding_stats(X_emb_hp_dense)
    X_emb_hp = sp.csr_matrix(np.asarray(X_emb_hp_dense, dtype=np.float32))
    LOGGER.info("X_emb_hp: shape=%s nnz=%d", X_emb_hp.shape, X_emb_hp.nnz)

    LOGGER.info("Loading InterPro mappings...")
    ipr_path = cfg["data"]["train_ipr"]
    n_hash = int(cfg.get("features", {}).get("ipr_hash_dim", 2**18))
    pid2ipr = load_protein2ipr(ipr_path)

    LOGGER.info("Building InterPro hash matrix (HP subset)...")
    X_ipr_hp = build_ipr_hash_matrix(train_ids_hp, pid2ipr, n_features=n_hash)
    log_ipr_coverage(train_ids_hp, pid2ipr, X_ipr_hp)

    # ===== Build FEATURES for FINAL subset =====
    LOGGER.info("Loading embeddings (FINAL subset)...")
    X_emb_final_dense = load_embeddings_h5(cfg["data"]["train_embeddings"], train_ids_final)
    log_embedding_stats(X_emb_final_dense)
    X_emb_final = sp.csr_matrix(np.asarray(X_emb_final_dense, dtype=np.float32))
    LOGGER.info("X_emb_final: shape=%s nnz=%d", X_emb_final.shape, X_emb_final.nnz)

    LOGGER.info("Building InterPro hash matrix (FINAL subset)...")
    X_ipr_final = build_ipr_hash_matrix(train_ids_final, pid2ipr, n_features=n_hash)
    log_ipr_coverage(train_ids_final, pid2ipr, X_ipr_final)

    # Load labels
    LOGGER.info("Loading labels...")
    labels = load_train_labels(cfg["data"]["train_labels"])
    aspects = cfg["run"]["aspects"]
    LOGGER.info("Aspects: %s", aspects)

    # Training parameters
    min_pos = int(cfg.get("train", {}).get("min_pos_in_subset", 1))
    cv_folds = int(cfg["run"].get("cv_folds", 3))
    cv = MultilabelStratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    thresholds = np.linspace(0.01, 0.50, 50)
    alphas = np.linspace(0.0, 1.0, 11)

    for aspect in aspects:
        LOGGER.info("=== Aspect: %s ===", aspect)
        asp_labels = labels[aspect]

        # Build shared label space from all labels for this aspect
        go_terms, _ = build_label_space(asp_labels)
        term2idx = {t: i for i, t in enumerate(go_terms)}

        # Y for hyperparameter subset (tuning/CV)
        Y_hp = build_Y(train_ids_hp, asp_labels, term2idx)
        # Y for final subset (final fit + saved models)
        Y_final = build_Y(train_ids_final, asp_labels, term2idx)

        LOGGER.info("Initial Y_hp.shape = %s (terms=%d)", Y_hp.shape, len(go_terms))
        LOGGER.info("Initial Y_final.shape = %s (terms=%d)", Y_final.shape, len(go_terms))

        # Filter based on HP subset, then apply same mask to FINAL
        Yf_hp, go_terms_f, keep = filter_terms_present_in_subset(Y_hp, go_terms, min_pos=min_pos)
        Y_train_hp = Yf_hp.toarray() if sp.issparse(Yf_hp) else np.asarray(Yf_hp)

        Yf_final = Y_final[:, keep] if sp.issparse(Y_final) else Y_final[:, keep]
        Y_train_final = Yf_final.toarray() if sp.issparse(Yf_final) else np.asarray(Yf_final)

        LOGGER.info("Classes kept (pos>=%d): %d / %d", min_pos, len(go_terms_f), len(go_terms))

        # ---- HP subset selection for CV/tuning ----
        has_label_hp = (Y_train_hp.sum(axis=1) > 0)
        n_lab_hp = int(has_label_hp.sum())
        LOGGER.info("HP subset proteins with >=1 label in %s: %d / %d", aspect, n_lab_hp, Y_train_hp.shape[0])

        if n_lab_hp < cv_folds:
            raise RuntimeError(
                f"Not enough labeled proteins for CV in HP subset (have {n_lab_hp}, folds={cv_folds})"
            )

        X_emb_hp_a = X_emb_hp[has_label_hp]
        X_ipr_hp_a = X_ipr_hp[has_label_hp]
        Y_hp_a = Y_train_hp[has_label_hp]

        # ---- FINAL subset selection for final fit ----
        has_label_final = (Y_train_final.sum(axis=1) > 0)
        n_lab_final = int(has_label_final.sum())
        LOGGER.info(
            "FINAL subset proteins with >=1 label in %s: %d / %d", aspect, n_lab_final, Y_train_final.shape[0]
        )

        X_emb_final_a = X_emb_final[has_label_final]
        X_ipr_final_a = X_ipr_final[has_label_final]
        Y_final_a = Y_train_final[has_label_final]

        # ---- CV on HP subset ----
        est = make_estimator(seed)

        LOGGER.info("Running CV (embeddings-only) on HP subset: folds=%d", cv_folds)
        t0 = time.time()
        P_emb = cross_val_predict(est, X_emb_hp_a, Y_hp_a, cv=cv, method="predict_proba", n_jobs=1)
        LOGGER.info("CV embeddings-only complete in %.2f s", time.time() - t0)

        LOGGER.info("Running CV (interpro-only) on HP subset: folds=%d", cv_folds)
        t0 = time.time()
        P_ipr = cross_val_predict(est, X_ipr_hp_a, Y_hp_a, cv=cv, method="predict_proba", n_jobs=1)
        LOGGER.info("CV interpro-only complete in %.2f s", time.time() - t0)

        f_emb, t_emb = compute_fmax(Y_hp_a, P_emb, thresholds)
        f_ipr, t_ipr = compute_fmax(Y_hp_a, P_ipr, thresholds)
        LOGGER.info("Fmax embeddings-only (HP) = %.4f @ t=%.3f", f_emb, t_emb)
        LOGGER.info("Fmax interpro-only   (HP) = %.4f @ t=%.3f", f_ipr, t_ipr)

        f_best, a_best, t_best = tune_alpha_and_threshold(Y_hp_a, P_emb, P_ipr, thresholds, alphas)
        LOGGER.info("Best ensemble (HP): Fmax=%.4f alpha=%.2f threshold=%.3f", f_best, a_best, t_best)

        # ---- CAFA inputs from HP-CV preds ----
        cafa_root = Path("results") / "cafa_inputs" / aspect
        pred_dir = cafa_root / "preds"
        pred_file = pred_dir / "pred.tsv"
        gt_file = cafa_root / "gt.tsv"

        pred_dir.mkdir(parents=True, exist_ok=True)
        cafa_root.mkdir(parents=True, exist_ok=True)
        for old in pred_dir.glob("*"):
            old.unlink()

        P_mix = a_best * P_emb + (1.0 - a_best) * P_ipr
        train_ids_lab = [pid for pid, ok in zip(train_ids_hp, has_label_hp) if ok]

        write_cafa_preds(pred_file, train_ids_lab, go_terms_f, P_mix, top_k=500)
        gt_lines = write_cafa_gt(gt_file, train_ids_lab, asp_labels)

        LOGGER.info("[CAFA inputs] preds: %s", pred_file)
        LOGGER.info("[CAFA inputs] gt:    %s (lines=%d)", gt_file, gt_lines)

        # ---- FINAL fit on FINAL subset ----
        LOGGER.info("Fitting FINAL per-aspect models on FINAL subset...")
        est_emb_final = make_estimator(seed)
        est_ipr_final = make_estimator(seed)

        t0 = time.time()
        est_emb_final.fit(X_emb_final_a, Y_final_a)
        LOGGER.info("Final embeddings model fit in %.2f s", time.time() - t0)

        t0 = time.time()
        est_ipr_final.fit(X_ipr_final_a, Y_final_a)
        LOGGER.info("Final interpro model fit in %.2f s", time.time() - t0)

        # Save GO term list
        go_terms_path = artifacts_dir / f"go_terms_{aspect}.pkl"
        joblib.dump(go_terms_f, go_terms_path)
        LOGGER.info("Saved GO term list: %s", go_terms_path)

        # Save ensemble bundle
        model_path = models_dir / f"model_{aspect}.pkl"
        bundle = {
            "aspect": aspect,
            "model_emb": est_emb_final,
            "model_ipr": est_ipr_final,
            "alpha": float(a_best),
            "threshold": float(t_best),
            "go_terms": go_terms_f,
            "metadata": {
                "ipr_hash_dim": int(n_hash),
                "ipr_doc_convention": "docs = ' '.join(sorted(set(pid2ipr[pid]))) ; tokenizer=str.split ; lowercase=False ; alternate_sign=False",
                "min_pos_in_subset": int(min_pos),
                "cv_folds": int(cv_folds),
                "seed": int(seed),
                "max_n_hyperparameters": None if max_n_hp is None else max_n_hp,
                "max_n_final": None if max_n_final is None else max_n_final,
            },
        }
        joblib.dump(bundle, model_path)
        LOGGER.info("Saved model bundle: %s", model_path)

        # Save metrics
        metrics_path = metrics_dir / f"metrics_{aspect}.json"
        metrics = {
            "aspect": aspect,
            "n_train_ids_hp": int(len(train_ids_hp)),
            "n_labeled_hp": int(n_lab_hp),
            "n_train_ids_final": int(len(train_ids_final)),
            "n_labeled_final": int(n_lab_final),
            "n_terms_initial": int(len(go_terms)),
            "n_terms_kept": int(len(go_terms_f)),
            "fmax_emb_hp": float(f_emb),
            "t_emb_hp": float(t_emb),
            "fmax_ipr_hp": float(f_ipr),
            "t_ipr_hp": float(t_ipr),
            "fmax_ens_hp": float(f_best),
            "alpha_best": float(a_best),
            "t_best": float(t_best),
        }
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        LOGGER.info("Saved metrics: %s", metrics_path)

    LOGGER.info("Elapsed time: %s", datetime.now() - start_time)


if __name__ == "__main__":
    main()
