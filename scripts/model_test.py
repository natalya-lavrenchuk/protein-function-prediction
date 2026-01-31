import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import logging
import warnings
import yaml
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.exceptions import ConvergenceWarning

from src.data.load_embeddings import load_embeddings_h5

LOGGER = logging.getLogger("model_test")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    lvl = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(lvl)
    ch = logging.StreamHandler()
    ch.setLevel(lvl)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root.addHandler(ch)
    LOGGER.setLevel(lvl)
    LOGGER.propagate = True


def read_id_list(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_protein2ipr(dat_path: str) -> dict[str, list[str]]:
    """
    Expects a tab-separated file:
      col0 = protein_id
      col1 = IPR id
    """
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


def predict_bundle(bundle: dict, X_emb: sp.csr_matrix, X_ipr: sp.csr_matrix) -> np.ndarray:
    """
    Returns ensemble probabilities P_mix for all test proteins.
    """
    m_emb = bundle["model_emb"]
    m_ipr = bundle["model_ipr"]
    a = float(bundle["alpha"])

    P_emb = m_emb.predict_proba(X_emb)
    P_ipr = m_ipr.predict_proba(X_ipr)
    return a * P_emb + (1.0 - a) * P_ipr


def main():
    cfg = yaml.safe_load(open("configs/config.yaml", "r", encoding="utf-8"))
    setup_logging(level=str(cfg.get("logging", {}).get("level", "INFO")))

    # Required test paths (put these in config.yaml)
    # data:
    #   test_ids: data/raw/biological_data_pfp/test/test_ids.txt
    #   test_embeddings: data/raw/biological_data_pfp/test/test_embeddings.h5
    #   test_ipr: data/raw/biological_data_pfp/test/test_protein2ipr.dat
    data = cfg.get("data", {})
    test_ids_path = data.get("test_ids")
    test_emb_path = data.get("test_embeddings")
    test_ipr_path = data.get("test_ipr")

    if not test_ids_path or not test_emb_path or not test_ipr_path:
        raise ValueError("config.yaml must define data.test_ids, data.test_embeddings, data.test_ipr")

    test_ids = read_id_list(test_ids_path)
    if not test_ids:
        raise ValueError("test_ids.txt was empty")

    aspects = cfg.get("run", {}).get("aspects", [])
    if not aspects:
        raise ValueError("config.yaml must define run.aspects (MF/BP/CC)")

    # IMPORTANT: use a distinct output root so it cannot be confused with training CV outputs
    out_root = Path("results") / "test_inference_outputs"
    out_root.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Test proteins: %d", len(test_ids))
    LOGGER.info("Writing test outputs under: %s", out_root)

    # Load test embeddings
    LOGGER.info("Loading test embeddings: %s", test_emb_path)
    X_emb_dense = load_embeddings_h5(test_emb_path, test_ids)
    X_emb = sp.csr_matrix(np.asarray(X_emb_dense, dtype=np.float32))
    LOGGER.info("X_emb test: %s", X_emb.shape)

    # Load test InterPro mapping once
    LOGGER.info("Loading test InterPro: %s", test_ipr_path)
    pid2ipr = load_protein2ipr(test_ipr_path)

    # For each aspect, load the saved model bundle and generate CAFA preds
    for aspect in aspects:
        model_path = Path("models") / f"model_{aspect}.pkl"
        if not model_path.exists():
            LOGGER.warning("Missing model file for %s: %s (skipping)", aspect, model_path)
            continue

        LOGGER.info("=== Aspect: %s ===", aspect)
        bundle = joblib.load(model_path)

        go_terms = bundle["go_terms"]
        n_hash = int(bundle.get("metadata", {}).get("ipr_hash_dim", 2**18))

        LOGGER.info("Building test IPR hash matrix (n_hash=%d)", n_hash)
        X_ipr = build_ipr_hash_matrix(test_ids, pid2ipr, n_features=n_hash)
        LOGGER.info("X_ipr test: %s", X_ipr.shape)

        LOGGER.info("Predicting...")
        P_mix = predict_bundle(bundle, X_emb, X_ipr)

        # Distinct filenames to avoid confusion with training CV cafa_inputs outputs
        aspect_dir = out_root / aspect
        pred_dir = aspect_dir / "preds"
        pred_file = pred_dir / "pred_test.tsv"

        write_cafa_preds(pred_file, test_ids, go_terms, P_mix, top_k=500)
        LOGGER.info("Wrote test predictions: %s", pred_file)

        # Optional: also save a tiny manifest for traceability
        manifest = {
            "aspect": aspect,
            "n_test": int(len(test_ids)),
            "n_terms": int(len(go_terms)),
            "model_path": str(model_path),
            "alpha": float(bundle.get("alpha", 0.5)),
            "threshold": float(bundle.get("threshold", 0.5)),
            "ipr_hash_dim": int(n_hash),
            "test_ids_path": str(test_ids_path),
            "test_embeddings_path": str(test_emb_path),
            "test_ipr_path": str(test_ipr_path),
            "output_pred_file": str(pred_file),
        }
        with (aspect_dir / "manifest_test.json").open("w", encoding="utf-8") as f:
            import json
            json.dump(manifest, f, indent=2)
        LOGGER.info("Wrote manifest: %s", aspect_dir / "manifest_test.json")

    LOGGER.info("Done. Test inference outputs in: %s", out_root)


if __name__ == "__main__":
    main()
