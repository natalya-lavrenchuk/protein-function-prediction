#Setting path to root directory so that the master_file can call it correctly
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

#Library import
import yaml
import joblib
import numpy as np
from src.utils.io import read_id_list
from src.data.load_embeddings import load_embeddings_h5

#Formatting rules for output: No zeros, 3 sig figs
def fmt_score(p: float, eps: float = 1e-6) -> str:
    p = float(p)
    if p <= 0:
        p = eps # Equal to 1e-6
    if p > 1:
        p = 1.0
    return f"{p:.3g}"  # 3 significant figures


def main():
    
    #Load test proteins and associated data
    cfg = yaml.safe_load(open("configs/config.yaml"))
    test_ids = read_id_list(cfg["data"]["test_ids"])
    X_test = load_embeddings_h5(cfg["data"]["test_embeddings"], test_ids)

    #Output file
    out_dir = Path(cfg.get("output", {}).get("dir", "outputs"))
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "submission.txt"

    #Structure for aggregation across aspects
    per_protein = {pid: [] for pid in test_ids}

    #Loop across BP, MF, CC
    for aspect in cfg["run"]["aspects"]:
        model_path = Path("models") / f"model_{aspect}.pkl"
        terms_path = Path("artifacts") / f"go_terms_{aspect}.pkl"
        metrics_path = Path("results") / f"metrics_{aspect}.npz"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")
        if not terms_path.exists():
            raise FileNotFoundError(f"Missing GO terms list: {terms_path}")
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics/threshold: {metrics_path}")

        #Load artifacts and thresholds
        model = joblib.load(model_path)
        go_terms = joblib.load(terms_path)
        metrics = np.load(metrics_path)
        threshold = float(metrics["threshold"])

        #Predict probabilities that each protein has given GO term 
        P = model.predict_proba(X_test)  # (n_test, n_terms)

        #Filter predictions by threshold
        for i, pid in enumerate(test_ids):
            probs = P[i]
            idx = np.where(probs >= threshold)[0]

            #If none meet cutoff, take top 5
            if idx.size == 0:
                idx = np.argsort(-probs)[:5]

            #Sort
            idx = idx[np.argsort(-probs[idx])]

            #Append to list
            for j in idx:
                per_protein[pid].append((go_terms[j], float(probs[j])))

    #Write file with 1500-term cap 
    with open(out_path, "w", encoding="utf-8") as f:
        for pid in test_ids:
            items = per_protein[pid]
            items.sort(key=lambda x: x[1], reverse=True)
            items = items[:1500]

            for go, p in items:
                f.write(f"{pid} {go} {fmt_score(p)}\n")

    print("Wrote submission:", out_path)
    print("Example lines:")
    with open(out_path, "r", encoding="utf-8") as f:
        for _ in range(10):
            print(f.readline().rstrip())


if __name__ == "__main__":
    main()
