import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import yaml
import joblib
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.exceptions import ConvergenceWarning

from src.utils.io import read_id_list
from src.data.load_labels import load_train_labels
from src.data.load_embeddings import load_embeddings_h5
from src.data.build_matrices import build_label_space, build_Y

# Ignore convergence warnings to keep output clean
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def compute_fmax(y_true, y_prob, thresholds):
    """
    Compute F-max by trying different probability thresholds
    and keeping the best F1 score.
    """
    best_f, best_t = 0.0, 0.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum(y_pred * y_true)
        fp = np.sum(y_pred * (1 - y_true))
        fn = np.sum((1 - y_pred) * y_true)
        if tp == 0:
            continue

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f = 2 * precision * recall / (precision + recall)
        if f > best_f:
            best_f, best_t = f, t

    return best_f, best_t

def main():
    start_time = datetime.now()
    cfg = yaml.safe_load(open("configs/config.yaml"))

    # Create output folders if they do not exist
    for folder in ["models", "artifacts", "results"]:
        Path(folder).mkdir(exist_ok=True)

    print("Loading training data...")

    #Read training protein IDs and subsample
    all_ids = read_id_list(cfg["data"]["train_ids"])
    max_n = cfg.get("train", {}).get("max_n", None)
    if max_n is None:
        train_ids = all_ids
    else:
        rng = np.random.default_rng(cfg.get("train", {}).get("subsample_seed", cfg["run"]["seed"]))
        max_n = int(max_n)
        if max_n > len(all_ids):
            raise ValueError(f"train.max_n={max_n} exceeds available IDs ({len(all_ids)})")
        train_ids = rng.choice(all_ids, size=max_n, replace=False).tolist()

    # Load GO labels and ProtT5 embeddings
    labels = load_train_labels(cfg["data"]["train_labels"])
    X = load_embeddings_h5(cfg["data"]["train_embeddings"], train_ids)

    print(f"Embeddings shape: {X.shape}")

    cv_folds = int(cfg["run"].get("cv_folds", 10))
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=cfg["run"]["seed"])

    for aspect in cfg["run"]["aspects"]:
        print(f"\nProcessing Aspect: {aspect}")

        # Get labels for the current GO aspect
        asp_labels = labels[aspect]
        go_terms, _ = build_label_space(asp_labels)
        term2idx = {t: i for i, t in enumerate(go_terms)}

        # Build multi-label target matrix
        Y = build_Y(train_ids, asp_labels, term2idx)
        Y = Y.toarray() if hasattr(Y, "toarray") else Y

        #I (Christy) removed your pruning limits, I think we should not remove possible labels during training [Remove very rare GO terms to stabilize training + Limit number of GO terms to keep runtime reasonable] but happy to discuss! I think the label space has already been limited by instructor

        if Y.shape[1] == 0:
            print("No usable GO terms found, skipping.")
            continue

        print(f"Using {Y.shape[1]} GO terms")

        # Save GO term order for later predictions
        joblib.dump(go_terms, f"artifacts/go_terms_{aspect}.pkl")

        # Cross-validation
        solver_cv = cfg.get("model", {}).get("solver_cv", "saga")
        max_iter_cv = int(cfg.get("model", {}).get("max_iter_cv", 100))
        cv_model = OneVsRestClassifier(LogisticRegression(
                solver=solver_cv,
                max_iter=max_iter_cv,
                class_weight="balanced",
                random_state=cfg["run"]["seed"],),n_jobs=-1)

        print("Running cross-validation...")
        Y_prob = cross_val_predict(
            cv_model, X, Y, cv=cv, method="predict_proba", n_jobs=1)

        # Evaluate model performance using F-max
        thresholds = np.linspace(0.1, 0.9, 10) #Thresholds changes from 20->10 
        fmax, best_t = compute_fmax(Y, Y_prob, thresholds)
        print(f"F-max = {fmax:.4f} at threshold ~ {best_t:.2f}")

        np.savez(f"results/metrics_{aspect}.npz",
            fmax=fmax,
            threshold=best_t)

         # Train final model using SAGA solver (as recommended in lectures)
        print("Training final model with SAGA solver...")
        solver_final = cfg.get("model", {}).get("solver_final", "saga")
        max_iter_final = int(cfg.get("model", {}).get("max_iter_final", 300))
        final_model = OneVsRestClassifier(LogisticRegression(
                solver=solver_final,
                max_iter=max_iter_final,
                class_weight="balanced",
                random_state=cfg["run"]["seed"]),n_jobs=-1)

        final_model.fit(X, Y)
        joblib.dump(final_model, f"models/model_{aspect}.pkl")

        print("Model saved successfully.")
    print(f"\nDone. Total time: {datetime.now() - start_time}")

if __name__ == "__main__":
    main()