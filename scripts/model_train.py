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

    # Read training protein IDs
    all_ids = read_id_list(cfg["data"]["train_ids"])

    # Use a smaller subset 
    train_ids = all_ids[:1000]

    # Load GO labels and ProtT5 embeddings
    labels = load_train_labels(cfg["data"]["train_labels"])
    X = load_embeddings_h5(cfg["data"]["train_embeddings"], train_ids)

    print(f"Embeddings shape: {X.shape}")

    # Use 2-fold CV to reduce runtime
    cv = KFold(n_splits=2, shuffle=True, random_state=42)

    for aspect in cfg["run"]["aspects"]:
        print(f"\nProcessing Aspect: {aspect}")

        # Get labels for the current GO aspect
        asp_labels = labels[aspect]
        go_terms, _ = build_label_space(asp_labels)
        term2idx = {t: i for i, t in enumerate(go_terms)}

        # Build multi-label target matrix
        Y = build_Y(train_ids, asp_labels, term2idx)
        Y = Y.toarray() if hasattr(Y, "toarray") else Y

        # Remove very rare GO terms to stabilize training
        min_samples = 10
        term_freq = np.sum(Y, axis=0)
        valid = term_freq >= min_samples

        Y = Y[:, valid]
        go_terms = [t for i, t in enumerate(go_terms) if valid[i]]

        # Limit number of GO terms to keep runtime reasonable
        max_terms = 200
        if Y.shape[1] > max_terms:
            idx = np.argsort(-term_freq[valid])[:max_terms]
            Y = Y[:, idx]
            go_terms = [go_terms[i] for i in idx]

        if Y.shape[1] == 0:
            print("No usable GO terms found, skipping.")
            continue

        print(f"Using {Y.shape[1]} GO terms")

        # Save GO term order for later predictions
        joblib.dump(go_terms, f"artifacts/go_terms_{aspect}.pkl")

        # Fast cross-validation using liblinear solver
        cv_model = OneVsRestClassifier(
            LogisticRegression(
                solver="liblinear",
                max_iter=100,
                class_weight="balanced"
            ),
            n_jobs=1
        )

        print("Running cross-validation...")
        Y_prob = cross_val_predict(
            cv_model, X, Y, cv=cv, method="predict_proba", n_jobs=1
        )

        # Evaluate model performance using F-max
        thresholds = np.linspace(0.1, 0.9, 20)
        fmax, best_t = compute_fmax(Y, Y_prob, thresholds)

        print(f"F-max = {fmax:.4f} at threshold ~ {best_t:.2f}")

        np.savez(
            f"results/metrics_{aspect}.npz",
            fmax=fmax,
            threshold=best_t
        )

         # Train final model using SAGA solver (as recommended in lectures)
        print("Training final model with SAGA solver...")
        final_model = OneVsRestClassifier(
            LogisticRegression(
                solver="saga",
                max_iter=300,
                class_weight="balanced"
            ),
            n_jobs=1
        )

        final_model.fit(X, Y)
        joblib.dump(final_model, f"models/model_{aspect}.pkl")

        print("Model saved successfully.")

    print(f"\nDone. Total time: {datetime.now() - start_time}")


if __name__ == "__main__":
    main()