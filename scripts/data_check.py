import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import yaml
from src.utils.io import read_id_list
from src.data.load_labels import load_train_labels
from src.data.load_embeddings import load_embeddings_h5
from src.data.build_matrices import build_label_space, build_Y

def main():
    cfg = yaml.safe_load(open("configs/config.yaml"))

    train_ids = read_id_list(cfg["data"]["train_ids"])
    test_ids = read_id_list(cfg["data"]["test_ids"])

    labels = load_train_labels(cfg["data"]["train_labels"])

    X_train = load_embeddings_h5(cfg["data"]["train_embeddings"], train_ids)
    X_test = load_embeddings_h5(cfg["data"]["test_embeddings"], test_ids)

    print("X_train:", X_train.shape, "X_test:", X_test.shape)

    for aspect in cfg["run"]["aspects"]:
        asp_labels = labels.get(aspect, {})
        terms, _ = build_label_space(asp_labels)
        Y = build_Y(train_ids, asp_labels, {t:i for i,t in enumerate(terms)})
        print(f"{aspect}: terms={len(terms)}, Y={Y.shape}")

if __name__ == "__main__":
    main()
