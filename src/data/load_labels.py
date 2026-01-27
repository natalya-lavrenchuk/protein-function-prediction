from collections import defaultdict
import csv

def load_train_labels(tsv_path: str):
    labels = defaultdict(lambda: defaultdict(set))
    with open(tsv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid = row["Protein_ID"].strip()
            asp = row["aspect"].strip()
            go  = row["GO_term"].strip()   # <-- FIXED
            if pid and asp and go:
                labels[asp][pid].add(go)
    return labels
