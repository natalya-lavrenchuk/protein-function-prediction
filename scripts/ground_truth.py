from pathlib import Path
import yaml

REPO_ROOT = Path(r"C:\work\protein-function-prediction")

CONFIG_PATH = REPO_ROOT / "configs/config.yaml"
TRAIN_SET_TSV = REPO_ROOT / "data/raw/biological_data_pfp/train/train_set.tsv"
TRAIN_IDS_TXT = REPO_ROOT / "data/raw/biological_data_pfp/train/train_ids.txt"
OUT_DIR = REPO_ROOT / "data/processed"


def load_config(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_train_ids(path: Path, max_n=None):
    """
    Load training protein IDs.
    If max_n is set, return only the first max_n IDs.
    """
    ids = [line.strip() for line in open(path) if line.strip()]

    if max_n is not None:
        ids = ids[:max_n]

    return set(ids)


def make_ground_truth(train_set_tsv: Path, out_tsv: Path, protein_ids, aspect):
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    wrote = 0
    with open(train_set_tsv, "r") as fin, open(out_tsv, "w") as fout:
        for line in fin:
            pid, asp, go = line.rstrip("\n").split("\t")

            if pid not in protein_ids:
                continue
            if asp != aspect:
                continue

            fout.write(f"{pid} {go}\n")
            wrote += 1

    print(f"[OK] {aspect}: wrote {wrote} lines → {out_tsv}")

def main():
    cfg = load_config(CONFIG_PATH)

    max_n = cfg["train"].get("max_n", None)
    aspects = cfg["run"]["aspects"]

    print(f"Using max_n = {max_n}")

    protein_ids = load_train_ids(TRAIN_IDS_TXT, max_n=max_n)
    print(f"Number of proteins in GT: {len(protein_ids)}")

    for aspect in aspects:
        out_file = OUT_DIR / f"ground_truth_{aspect}.tsv"
        make_ground_truth(
            train_set_tsv=TRAIN_SET_TSV,
            out_tsv=out_file,
            protein_ids=protein_ids,
            aspect=aspect
        )


if __name__ == "__main__":
    main()
