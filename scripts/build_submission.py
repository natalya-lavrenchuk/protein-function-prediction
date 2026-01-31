# Setting path to root directory so that the master_file can call it correctly
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import yaml
from collections import defaultdict
from typing import Dict, List, Tuple


# Formatting rules for output: no zeros, 3 sig figs
def fmt_score(p: float, eps: float = 1e-6) -> str:
    p = float(p)
    if p <= 0:
        p = eps
    if p > 1:
        p = 1.0
    return f"{p:.3g}"


def main():
    cfg = yaml.safe_load(open("configs/config.yaml", "r", encoding="utf-8"))

    # Where the test inference script wrote its outputs
    # results/test_inference_outputs/<aspect>/preds/pred_test.tsv
    test_preds_root = Path("results") / "test_inference_outputs"

    aspects = cfg["run"]["aspects"]

    # Output file (distinct name + folder)
    out_dir = Path(cfg.get("output", {}).get("dir", "outputs")) / "test_inference_aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "submission_test_inference.txt"

    # Aggregate all predictions across aspects per protein
    # per_protein[pid] -> list of (go_term, score)
    per_protein: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    # Read each aspect pred_test.tsv
    for aspect in aspects:
        pred_file = test_preds_root / aspect / "preds" / "pred_test.tsv"
        if not pred_file.exists():
            raise FileNotFoundError(f"Missing test inference preds for aspect '{aspect}': {pred_file}")

        with pred_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Expected format: PID GO:XXXXXXX score
                pid, go, s = line.split()
                per_protein[pid].append((go, float(s)))

    # Write file with 1500-term cap per protein
    # IMPORTANT: we need a stable list of protein IDs. Use the test_ids file from config.
    test_ids_path = cfg["data"]["test_ids"]
    with open(test_ids_path, "r", encoding="utf-8") as f:
        test_ids = [ln.strip() for ln in f if ln.strip()]

    with out_path.open("w", encoding="utf-8") as f:
        for pid in test_ids:
            items = per_protein.get(pid, [])
            # sort by score desc
            items.sort(key=lambda x: x[1], reverse=True)
            items = items[:1500]
            for go, p in items:
                f.write(f"{pid} {go} {fmt_score(p)}\n")

    print("Wrote submission:", out_path)
    print("Example lines:")
    with out_path.open("r", encoding="utf-8") as f:
        for _ in range(10):
            line = f.readline()
            if not line:
                break
            print(line.rstrip())


if __name__ == "__main__":
    main()
