import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import pandas as pd
from cafaeval.evaluation import cafa_eval, write_results


def extract_fmax(out_dir: Path) -> float:
    """
    Robust extraction of Fmax from cafaeval outputs.
    Looks for evaluation_best_f.tsv (common) or best-* files.
    """
    best_path = out_dir / "evaluation_best_f.tsv"
    if not best_path.exists():
        candidates = list(out_dir.glob("evaluation_best*.tsv"))
        if not candidates:
            raise FileNotFoundError(f"No cafaeval best-results file found in {out_dir}")
        best_path = candidates[0]

    df = pd.read_csv(best_path, sep="\t")
    for col in ["fmax", "Fmax", "f_max", "F_max"]:
        if col in df.columns:
            return float(df[col].iloc[0])

    # fallback: pick the first column that looks like f
    for col in df.columns:
        if "f" in col.lower():
            try:
                return float(df[col].iloc[0])
            except Exception:
                pass

    raise ValueError(f"Could not find an Fmax column in {best_path}. Columns: {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser(description="Run CAFA evaluator on predictions.")
    ap.add_argument("--obo", required=True, help="Path to go-basic.obo")
    ap.add_argument("--pred_dir", required=True, help="Directory containing prediction .tsv files")
    ap.add_argument("--gt", required=True, help="Ground truth .tsv file (CAFA format: PID GO)")
    ap.add_argument("--out_dir", required=True, help="Output directory for cafaeval results")
    ap.add_argument("--th_step", type=float, default=None, help="Optional threshold step (e.g., 0.02)")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluator
    if args.th_step is None:
        res = cafa_eval(args.obo, args.pred_dir, args.gt)
    else:
        res = cafa_eval(args.obo, args.pred_dir, args.gt, th_step=args.th_step)

    # If overlap is 0, cafaeval returns empty/None structures
    if res is None or len(res) < 2 or res[0] is None:
        raise RuntimeError(
            "cafaeval produced no evaluation (likely empty predictions or zero overlap with ground truth)."
        )

    df, dfs_best = res[0], res[1]
    # Some cafaeval versions also return extra elements; write_results accepts (df, dfs_best,...)
    write_results(df, dfs_best, out_dir=str(out_dir))

    fmax = extract_fmax(out_dir)
    print(f"Fmax: {fmax:.4f}")
    print(f"Results written to: {out_dir}")


if __name__ == "__main__":
    main()
