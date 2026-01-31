import argparse
import subprocess
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obo", required=True, help="Path to go-basic.obo")
    ap.add_argument("--pred_dir", required=True, help="Directory containing CAFA preds files")
    ap.add_argument("--gt", required=True, help="Ground truth tsv (protein_id GO:xxxxxxx)")
    ap.add_argument("--out_dir", required=True, help="Output directory for evaluator results")
    ap.add_argument("--use_module", action="store_true",
                    help="If set, run evaluator as: python -m cafaeval ... instead of cafaeval ...")
    args = ap.parse_args()

    obo = Path(args.obo)
    pred_dir = Path(args.pred_dir)
    gt = Path(args.gt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not obo.exists():
        raise FileNotFoundError(obo)
    if not pred_dir.exists():
        raise FileNotFoundError(pred_dir)
    if not gt.exists():
        raise FileNotFoundError(gt)

    # ---- Adjust this command if your course uses a different evaluator entrypoint ----
    if args.use_module:
        cmd = [sys.executable, "-m", "cafaeval",
               "--obo", str(obo),
               "--pred_dir", str(pred_dir),
               "--gt", str(gt),
               "--out_dir", str(out_dir)]
    else:
        cmd = ["cafaeval",
               "--obo", str(obo),
               "--pred_dir", str(pred_dir),
               "--gt", str(gt),
               "--out_dir", str(out_dir)]

    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)

    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        raise SystemExit(res.returncode)


if __name__ == "__main__":
    main()
