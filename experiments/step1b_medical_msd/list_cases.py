"""
list_cases.py
=============
Lists the first available cases in imagesTr/ for an MSD task.
Useful for retrieving a case name for inference.

Usage:
    python experiments/step1b_medical_msd/list_cases.py --data_dir ./data/step1b_medical_msd/Task09_Spleen
    python experiments/step1b_medical_msd/list_cases.py --data_dir ./data/step1b_medical_msd/Task09_Spleen --n 5
"""

import argparse
from pathlib import Path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    img_dir = Path(args.data_dir) / "imagesTr"
    lbl_dir = Path(args.data_dir) / "labelsTr"
    cases = sorted([f.name for f in img_dir.glob("*.nii.gz") if not f.name.startswith("._")])
    print(f"\n[{Path(args.data_dir).name}]  {len(cases)} cases in imagesTr/\n")
    for c in cases[:args.n]:
        has_lbl = (lbl_dir / c).exists()
        print(f"  {c}  {'[label OK]' if has_lbl else '[NO LABEL]'}")
    if len(cases) > args.n:
        print(f"  ... ({len(cases) - args.n} more)")
    print(f"\nFirst case: {cases[0]}")
