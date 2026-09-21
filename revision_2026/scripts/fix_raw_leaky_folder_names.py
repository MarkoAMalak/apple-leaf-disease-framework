#!/usr/bin/env python3
"""One-time fix for a data-path mismatch caught while reviewing 02_leaky_split_experiment.py.

02_leaky_split_experiment.py's --raw-root is loaded with PathImageFolder using
class_to_idx built from common.py's CLASSES = ["apple_scab", "black_rot",
"cedar_apple_rust", "healthy"] -- it only looks for subfolders with exactly
those (lowercase, no prefix) names.

But prepare_data.py's Section 7 extracted archive.zip's own folder names into
data/plantvillage_raw_leaky/, which are "Apple___Apple_scab",
"Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy" (matching
the zip's internal Dataset-Classification layout, not common.py's CLASSES).

Net effect if this isn't fixed: PathImageFolder silently finds ZERO images
under --raw-root, and 02_leaky_split_experiment.py would crash (or worse,
silently run on an empty dataset) the moment you try to use it.

This script renames the 4 class folders in place to match what
02_leaky_split_experiment.py actually expects. Safe to re-run: skips a
folder if the target name already exists, or if the source is missing.

Run this ONCE, from the "code soic" folder (the same folder prepare_data.py
lives in, one level above "missing_experiments"):
    python fix_raw_leaky_folder_names.py
"""
from pathlib import Path

RAW_LEAKY = Path(__file__).resolve().parent / "data" / "plantvillage_raw_leaky"

RENAME_MAP = {
    "Apple___Apple_scab": "apple_scab",
    "Apple___Black_rot": "black_rot",
    "Apple___Cedar_apple_rust": "cedar_apple_rust",
    "Apple___healthy": "healthy",
}

print(f"Looking in: {RAW_LEAKY}")
if not RAW_LEAKY.exists():
    print(f"  [MISSING] {RAW_LEAKY} does not exist -- did prepare_data.py's Section 7 run/succeed?")
else:
    for old, new in RENAME_MAP.items():
        src = RAW_LEAKY / old
        dst = RAW_LEAKY / new
        if dst.exists():
            n = sum(1 for _ in dst.iterdir())
            print(f"  [skip, already exists] {dst}  ({n} files)")
        elif src.exists():
            src.rename(dst)
            n = sum(1 for _ in dst.iterdir())
            print(f"  [renamed] {old}  ->  {new}   ({n} files)")
        else:
            print(f"  [MISSING] {src} not found -- check the folder listing below")

    print("\nCurrent contents of plantvillage_raw_leaky:")
    for p in sorted(RAW_LEAKY.iterdir()):
        n = sum(1 for _ in p.iterdir()) if p.is_dir() else "-"
        print(f"    {p.name}  ({n} files)" if p.is_dir() else f"    {p.name}")

print("\nDone. 02_leaky_split_experiment.py's --raw-root should now find real images.")
