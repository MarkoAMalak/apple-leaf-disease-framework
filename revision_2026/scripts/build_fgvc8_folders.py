"""
build_fgvc8_folders.py
=========================
Converts the raw FGVC8 (Plant Pathology 2021 - Kaggle) CSV format into
ImageFolder-style class folders, in TWO layouts, for the SOIC
missing-experiments notebook:

  1. fgvc8_by_class/{apple_scab,black_rot,cedar_apple_rust,healthy}/*.jpg
     -- the full 4-class taxonomy used by Sections 3 and 7. FGVC8 has no
     true "black_rot" label, so -- exactly as already discussed in the
     manuscript's Section 2.6 -- frog_eye_leaf_spot is used as the closest
     visual proxy for that class.

  2. fgvc8_frogeye_only/frogeye_leaf_spot/*.jpg
     -- just the same frog_eye_leaf_spot images, under their real name,
     for Section 1's t-SNE code (which looks for exactly that folder name).

Only single-label rows are used (skips multi-label rows like
"scab frog_eye_leaf_spot complex") so each class folder is visually pure.
Uses hardlinks when possible (instant, zero extra disk space on the same
NTFS volume); falls back to a normal copy otherwise.

Usage:
    python build_fgvc8_folders.py ^
        --fgvc8-root "H:\\Master\\co work\\dataset real" ^
        --out-root "H:\\Master\\co work\\SOIC major revision\\code soic\\data"
"""
import argparse
import csv
import os
import shutil

LABEL_MAP = {
    "scab": "apple_scab",
    "rust": "cedar_apple_rust",
    "healthy": "healthy",
    "frog_eye_leaf_spot": "black_rot",  # proxy -- FGVC8 has no black_rot; see manuscript Sec 2.6
}


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)  # instant, zero extra disk space on the same NTFS volume
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fgvc8-root", required=True, help="folder containing train_images/ and train.csv")
    ap.add_argument("--out-root", required=True, help="the notebook's data/ folder")
    args = ap.parse_args()

    csv_path = os.path.join(args.fgvc8_root, "train.csv")
    images_dir = os.path.join(args.fgvc8_root, "train_images")
    by_class_dir = os.path.join(args.out_root, "fgvc8_by_class")
    frogeye_dir = os.path.join(args.out_root, "fgvc8_frogeye_only", "frogeye_leaf_spot")
    os.makedirs(frogeye_dir, exist_ok=True)

    counts = {v: 0 for v in LABEL_MAP.values()}
    n_frogeye = 0
    skipped_multilabel = 0
    skipped_other = 0
    skipped_missing_file = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["labels"].strip()
            if " " in label:
                skipped_multilabel += 1
                continue
            if label not in LABEL_MAP:
                skipped_other += 1
                continue

            src = os.path.join(images_dir, row["image"])
            if not os.path.exists(src):
                skipped_missing_file += 1
                continue

            target_class = LABEL_MAP[label]
            class_dir = os.path.join(by_class_dir, target_class)
            os.makedirs(class_dir, exist_ok=True)
            dst = os.path.join(class_dir, row["image"])
            was_new = not os.path.exists(dst)
            link_or_copy(src, dst)
            if was_new:
                counts[target_class] += 1

            if label == "frog_eye_leaf_spot":
                dst2 = os.path.join(frogeye_dir, row["image"])
                was_new2 = not os.path.exists(dst2)
                link_or_copy(src, dst2)
                if was_new2:
                    n_frogeye += 1

    print("Built fgvc8_by_class at", by_class_dir)
    for k, v in counts.items():
        print(f"  {k}: {v} images")
    print(f"\nBuilt fgvc8_frogeye_only at {frogeye_dir}")
    print(f"  frogeye_leaf_spot: {n_frogeye} images")
    print(f"\nSkipped {skipped_multilabel} multi-label rows, {skipped_other} other-label rows "
          f"(complex/powdery_mildew, not part of the 4-class taxonomy), "
          f"{skipped_missing_file} rows whose image file was not found.")
    print("\nNote: 'black_rot' here is really frog_eye_leaf_spot used as a visual proxy -- "
          "FGVC8 has no true black_rot label. This matches the manuscript's own Section 2.6 discussion.")


if __name__ == "__main__":
    main()
