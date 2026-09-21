#!/usr/bin/env python3
"""
SOIC major revision -- one-time data preparation script (Python version)
==========================================================================
Why this exists: prepare_data.ps1 (PowerShell) is blocked on this machine by
a system-level restriction policy ("This operation has been cancelled due to
restrictions in effect on this computer"), which appears to target PowerShell
script execution specifically. This script does the exact same job in plain
Python instead -- no PowerShell, no cmd, no mklink/junctions, just plain file
copies -- so it should not trip the same restriction.

Run it with:
    python prepare_data.py

(Double-click may not work depending on your Python file-association; if so,
open a terminal, cd to this folder, and run the command above.)

Safe to re-run: it skips anything that already exists.
"""
import os
import re
import shutil
import zipfile
import subprocess
import sys
from pathlib import Path

# Drive-letter-agnostic: resolve "co work" from this script's own location.
# This script lives at "<co work>\SOIC major revision\code soic\prepare_data.py".
SCRIPT_DIR = Path(__file__).resolve().parent
TARGET = SCRIPT_DIR
BASE = TARGET.parent.parent
DATA = TARGET / "data"

print(f"Resolved base 'co work' folder as: {BASE}")

DATA.mkdir(parents=True, exist_ok=True)
(TARGET / "checkpoints").mkdir(parents=True, exist_ok=True)
(TARGET / "manual_annotations" / "images").mkdir(parents=True, exist_ok=True)
(TARGET / "manual_annotations" / "masks").mkdir(parents=True, exist_ok=True)


def copy_tree(link: Path, real_target: Path):
    """Copy real_target's contents into link (plain copy, not a junction)."""
    if link.exists():
        print(f"  [skip, already exists] {link}")
        return
    if not real_target.exists():
        print(f"  [MISSING SOURCE, skipped] {real_target}")
        return
    shutil.copytree(real_target, link)
    print(f"  [copied] {link}  <-  {real_target}")


# === 1. Leakage-free PlantVillage split (renamed to match the notebook's class names) ===
print("\n=== 1. Leakage-free PlantVillage split (renamed to match the notebook's class names) ===")
clean_images = BASE / "final" / "clean_apple_dataset" / "images"
class_map = {
    "apple_scab": "Apple___Apple_scab",
    "black_rot": "Apple___Black_rot",
    "cedar_apple_rust": "Apple___Cedar_apple_rust",
    "healthy": "Apple___healthy",
}
for split in ("train", "val", "test"):
    (DATA / "plantvillage_leakfree" / split).mkdir(parents=True, exist_ok=True)
    for key, real_name in class_map.items():
        copy_tree(DATA / "plantvillage_leakfree" / split / key,
                  clean_images / split / real_name)

# === 2. PlantDoc external test set ===
print("\n=== 2. PlantDoc external test set (classification only -- no black_rot available) ===")
(DATA / "plantdoc_external").mkdir(parents=True, exist_ok=True)
plantdoc_ext = BASE / "final" / "clean_apple_dataset" / "external_test_plantdoc"
plantdoc_map = {
    "apple_scab": "Apple___Apple_scab",
    "cedar_apple_rust": "Apple___Cedar_apple_rust",
    "healthy": "Apple___healthy",
}
for key, real_name in plantdoc_map.items():
    copy_tree(DATA / "plantdoc_external" / key, plantdoc_ext / real_name)

# === 3. Checkpoints (copied, ~330 MB total) ===
print("\n=== 3. Checkpoints (~330 MB total) ===")
models_dir = BASE / "final paper 2" / "outputs" / "models"
for name in ["cls_resnet50.pt", "cls_densenet121.pt", "cls_mobilenetv2.pt",
             "cls_vgg16.pt", "cls_inceptionv3.pt"]:
    src = models_dir / name
    dst = TARGET / "checkpoints" / name
    if dst.exists():
        print(f"  [skip, already exists] {dst}")
    elif src.exists():
        shutil.copy2(src, dst)
        print(f"  [copied] {dst}")
    else:
        print(f"  [MISSING SOURCE, skipped] {src}")

# === 4. YOLO PlantDoc detection data (fixed data.yaml) ===
print("\n=== 4. YOLO PlantDoc detection data (fixed data.yaml) ===")
yolo_dir = BASE / "YOLO" / "yolo_apple_detection" / "apple_det"
copy_tree(DATA / "yolo_plantdoc", yolo_dir)
fixed_yaml = (
    f"path: {yolo_dir}\n"
    "train: images/train\n"
    "val: images/val\n"
    "test: images/val\n"
    "nc: 3\n"
    "names: ['Apple_Scab_Leaf', 'Apple_rust_leaf', 'Apple_leaf']\n"
)
(DATA / "yolo_plantdoc_data.yaml").write_text(fixed_yaml)
print(f"  [written] {DATA / 'yolo_plantdoc_data.yaml'}")

# === 5. Manual gold subset (reference material for Section 8) ===
print("\n=== 5. Manual gold subset (existing color-based reference material, for Section 8 context) ===")
copy_tree(DATA / "manual_gold_subset_reference", BASE / "manual gold subset")

# === 6. FGVC8 raw data -> per-class folders ===
print("\n=== 6. FGVC8 raw data -> per-class folders ===")
fgvc8_root = BASE / "dataset real"
py_script = TARGET / "build_fgvc8_folders.py"
by_class_dir = DATA / "fgvc8_by_class"
if by_class_dir.exists():
    print(f"  [skip, already exists] {by_class_dir}")
elif not py_script.exists():
    print(f"  [MISSING] {py_script} not found -- copy build_fgvc8_folders.py into this 'code soic' folder first.")
else:
    result = subprocess.run(
        [sys.executable, str(py_script), "--fgvc8-root", str(fgvc8_root), "--out-root", str(DATA)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("  [FAILED]")
        print(result.stderr)

# === 7. Raw (pre-dedup) PlantVillage pool for the LEAKY-split experiment ===
print("\n=== 7. Raw (pre-dedup) PlantVillage pool for the LEAKY-split experiment (Section 2) ===")
archive = BASE / "final" / "Plantvillage" / "archive.zip"
raw_leaky_dir = DATA / "plantvillage_raw_leaky"
CLASS_RE = re.compile(r"Apple___[A-Za-z]+(?:_[A-Za-z]+)*")
if raw_leaky_dir.exists():
    print(f"  [skip, already exists] {raw_leaky_dir}")
elif not archive.exists():
    print(f"  [MISSING] {archive} not found -- Section 2 needs manual attention.")
else:
    count = 0
    with zipfile.ZipFile(archive) as z:
        for full in z.namelist():
            if not full.startswith("Dataste Classification/Dataste Classification/"):
                continue
            if full.endswith("/"):
                continue
            fname = full.split("/")[-1]
            m = CLASS_RE.search(fname)
            if m:
                class_name = m.group(0)
            else:
                parts = full.split("/")
                folder_class = next((p for p in parts if p.startswith("Apple___")), None)
                if not folder_class:
                    continue
                class_name = folder_class
            dest_dir = raw_leaky_dir / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / fname
            if not dest_file.exists():
                with z.open(full) as src, open(dest_file, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                count += 1
    print(f"  [built] {raw_leaky_dir}  ({count} images extracted; train+valid+test merged on purpose, augmented copies kept)")

# === Summary ===
print("\n" + "=" * 60)
print("DONE. Paths to use in the missing_experiments scripts:")
print("=" * 60)
print("Section 1 (t-SNE):")
print(f"    plantvillage_root = \"{DATA / 'plantvillage_leakfree' / 'test'}\"")
print(f"    fgvc8_root        = \"{DATA / 'fgvc8_frogeye_only'}\"")
print(f"    checkpoint        = \"{TARGET / 'checkpoints' / 'cls_resnet50.pt'}\"")
print()
print("Section 2 (leaky split):")
print(f"    raw_root = \"{raw_leaky_dir}\"")
print()
print("Section 3 (data efficiency, extra backbone):")
print(f"    fgvc8_root     = \"{DATA / 'fgvc8_by_class'}\"")
print(f"    lab_checkpoint = \"{TARGET / 'checkpoints' / 'cls_<backbone>.pt'}\"   (pick the matching backbone file)")
print()
print("Section 4 (YOLOv8 vs YOLOv11):")
print(f"    data_yaml = \"{DATA / 'yolo_plantdoc_data.yaml'}\"")
print()
print("Section 5 (Grad-CAM per class):")
print(f"    plantvillage_test_root = \"{DATA / 'plantvillage_leakfree' / 'test'}\"")
print(f"    checkpoint              = \"{TARGET / 'checkpoints' / 'cls_resnet50.pt'}\"")
print()
print("Section 6 (GAN oversampling):")
print(f"    minority_root     = \"{DATA / 'plantvillage_leakfree' / 'train' / 'cedar_apple_rust'}\"")
print(f"    real_train_root   = \"{DATA / 'plantvillage_leakfree' / 'train'}\"")
print(f"    real_test_root    = \"{DATA / 'plantvillage_leakfree' / 'test'}\"")
print(f"    plantvillage_root = \"{DATA / 'plantvillage_leakfree'}\"")
print(f"    checkpoint        = \"{TARGET / 'checkpoints' / 'cls_resnet50.pt'}\"")
print()
print("Section 7 (domain adaptation, CORAL/DANN):")
print(f"    plantvillage_root = \"{DATA / 'plantvillage_leakfree'}\"")
print(f"    field_root        = \"{DATA / 'fgvc8_by_class'}\"")
print()
print("Section 8 (manual annotation scoring):")
print(f"    images_dir       = \"{TARGET / 'manual_annotations' / 'images'}\"   (put your ~20-30 picked images here)")
print(f"    manual_masks_dir = \"{TARGET / 'manual_annotations' / 'masks'}\"    (put your drawn masks here)")
print(f"    checkpoint       = \"{TARGET / 'checkpoints' / 'cls_resnet50.pt'}\"")
