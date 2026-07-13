#!/usr/bin/env python3
"""
prepare_plantdoc_yolo.py
------------------------
Prepares the PlantDoc object-detection dataset for YOLO training, restricted to
the three apple leaf classes that carry bounding boxes:

    0  Apple_Scab_Leaf   (apple scab)
    1  Apple_rust_leaf   (cedar apple rust)
    2  Apple_leaf        (healthy apple leaf)

It clones the public PlantDoc detection repo, reads the Pascal-VOC XML boxes,
keeps only images that contain at least one apple box, converts the boxes to
YOLO format, and writes a ready-to-train dataset + data.yaml.

Usage:
    python prepare_plantdoc_yolo.py --out ./apple_det

The official PlantDoc TRAIN/ folder becomes the YOLO "train" split and the
official TEST/ folder becomes the YOLO "val" split (so results are reported on
the held-out PlantDoc test images, consistent with the rest of the thesis).
"""

import argparse
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_URL = "https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset.git"

# Canonical apple class names in PlantDoc (matched case-insensitively, spaces/underscores ignored).
APPLE_CLASSES = {
    "applescableaf": 0,   # Apple Scab Leaf
    "applerustleaf": 1,   # Apple rust leaf
    "appleleaf": 2,       # Apple leaf (healthy)
}
CLASS_NAMES = ["Apple_Scab_Leaf", "Apple_rust_leaf", "Apple_leaf"]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def norm(name: str) -> str:
    return "".join(name.lower().split()).replace("_", "")


def clone_repo(work_dir: Path) -> Path:
    repo_dir = work_dir / "PlantDoc-Object-Detection-Dataset"
    if repo_dir.exists():
        print(f"[ok] repo already present at {repo_dir}")
        return repo_dir
    print(f"[..] cloning {REPO_URL}")
    try:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)], check=True)
    except subprocess.CalledProcessError:
        print(
            "\n[error] git clone failed. Two easy fixes:\n"
            "  1) Download the repo ZIP manually from:\n"
            "       https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset\n"
            "     unzip it, and pass its folder with --repo_dir <path> (it must contain TRAIN/ and TEST/).\n"
            "  2) Or grab PlantDoc in YOLO format from Roboflow:\n"
            "       https://public.roboflow.com/object-detection/plantdoc\n"
            "     then skip this script and point train_eval on its data.yaml (keep only the 3 apple classes).\n"
        )
        raise
    return repo_dir


def find_image(xml_path: Path) -> Path | None:
    """Find the image file that matches an XML annotation."""
    stem = xml_path.stem
    for ext in IMG_EXTS:
        cand = xml_path.with_suffix(ext)
        if cand.exists():
            return cand
    # fall back: search same directory by stem
    for p in xml_path.parent.iterdir():
        if p.stem == stem and p.suffix in IMG_EXTS:
            return p
    return None


def convert_split(src_dir: Path, out_img: Path, out_lbl: Path) -> tuple[int, int, dict]:
    """Convert one PlantDoc folder (TRAIN or TEST) to YOLO. Returns (imgs, boxes, per-class counts)."""
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    n_imgs, n_boxes = 0, 0
    per_class = {name: 0 for name in CLASS_NAMES}

    xmls = sorted(src_dir.rglob("*.xml"))
    for xml_path in xmls:
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        size = root.find("size")
        if size is None:
            continue
        W = float(size.findtext("width") or 0)
        H = float(size.findtext("height") or 0)
        if W <= 0 or H <= 0:
            # some PlantDoc XMLs have 0 size; read from the image instead
            img_tmp = find_image(xml_path)
            if img_tmp is None:
                continue
            try:
                from PIL import Image
                W, H = Image.open(img_tmp).size
            except Exception:
                continue

        lines = []
        for obj in root.findall("object"):
            raw = obj.findtext("name") or ""
            key = norm(raw)
            if key not in APPLE_CLASSES:
                continue
            cid = APPLE_CLASSES[key]
            bb = obj.find("bndbox")
            if bb is None:
                continue
            xmin = max(0.0, float(bb.findtext("xmin")))
            ymin = max(0.0, float(bb.findtext("ymin")))
            xmax = min(W, float(bb.findtext("xmax")))
            ymax = min(H, float(bb.findtext("ymax")))
            if xmax <= xmin or ymax <= ymin:
                continue
            xc = (xmin + xmax) / 2.0 / W
            yc = (ymin + ymax) / 2.0 / H
            bw = (xmax - xmin) / W
            bh = (ymax - ymin) / H
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            per_class[CLASS_NAMES[cid]] += 1
            n_boxes += 1

        if not lines:
            continue  # skip images with no apple boxes

        img_path = find_image(xml_path)
        if img_path is None:
            continue
        # unique destination name (avoid collisions across subfolders)
        dst_stem = f"{img_path.parent.name}_{img_path.stem}"
        shutil.copy(img_path, out_img / f"{dst_stem}{img_path.suffix.lower()}")
        (out_lbl / f"{dst_stem}.txt").write_text("\n".join(lines) + "\n")
        n_imgs += 1

    return n_imgs, n_boxes, per_class


def locate_split_dirs(repo_dir: Path) -> tuple[Path, Path]:
    """Find TRAIN and TEST folders regardless of exact casing/nesting."""
    train_dir = test_dir = None
    for p in repo_dir.rglob("*"):
        if p.is_dir():
            if p.name.upper() == "TRAIN" and train_dir is None:
                train_dir = p
            elif p.name.upper() == "TEST" and test_dir is None:
                test_dir = p
    if train_dir is None or test_dir is None:
        # fallback: any dir containing xml files
        xml_dirs = sorted({x.parent for x in repo_dir.rglob("*.xml")})
        if len(xml_dirs) >= 2:
            train_dir, test_dir = xml_dirs[0], xml_dirs[1]
    if train_dir is None or test_dir is None:
        sys.exit("[error] could not locate TRAIN/TEST folders in the repo")
    return train_dir, test_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./apple_det", help="output dataset directory")
    ap.add_argument("--work", default="./_plantdoc_src", help="working dir for the cloned repo")
    ap.add_argument("--repo_dir", default=None,
                    help="path to an already-downloaded PlantDoc repo (must contain TRAIN/ and TEST/)")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)

    if args.repo_dir:
        repo_dir = Path(args.repo_dir).resolve()
        if not repo_dir.exists():
            sys.exit(f"[error] --repo_dir not found: {repo_dir}")
        print(f"[ok] using local repo at {repo_dir}")
    else:
        repo_dir = clone_repo(work)
    train_src, test_src = locate_split_dirs(repo_dir)
    print(f"[ok] TRAIN: {train_src}")
    print(f"[ok] TEST : {test_src}")

    if out.exists():
        shutil.rmtree(out)
    tr_imgs, tr_boxes, tr_pc = convert_split(train_src, out / "images/train", out / "labels/train")
    va_imgs, va_boxes, va_pc = convert_split(test_src, out / "images/val", out / "labels/val")

    # data.yaml
    yaml_txt = (
        f"path: {out}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    (out / "data.yaml").write_text(yaml_txt)

    print("\n================ SUMMARY ================")
    print(f"train images: {tr_imgs:4d}   boxes: {tr_boxes:4d}   {tr_pc}")
    print(f"val   images: {va_imgs:4d}   boxes: {va_boxes:4d}   {va_pc}")
    print(f"data.yaml -> {out/'data.yaml'}")
    print("========================================")
    if tr_imgs == 0 or va_imgs == 0:
        sys.exit("[error] no apple images found — check the repo structure")
    print("[done] dataset ready for YOLO training.")


if __name__ == "__main__":
    main()
