#!/usr/bin/env python3
"""
build_colab_bundle.py (v2 -- shrinks images so the bundle fits without Google Drive)
=======================================================================================
Builds ONE zip file with everything needed to run the remaining
missing-experiments scripts (01, 02, 03, 04, 06, 07) on Google Colab's free GPU:
the already-fixed scripts + common.py, the checkpoints, and every data folder
those scripts need.

WHY v2: the first version copied every image byte-for-byte and came out to
15.2 GB -- bigger than a free Google Drive account's entire 15 GB quota. But
every one of these scripts resizes images down to 224x224 (299x299 for
InceptionV3, 640x640 for the YOLO detector) before using them anyway -- so
shipping full-resolution originals was pure waste. This version downscales
every image (longer side capped at 640px, matching the YOLO input size, well
above what the classifiers need) while building the zip. Numerically this
changes nothing that matters: the training pipeline was going to shrink these
images that much regardless.

For the YOLO detection data specifically: resizing is safe because YOLO label
files store box coordinates as FRACTIONS of image width/height (0-1), not
absolute pixels -- a uniform resize (no cropping, aspect ratio preserved)
leaves every label file still exactly correct. Label .txt files themselves are
copied untouched.

You do NOT need to fix or touch any code -- this just packages what already
works on this machine, shrunk for transport. Run it ONCE, then follow
SOIC_colab_notebook.ipynb.

Run this from the "code soic" folder (same folder as prepare_data.py):
    python build_colab_bundle.py

Requires Pillow (install if missing: pip install Pillow).

Output: colab_bundle.zip in this same folder. Should now be small enough to
upload directly into the Colab session itself (see the notebook's cell 1B) --
no Google Drive space needed at all. If it's still uncomfortably large, lower
MAX_SIDE below (e.g. 448) and rerun; it's safe to re-run any time.
"""
import io
import zipfile
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow is required for this version. Install it with:\n    pip install Pillow")

BASE = Path(__file__).resolve().parent  # "code soic"
OUT_ZIP = BASE / "colab_bundle.zip"

MAX_SIDE = 640          # longer image side, in pixels, after resizing (matches YOLO's imgsz)
JPEG_QUALITY = 88
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# plain copy -- included byte-for-byte (non-image files), or resized (image files)
PLAIN_INCLUDE = [
    BASE / "missing_experiments",          # code -- never resized (not images)
    BASE / "checkpoints",                  # .pt files -- never resized
    BASE / "data" / "plantvillage_leakfree",
    BASE / "data" / "fgvc8_by_class",
    BASE / "data" / "fgvc8_frogeye_only",
    BASE / "data" / "yolo_plantdoc",       # images resized; label .txt files untouched
]

RAW_LEAKY_RENAME = {
    "Apple___Apple_scab": "apple_scab",
    "Apple___Black_rot": "black_rot",
    "Apple___Cedar_apple_rust": "cedar_apple_rust",
    "Apple___healthy": "healthy",
}

YOLO_YAML_COLAB = (
    "path: /content/soic_bundle/data/yolo_plantdoc\n"
    "train: images/train\n"
    "val: images/val\n"
    "test: images/val\n"
    "nc: 3\n"
    "names: ['Apple_Scab_Leaf', 'Apple_rust_leaf', 'Apple_leaf']\n"
)

totals = {"orig_bytes": 0, "zip_bytes": 0, "images": 0, "other_files": 0}


def resized_image_bytes(path: Path) -> bytes:
    """Downscale one image (longer side -> MAX_SIDE) and return re-encoded bytes.
    Falls back to the raw file bytes if PIL can't read/re-save it for any reason."""
    try:
        im = Image.open(path)
        im.load()
        fmt = (im.format or "JPEG").upper()
        if fmt not in ("JPEG", "PNG", "BMP"):
            fmt = "JPEG"
        w, h = im.size
        if max(w, h) > MAX_SIDE:
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        buf = io.BytesIO()
        if fmt == "JPEG":
            if im.mode in ("RGBA", "P", "LA"):
                im = im.convert("RGB")
            im.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        else:
            im.save(buf, format=fmt)
        return buf.getvalue()
    except Exception as e:
        print(f"    [could not resize, copied as-is] {path.name}: {e}")
        return path.read_bytes()


def add_path(zf, path: Path, arcname: str):
    orig_size = path.stat().st_size
    totals["orig_bytes"] += orig_size
    if path.suffix.lower() in IMAGE_EXTS:
        data = resized_image_bytes(path)
        totals["images"] += 1
    else:
        data = path.read_bytes()
        totals["other_files"] += 1
    zf.writestr(arcname, data)
    totals["zip_bytes"] += len(data)


def add_plain(zf, path: Path):
    if path.is_file():
        add_path(zf, path, str(path.relative_to(BASE)))
    elif path.is_dir():
        n = 0
        for p in path.rglob("*"):
            if p.is_file():
                add_path(zf, p, str(p.relative_to(BASE)))
                n += 1
        print(f"    ({n} files)")
    else:
        print(f"    [MISSING, skipped] {path}")


def add_raw_leaky_renamed(zf, raw_leaky_dir: Path):
    if not raw_leaky_dir.exists():
        print(f"    [MISSING, skipped] {raw_leaky_dir}")
        return
    n = 0
    for p in raw_leaky_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(raw_leaky_dir)
        parts = list(rel.parts)
        if parts and parts[0] in RAW_LEAKY_RENAME:
            parts[0] = RAW_LEAKY_RENAME[parts[0]]
        arcname = "data/plantvillage_raw_leaky/" + "/".join(parts)
        add_path(zf, p, arcname)
        n += 1
    print(f"    ({n} files, renamed to apple_scab/black_rot/cedar_apple_rust/healthy)")


print(f"Building {OUT_ZIP}  (images capped at {MAX_SIDE}px longer side) ...")
with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for item in PLAIN_INCLUDE:
        print(f"  adding {item.relative_to(BASE)} ...")
        add_plain(zf, item)

    print("  adding data/plantvillage_raw_leaky (renaming class folders) ...")
    add_raw_leaky_renamed(zf, BASE / "data" / "plantvillage_raw_leaky")

    print("  writing data/yolo_plantdoc_data_colab.yaml (Colab-path version) ...")
    zf.writestr("data/yolo_plantdoc_data_colab.yaml", YOLO_YAML_COLAB)

orig_mb = totals["orig_bytes"] / (1024 * 1024)
zip_mb = OUT_ZIP.stat().st_size / (1024 * 1024)
print(f"\nDone: {OUT_ZIP}")
print(f"  {totals['images']} images resized, {totals['other_files']} other files copied as-is")
print(f"  source data on disk:  {orig_mb:,.0f} MB")
print(f"  final zip:            {zip_mb:,.0f} MB")
if zip_mb > 1500:
    print(f"\n  Still fairly large. Open this script, lower MAX_SIDE (currently {MAX_SIDE}) to e.g. 448, and rerun.")
print("\nNext: open SOIC_colab_notebook.ipynb in Google Colab and use its direct-upload")
print("cell (1B) to upload colab_bundle.zip straight into the Colab session -- no Google")
print("Drive space needed.")
