#!/usr/bin/env python3
"""
train_eval_yolo.py
------------------
Trains and evaluates a YOLO detector for apple-leaf disease *detection* on the
PlantDoc apple subset produced by prepare_plantdoc_yolo.py.

It reports the standard detection metrics (precision, recall, mAP@0.5,
mAP@0.5:0.95) overall and per class, and saves all figures Ultralytics
generates (results curves, confusion matrix, PR curve, validation batches).
A compact metrics summary is written to  runs/apple_yolo/metrics_summary.csv
and .json  so the numbers can be dropped straight into the thesis and paper.

Usage (defaults are sensible for a single GPU):
    python train_eval_yolo.py --data ./apple_det/data.yaml --model yolo11s.pt \
        --epochs 100 --imgsz 640 --batch 16

If yolo11s.pt is unavailable on your ultralytics version, use --model yolov8s.pt.
"""

import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./apple_det/data.yaml")
    ap.add_argument("--model", default="yolo11s.pt",
                    help="pretrained weights: yolo11s.pt (or yolov8s.pt as a fallback)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="apple_yolo")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=30)
    args = ap.parse_args()

    from ultralytics import YOLO
    import torch

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}")

    model = YOLO(args.model)

    # ---- train ----
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        patience=args.patience,
        project=args.project,
        name=args.name,
        pretrained=True,
        cos_lr=True,
        plots=True,          # save results.png, PR_curve, confusion_matrix, etc.
        verbose=True,
    )

    # ---- validate best weights on the val split ----
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        split="val",
        project=args.project,
        name=args.name + "_val",
        plots=True,
        verbose=True,
    )

    # ---- collect metrics ----
    names = metrics.names  # {id: name}
    box = metrics.box
    overall = {
        "precision(B)": float(box.mp),
        "recall(B)": float(box.mr),
        "mAP50(B)": float(box.map50),
        "mAP50-95(B)": float(box.map),
    }
    per_class = {}
    try:
        for i, c in enumerate(box.ap_class_index):
            cname = names[int(c)]
            p, r, ap50, ap = box.class_result(i)
            per_class[cname] = {
                "precision": float(p), "recall": float(r),
                "mAP50": float(ap50), "mAP50-95": float(ap),
            }
    except Exception as e:
        print(f"[warn] per-class extraction failed: {e}")

    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"overall": overall, "per_class": per_class,
               "model": args.model, "epochs": args.epochs, "imgsz": args.imgsz}
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))

    with (out_dir / "metrics_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "mAP@0.5", "mAP@0.5:0.95"])
        w.writerow(["ALL", f"{overall['precision(B)']:.4f}", f"{overall['recall(B)']:.4f}",
                    f"{overall['mAP50(B)']:.4f}", f"{overall['mAP50-95(B)']:.4f}"])
        for cname, m in per_class.items():
            w.writerow([cname, f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                        f"{m['mAP50']:.4f}", f"{m['mAP50-95']:.4f}"])

    # ---- a few sample detections for the thesis figure ----
    try:
        val_imgs = list((Path(args.data).resolve().parent / "images/val").glob("*"))[:12]
        if val_imgs:
            model.predict(val_imgs, imgsz=args.imgsz, conf=0.25, save=True,
                          project=args.project, name=args.name + "_preds", verbose=False)
    except Exception as e:
        print(f"[warn] sample prediction failed: {e}")

    print("\n================ RESULTS ================")
    print(f"Overall  P={overall['precision(B)']:.3f}  R={overall['recall(B)']:.3f}  "
          f"mAP50={overall['mAP50(B)']:.3f}  mAP50-95={overall['mAP50-95(B)']:.3f}")
    for cname, m in per_class.items():
        print(f"  {cname:18s} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"mAP50={m['mAP50']:.3f} mAP50-95={m['mAP50-95']:.3f}")
    print(f"\nSaved: {out_dir/'metrics_summary.csv'} and .json")
    print("Figures in:", out_dir, "and", str(out_dir) + "_val")
    print("========================================")


if __name__ == "__main__":
    main()
