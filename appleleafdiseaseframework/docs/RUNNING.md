# Running every experiment

All commands assume the repository root and an activated environment
(`pip install -r requirements.txt`). A CUDA GPU is recommended.

---

## 1. Object detection (YOLOv11, PlantDoc apple boxes)

Reproduces Table (detection) and the detection figures.

```bash
cd detection
python prepare_plantdoc_yolo.py --out ./apple_det
python train_eval_yolo.py --data ./apple_det/data.yaml --model yolo11s.pt \
    --epochs 100 --imgsz 640 --batch 16 --name apple_yolo
```

Outputs: `runs/apple_yolo/metrics_summary.csv` (per-class + overall mAP), plus the
Ultralytics figures (results curves, confusion matrix, PR curve, sample predictions).
Test-time augmentation:

```bash
yolo val model=runs/apple_yolo/weights/best.pt data=apple_det/data.yaml imgsz=640 augment=True
```

---

## 2. Data-efficiency and domain adaptation (FGVC8 field set)

Reproduces the data-efficiency table and curve (ImageNet-init vs leakage-free-lab-init).

**Prerequisites**

* The FGVC8 four-class field set as an `ImageFolder` (one subfolder per class):
  `apple_scab/`, `black_rot/` (or `frogeye_leaf_spot/`), `cedar_apple_rust/`, `healthy/`.
* Your leakage-free PlantVillage-trained ResNet50 checkpoint, saved with
  `torch.save(model.state_dict(), "resnet50_plantvillage.pt")`.

```bash
cd data_efficiency
python data_efficiency.py \
    --field_dir /path/to/FGVC8_4class \
    --lab_ckpt  /path/to/resnet50_plantvillage.pt \
    --fractions 5,10,25,50,100 --seeds 3 --epochs 15 --out runs_de_full
```

Outputs: `runs_de_full/data_efficiency.csv` (macro-F1 mean ± SD per fraction and
initialization), `data_efficiency_curve.png`, `raw_results.json`.

### 2b. Second confirmation on PlantDoc (recommended for the camera-ready)

Repeat the same study on the three-class PlantDoc field set to show the result is not
specific to one dataset:

```bash
python data_efficiency.py \
    --field_dir /path/to/PlantDoc_3class \
    --lab_ckpt  /path/to/resnet50_plantvillage.pt \
    --fractions 10,25,50,100 --seeds 3 --epochs 15 --out runs_de_plantdoc
```

(5% is omitted because PlantDoc is small; adjust fractions to the images available.)

---

## 3. Notes on honest evaluation

* **Zero-shot vs cross-validation** must be reported separately. The zero-shot number
  uses *no* field image in training and is the only genuine generalization measure; the
  cross-validation number mixes field data into training and is an adaptation result.
* **frogeye leaf spot → black rot.** FGVC8 contains no separately labelled apple black
  rot; the black-rot field result is obtained by mapping frogeye leaf spot (the foliar
  phase of the same fungus, *Botryosphaeria obtusa*) to black rot. Report this mapping
  explicitly so readers can judge it; the classification metrics are unchanged by the
  label name.
