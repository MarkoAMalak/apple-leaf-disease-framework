# Revision 2026 (SOIC manuscript 4492): code and results

This folder holds the code and results of the major revision of

> M. A. Malak, M. Thabet, A. S. Aziz, A. A. Alhelbawy, *Detection and Classification of Apple Diseases using ML Techniques: A Leakage-Free Deep-Learning Framework with Honest Cross-Dataset Evaluation*, Statistics, Optimization and Information Computing (SOIC), manuscript 4492.

The code and results of the original submission are still in `../appleleafdiseaseframework/`.

## Contents

| Folder | What it contains |
|---|---|
| `notebooks/SOIC_v2_colab.ipynb` | The main revision notebook. It is resumable after every epoch and was run on Google Colab (T4). Sections: leaky vs leakage-free training (9); data efficiency with 3 backbones and 2 fine-tuning budgets, plus PlantDoc (8, 8b); CORAL/DANN domain adaptation (10); GAN oversampling with a near-duplicate filter (11); augmentation ablation by transform family (12); five-seed in-domain sweep (13). |
| `notebooks/SOIC_revision_experiments_v2.ipynb` | The local notebook: sanity re-evaluation of the released checkpoints, zero-shot results for all backbones, bootstrap CIs, error analysis versus image properties, latency, t-SNE of the frogeye-to-black-rot mapping, Grad-CAM per class with an independent GrabCut mask, and the YOLOv8 vs YOLO11 comparison. |
| `scripts/` | Data preparation (`prepare_data.py`, `build_fgvc8_folders.py`, `fix_raw_leaky_folder_names.py`), the original short-budget data-efficiency script (`data_efficiency.py`), the Colab bundle builder, and `PATHS.md`. |
| `splits/` | Exact image lists for every split: the leakage-free PlantVillage train/val/test, the FGVC8 evaluation sample (700 per class), the FGVC8 data-efficiency pool and fixed test split, and the PlantDoc external set. Paths are relative to the dataset root. |
| `results/` | `ALL_RESULTS_SUMMARY.md` (every number reported in the revised manuscript), section summaries, per-image Grad-CAM scores, t-SNE statistics, latency, and sanity results. |
| `results/predictions/` | Per-image zero-shot predictions and class probabilities of the five released checkpoints on the PlantVillage test split, PlantDoc and FGVC8. |

Datasets, images and model weights are not included (see the licences of PlantVillage, PlantDoc and Plant Pathology 2021/FGVC8). The notebooks rebuild everything from the public datasets and the split files. Local paths such as `D:\co work\...` in the notebook configuration cells are only defaults; set `DATA_ROOT` and `RESULTS_ROOT` to your own folders.

## Key results of the revision

- **Leakage:**
  - The leaky image-level split reports 99.98% accuracy, and 100% even on clean test leaves, so the leak is invisible in the lab.
  - On external FGVC8 the leaky model is worse: 39.0% vs 43.9% (p = 0.04).
- **In-domain accuracy:** with five seeds all five backbones reach 99.35–99.80%, a statistical tie.
- **Data efficiency:**
  - Leakage-free lab initialization gives +0.21 macro-F1 with 80 field images under a short 15-epoch budget.
  - This holds for ResNet50, MobileNetV2 and DenseNet121.
  - There is no advantage when fine-tuning runs to convergence.
- **Domain adaptation:** DANN reaches 54.4% on FGVC8 without labels, against 37.2% source-only; fine-tuning on 68 labelled images reaches 76.7%.
- **GAN oversampling:**
  - In-domain it adds +0.3 points.
  - In the field it costs 4.3 points on FGVC8 zero-shot and halves cedar-rust recall.
- **Grad-CAM:** CAM-in-leaf is 86.2% on all 490 test images; the class ordering is confirmed with GrabCut, and no low score is caused by a mask failure.
- **Detection:** YOLO11s vs YOLOv8s mAP@0.5 is 0.929 vs 0.881 (3 seeds each).
