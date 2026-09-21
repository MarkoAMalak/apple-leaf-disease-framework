# Apple Leaf Disease Framework

[![DOI](https://zenodo.org/badge/1298983948.svg)](https://doi.org/10.5281/zenodo.21336556)

This repository holds the code, split files and results for the paper *Detection and Classification of Apple Diseases using ML Techniques: A Leakage-Free Deep-Learning Framework with Honest Cross-Dataset Evaluation* (M. A. Malak, M. Thabet, A. S. Aziz, A. A. Alhelbawy; Statistics, Optimization and Information Computing, manuscript 4492). The framework is a leakage-free deep-learning pipeline for apple-leaf disease classification and detection. It separates honest zero-shot cross-dataset evaluation from cross-validation, and adds YOLO detection, unsupervised domain-adaptation baselines and a data-efficiency study. That study shows leakage-free pre-training makes field adaptation faster under a short fine-tuning budget.

## Repository structure

- **`revision_2026/`** is the major revision (latest). It contains:
  - the notebooks and scripts;
  - the exact split files;
  - per-image predictions;
  - `ALL_RESULTS_SUMMARY.md`, which has every number in the revised manuscript.

  See [`revision_2026/README.md`](revision_2026/README.md).
- **`appleleafdiseaseframework/`** holds the code, notebooks and results of the original submission.
- **`PUSH_INSTRUCTIONS.md`** has notes for publishing the repository and archiving releases on Zenodo.

## Key results (revision)

- **Leakage:**
  - The leak cannot be seen in the lab: the leaky split reports 99.98% accuracy, and 100% on test leaves never seen in training.
  - On external FGVC8 the leaky model is less accurate than the leakage-free one: 39.0% vs 43.9% (p = 0.04).
- **In-domain accuracy:** five backbones reach 99.35–99.80% over five seeds, a statistical tie. MobileNetV2 is 5–10× faster than the larger models.
- **Data efficiency:**
  - Leakage-free lab pre-training gives +0.21 macro-F1 with 80 field images under a short 15-epoch budget. This holds for three backbones and on PlantDoc.
  - There is no advantage once fine-tuning runs to convergence.
- **Domain adaptation:** DANN reaches 54.4% on FGVC8 without field labels. Fine-tuning on 68 labelled field images reaches 76.7%.
- **GAN oversampling:** synthetic minority-class images add in-domain accuracy but lower field accuracy.
- **Detection:** YOLO11s outperforms YOLOv8s (mAP@0.5 0.929 vs 0.881, 3 seeds each).

## Citation and archive

Every GitHub release is archived on Zenodo. The concept DOI [10.5281/zenodo.21336556](https://doi.org/10.5281/zenodo.21336556) always resolves to the latest version.

Datasets and model weights are not included. PlantVillage, PlantDoc and Plant Pathology 2021 (FGVC8) are public; prepare them locally with the scripts provided.

## Topics

deep-learning, computer-vision, pytorch, plant-disease, apple-leaf-disease, image-classification, object-detection, yolo, yolov11, transfer-learning, data-leakage, domain-adaptation, reproducible-research, agriculture, plantvillage
