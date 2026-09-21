---
# SOIC 4492 revision - experiment results

(Transcribed from Google Drive: soic_v2_result/ALL_RESULTS_SUMMARY.md, generated 2026-09-21 17:13 (cuda), FAST_DEV_RUN=False.
The complete result folder with per-run JSON files, checkpoints and figures is soic_v2_result on Google Drive.)

## 3.2 sanity + zero-shot all backbones

Paper checkpoints re-evaluated (paper: PlantDoc zero-shot 41.46%, FGVC8 32.86% for ResNet50).
'PD 3-way' = argmax restricted to the three PlantDoc classes.

| backbone | PV test acc % | paper % | PlantDoc ZS % | PD 3-way % | FGVC8 ZS % | FGVC8 macro-F1 |
|---|---|---|---|---|---|---|
| resnet50 | 99.80 | 99.86 | 62.37 | 68.99 | 33.36 | 0.271 |
| vgg16 | 99.80 | 99.86 | 43.90 | 56.45 | 32.39 | 0.215 |
| densenet121 | 99.59 | 99.73 | 47.04 | 54.36 | 29.57 | 0.198 |
| inception_v3 | 99.39 | 99.73 | 49.48 | 57.49 | 33.29 | 0.253 |
| mobilenet_v2 | 99.80 | 99.12 | 37.98 | 72.82 | 38.36 | 0.312 |

## 3.3 bootstrap 95% CI (ResNet50 zero-shot)

| dataset | n | accuracy % [95% CI] | macro-F1 [95% CI] |
|---|---|---|---|
| plantdoc | 287 | 62.4 [56.4, 67.9] | 0.676 [0.617, 0.725] |
| fgvc8 | 2800 | 33.4 [31.6, 35.1] | 0.271 [0.255, 0.286] |

## 3.4 zero-shot accuracy vs image properties (ResNet50)

Accuracy (%) in the low / middle / high tercile of each property; Mann-Whitney p compares correct vs wrong images.

| dataset | property | low | mid | high | median (correct) | median (wrong) | p |
|---|---|---|---|---|---|---|---|
| plantdoc | brightness | 57.3 | 55.8 | 74.0 | 151 | 137 | 0.00271 |
| plantdoc | sharpness | 61.5 | 63.2 | 62.5 | 631 | 704 | 0.921 |
| plantdoc | vegetation_fraction | 80.2 | 54.7 | 52.1 | 0.569 | 0.707 | 0.000482 |
| fgvc8 | brightness | 38.3 | 31.5 | 30.3 | 160 | 163 | 0.000698 |
| fgvc8 | sharpness | 40.8 | 34.9 | 24.3 | 289 | 346 | 4.6e-14 |
| fgvc8 | vegetation_fraction | 27.3 | 29.2 | 43.5 | 0.95 | 0.93 | 1.81e-13 |

## 3.5 latency on cuda

{"device": "cuda", "gpu": "NVIDIA GeForce 930MX", "cpu": "Intel64 Family 6 Model 142 Stepping 9, GenuineIntel", "threads": 2, "torch": "2.1.0+cu118", "amp": true, "protocol": "batch 1, 3 warm-up + 20 timed passes, synchronize before each timestamp"}

| backbone | params (M) | forward only (ms) | incl. load+resize (ms) |
|---|---|---|---|
| resnet50 | 23.52 | 50.8 ± 15.2 | 53.2 ± 10.4 |
| vgg16 | 27.56 | 109.7 ± 0.3 | 113.2 ± 0.7 |
| densenet121 | 7.22 | 89.4 ± 3.3 | 98.6 ± 21.1 |
| inception_v3 | 22.31 | 89.2 ± 0.2 | 93.7 ± 0.9 |
| mobilenet_v2 | 2.39 | 10.4 ± 0.3 | 14.9 ± 1.4 |

## 4.1 frogeye mapping in feature space

**finetuned_leakfree** (mean cosine similarity to each PlantVillage class centroid)

| field class | PV Apple scab | PV Black rot | PV Cedar apple rust | PV Healthy | nearest-centroid = own class | most common nearest centroid | 10-NN = own class |
|---|---|---|---|---|---|---|---|
| FGVC8 Apple scab | 0.823 | 0.744 | 0.674 | 0.589 | 83% | Apple scab | 77% |
| FGVC8 Black rot (= frogeye) | 0.804 | 0.782 | 0.699 | 0.590 | 38% | Apple scab | 42% |
| FGVC8 Cedar apple rust | 0.782 | 0.779 | 0.717 | 0.616 | 10% | Black rot | 10% |
| FGVC8 Healthy | 0.789 | 0.750 | 0.670 | 0.612 | 0% | Apple scab | 2% |

**imagenet_only** (mean cosine similarity to each PlantVillage class centroid)

| field class | PV Apple scab | PV Black rot | PV Cedar apple rust | PV Healthy | nearest-centroid = own class | most common nearest centroid | 10-NN = own class |
|---|---|---|---|---|---|---|---|
| FGVC8 Apple scab | 0.600 | 0.533 | 0.554 | 0.542 | 98% | Apple scab | 52% |
| FGVC8 Black rot (= frogeye) | 0.578 | 0.541 | 0.536 | 0.516 | 4% | Apple scab | 3% |
| FGVC8 Cedar apple rust | 0.581 | 0.540 | 0.546 | 0.525 | 7% | Apple scab | 7% |
| FGVC8 Healthy | 0.572 | 0.507 | 0.519 | 0.539 | 11% | Apple scab | 55% |

## 5.1 CAM-in-leaf per class

CAM-in-leaf (%) with the paper's HSV mask and with an independent GrabCut mask (n = test images).
'low' = HSV score < 0.60; of those, 'mask failure' = GrabCut score >= 0.80, 'off-leaf' = GrabCut score < 0.60.

| class | n | CAM-in-leaf HSV | CAM-in-leaf GrabCut | IoU(HSV, GrabCut) | low | mask failure | off-leaf |
|---|---|---|---|---|---|---|---|
| Apple scab | 95 | 96.9 ± 4.7 | 74.2 ± 6.4 | 0.55 ± 0.12 | 0 | 0 | 0 |
| Black rot | 94 | 91.4 ± 8.4 | 69.7 ± 6.5 | 0.58 ± 0.13 | 0 | 0 | 0 |
| Cedar apple rust | 47 | 67.7 ± 14.7 | 57.3 ± 8.3 | 0.76 ± 0.14 | 12 | 0 | 12 |
| Healthy | 254 | 83.8 ± 15.6 | 69.7 ± 7.5 | 0.62 ± 0.21 | 16 | 0 | 11 |
| all (pooled) | 490 | 86.2 ± 15.1 | 69.4 ± 8.4 | 0.61 ± 0.18 | 28 | 0 | 23 |

## 7.1 YOLOv8s vs YOLO11s (30 epochs, imported)

PlantDoc apple test split (29 images, 34 boxes); mean ± SD over seeds.

| model | eval | seeds | precision | recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|
| yolov8s | base | 3 | 0.802 ± 0.030 | 0.853 ± 0.016 | 0.879 ± 0.004 | 0.688 ± 0.012 |
| yolov8s | tta | 3 | 0.837 ± 0.066 | 0.859 ± 0.031 | 0.881 ± 0.009 | 0.693 ± 0.028 |
| yolo11s | base | 3 | 0.839 ± 0.082 | 0.873 ± 0.041 | 0.929 ± 0.017 | 0.722 ± 0.015 |
| yolo11s | tta | 3 | 0.847 ± 0.025 | 0.903 ± 0.023 | 0.929 ± 0.005 | 0.726 ± 0.019 |

## 8b data efficiency: short vs full protocol

**resnet50: SHORT budget (15 epochs, original data_efficiency.py protocol)** (FGVC8 field test n=400; macro-F1 mean ± SD over seeds)

| field data (n train) | ImageNet | Lab (leak-free) | Leaky lab | gain lab-ImageNet |
|---|---|---|---|---|
| 5% (80) | 0.599 ± 0.047 | 0.808 ± 0.021 | 0.665 ± 0.053 | +0.209 |
| 10% (160) | 0.810 ± 0.014 | 0.895 ± 0.018 | 0.893 ± 0.015 | +0.084 |
| 25% (400) | 0.912 ± 0.007 | 0.932 ± 0.007 | 0.943 ± 0.004 | +0.020 |
| 50% (800) | 0.935 ± 0.008 | 0.947 ± 0.018 | 0.963 ± 0.012 | +0.013 |
| 100% (1600) | 0.959 ± 0.005 | 0.961 ± 0.001 | 0.964 ± 0.001 | +0.002 |

- all fractions: lab > imagenet in 13/15 pairs, one-sided p=0.000214, two-sided p=0.000427
- 5-10% only: lab > imagenet in 6/6 pairs, one-sided p=0.0156, two-sided p=0.0312
- lab vs leaky: lab > leaky in 4/15 pairs, one-sided p=0.681, two-sided p=0.679

**resnet50: FULL paper protocol (two-phase + early stopping, Section 8)** (FGVC8 field test n=400; macro-F1 mean ± SD over seeds)

| field data (n train) | ImageNet | Lab (leak-free) | Leaky lab | gain lab-ImageNet |
|---|---|---|---|---|
| 5% (68) | 0.751 ± 0.082 | 0.762 ± 0.030 | 0.708 ± 0.102 | +0.011 |
| 10% (136) | 0.869 ± 0.008 | 0.868 ± 0.029 | 0.877 ± 0.030 | -0.001 |
| 25% (340) | 0.918 ± 0.030 | 0.916 ± 0.014 | 0.913 ± 0.035 | -0.002 |
| 50% (680) | 0.941 ± 0.013 | 0.941 ± 0.016 | 0.943 ± 0.012 | -0.000 |
| 100% (1360) | 0.969 ± 0.005 | 0.949 ± 0.014 | 0.964 ± 0.010 | -0.021 |

- all fractions: lab > imagenet in 6/15 pairs, one-sided p=0.661, two-sided p=0.72
- 5-10% only: lab > imagenet in 4/6 pairs, one-sided p=0.422, two-sided p=0.844
- lab vs leaky: lab > leaky in 6/15 pairs, one-sided p=0.661, two-sided p=0.72

**mobilenet_v2: SHORT budget** (FGVC8 field test n=400; macro-F1 mean ± SD over seeds)

| field data (n train) | ImageNet | Lab (leak-free) | gain lab-ImageNet |
|---|---|---|---|
| 5% (80) | 0.739 ± 0.019 | 0.784 ± 0.019 | +0.045 |
| 10% (160) | 0.873 ± 0.009 | 0.891 ± 0.006 | +0.017 |
| 25% (400) | 0.922 ± 0.009 | 0.935 ± 0.002 | +0.014 |
| 50% (800) | 0.941 ± 0.004 | 0.945 ± 0.004 | +0.003 |
| 100% (1600) | 0.957 ± 0.005 | 0.952 ± 0.005 | -0.005 |

- all fractions: lab > imagenet in 13/15 pairs, one-sided p=0.00418, two-sided p=0.00836
- 5-10% only: lab > imagenet in 6/6 pairs, one-sided p=0.0156, two-sided p=0.0312

**mobilenet_v2: FULL paper protocol**

| field data (n train) | ImageNet | Lab (leak-free) | gain lab-ImageNet |
|---|---|---|---|
| 5% (68) | 0.759 ± 0.022 | 0.769 ± 0.018 | +0.009 |
| 10% (136) | 0.841 ± 0.021 | 0.810 ± 0.045 | -0.032 |
| 25% (340) | 0.922 ± 0.009 | 0.916 ± 0.008 | -0.007 |
| 50% (680) | 0.935 ± 0.011 | 0.928 ± 0.016 | -0.007 |
| 100% (1360) | 0.962 ± 0.012 | 0.941 ± 0.018 | -0.021 |

- all fractions: lab > imagenet in 6/15 pairs, one-sided p=0.932, two-sided p=0.151
- 5-10% only: lab > imagenet in 3/6 pairs, one-sided p=0.719, two-sided p=0.688

**densenet121: SHORT budget**

| field data (n train) | ImageNet | Lab (leak-free) | gain lab-ImageNet |
|---|---|---|---|
| 5% (80) | 0.666 ± 0.022 | 0.732 ± 0.056 | +0.066 |
| 10% (160) | 0.848 ± 0.016 | 0.875 ± 0.027 | +0.027 |
| 25% (400) | 0.935 ± 0.016 | 0.935 ± 0.012 | +0.001 |
| 50% (800) | 0.950 ± 0.006 | 0.955 ± 0.004 | +0.004 |
| 100% (1600) | 0.955 ± 0.003 | 0.958 ± 0.005 | +0.003 |

- all fractions: lab > imagenet in 12/15 pairs, one-sided p=0.00513, two-sided p=0.0103
- 5-10% only: lab > imagenet in 6/6 pairs, one-sided p=0.0156, two-sided p=0.0312

**densenet121: FULL paper protocol**

| field data (n train) | ImageNet | Lab (leak-free) | gain lab-ImageNet |
|---|---|---|---|
| 5% (68) | 0.780 ± 0.065 | 0.788 ± 0.035 | +0.009 |
| 10% (136) | 0.856 ± 0.021 | 0.827 ± 0.019 | -0.029 |
| 25% (340) | 0.917 ± 0.011 | 0.919 ± 0.016 | +0.002 |
| 50% (680) | 0.938 ± 0.016 | 0.946 ± 0.014 | +0.008 |
| 100% (1360) | 0.963 ± 0.010 | 0.957 ± 0.016 | -0.006 |

- all fractions: lab > imagenet in 6/15 pairs, one-sided p=0.681, two-sided p=0.679
- 5-10% only: lab > imagenet in 2/6 pairs, one-sided p=0.781, two-sided p=0.562

**PlantDoc (Table 8), ResNet50, short protocol** (test n=58, macro-F1 over 3 classes)

| field data (n train) | ImageNet | Lab (leak-free) | gain lab-ImageNet |
|---|---|---|---|
| 10% (22) | 0.510 ± 0.077 | 0.697 ± 0.023 | +0.187 |
| 25% (57) | 0.539 ± 0.032 | 0.733 ± 0.039 | +0.194 |
| 50% (114) | 0.753 ± 0.049 | 0.812 ± 0.036 | +0.058 |
| 100% (229) | 0.886 ± 0.019 | 0.839 ± 0.012 | -0.047 |

- lab > imagenet in 8/11 pairs, one-sided p=0.00684, two-sided p=0.0137
- (lab 100% has two seeds; the third run, s2, did not complete)

Note: the FULL protocol trains on 85% of each subset (15% is its early-stopping validation split); the SHORT protocol trains on the whole subset, exactly like the original script.

## 8.2 data efficiency (full protocol, images-to-target)

- resnet50: macro-F1 0.85: imagenet ~125 img, lab ~124 img, leaky ~125 img; 0.90: ~263 / ~270 / ~267; 0.95: ~897 / not reached / ~908
- mobilenet_v2: 0.85: imagenet ~157, lab ~213; 0.90: ~284 / ~310; 0.95: ~1064 / not reached
- densenet121: 0.85: imagenet ~131, lab ~187; 0.90: ~284 / ~299; 0.95: ~1007 / ~932

## 9.2 leaky vs honest

ResNet50, identical training code. 'reported' = accuracy on the model's own test split; 'contaminated/clean' = test images whose physical leaf is / is not in the training split.

| model | seeds | reported test acc % | test images contaminated % | acc contaminated % | acc clean % | PlantDoc ZS % | FGVC8 ZS % | FGVC8 ZS macro-F1 |
|---|---|---|---|---|---|---|---|---|
| leaky | 3 | 99.98 ± 0.04 | 84.9 ± 0.9 | 99.97 ± 0.05 | 100.00 ± 0.00 | 56.79 ± 1.60 | 39.04 ± 1.91 | 0.362 ± 0.035 |
| honest | 3 | 99.66 ± 0.24 | 0.0 ± 0.0 | - | 99.66 ± 0.24 | 59.47 ± 5.68 | 43.94 ± 0.46 | 0.421 ± 0.010 |

## 10 domain adaptation vs fine-tuning

Same FGVC8 field test split (n=400).

| method | labelled field images | seeds | field acc % | field macro-F1 | PV test acc % | PlantDoc acc % |
|---|---|---|---|---|---|---|
| source_only | 0 (unlabelled pool) | 3 | 37.17 ± 2.98 | 0.337 ± 0.022 | 99.93 ± 0.12 | 55.40 ± 9.04 |
| coral | 0 (unlabelled pool) | 3 | 38.58 ± 5.48 | 0.346 ± 0.065 | 99.05 ± 0.62 | 60.74 ± 9.09 |
| dann | 0 (unlabelled pool) | 3 | 54.42 ± 2.74 | 0.525 ± 0.021 | 99.52 ± 0.51 | 59.70 ± 7.51 |
| fine-tune imagenet-init (Sec. 8) | 68 | 3 | 75.67 ± 7.67 | 0.751 ± 0.082 | - | - |
| fine-tune imagenet-init (Sec. 8) | 1360 | 3 | 96.92 ± 0.52 | 0.969 ± 0.005 | - | - |
| fine-tune lab-init (Sec. 8) | 68 | 3 | 76.67 ± 3.26 | 0.762 ± 0.030 | - | - |
| fine-tune lab-init (Sec. 8) | 1360 | 3 | 94.92 ± 1.42 | 0.949 ± 0.014 | - | - |

## 11.2 GAN filter

300 synthetic images: near-duplicates 0, classified as cedar rust with p>=0.9: 299, kept 299.
Classifier's predicted class for the synthetic images: {'cedar_apple_rust': 300}. Max similarity to a real training image: median 0.857, max 0.906.

## 11.3 GAN oversampling

| condition | synthetic imgs | seeds | PV acc % | PV cedar F1 | FGVC8 ZS acc % | FGVC8 cedar recall | PlantDoc ZS acc % | PlantDoc cedar recall |
|---|---|---|---|---|---|---|---|---|
| baseline | 0 | 3 | 99.66 ± 0.24 | 0.990 ± 0.010 | 43.94 ± 0.46 | 0.328 ± 0.115 | 59.47 ± 5.68 | 0.550 ± 0.162 |
| gan | 299 | 3 | 99.93 ± 0.12 | 1.000 ± 0.000 | 39.65 ± 2.98 | 0.158 ± 0.050 | 58.77 ± 3.44 | 0.431 ± 0.070 |

## 12 augmentation ablation

| policy | seeds | PV acc % | PlantDoc ZS % | FGVC8 ZS % |
|---|---|---|---|---|
| none | 1 | 100.00 | 56.79 | 36.50 |
| geometric | 1 | 99.59 | 64.81 | 43.54 |
| photometric | 1 | 99.80 | 51.92 | 41.89 |
| occlusion | 1 | 99.80 | 55.75 | 40.96 |
| full | 1 | 99.39 | 62.02 | 43.61 |

## 13 five seeds

| backbone | seeds | accuracy % | macro-F1 |
|---|---|---|---|
| resnet50 | 5 | 99.80 ± 0.25 | 0.9969 ± 0.0040 |
| vgg16 | 5 | 99.35 ± 0.44 | 0.9898 ± 0.0071 |
| densenet121 | 5 | 99.59 ± 0.29 | 0.9938 ± 0.0048 |
| inception_v3 | 5 | 99.59 ± 0.50 | 0.9929 ± 0.0090 |
| mobilenet_v2 | 5 | 99.71 ± 0.34 | 0.9966 ± 0.0039 |
