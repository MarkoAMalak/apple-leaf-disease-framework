# Detection and Classification of Apple Diseases using ML Techniques
### A Leakage-Free Deep-Learning Framework with Honest Cross-Dataset Evaluation

Code, notebooks, and results for the M.Sc. thesis and the companion paper. The framework
studies apple-leaf disease recognition with two methodological issues placed at the centre
of the design — **data leakage** and the **laboratory→field gap** — and adds an explicit
**detection** component and a **data-efficiency** analysis.

> **Data policy.** No dataset images or model weights are stored in this repository. The
> datasets used are public (see below) and are downloaded/prepared locally by the scripts.
> The `.gitignore` blocks all image, dataset, and checkpoint files from ever being committed.

## What is here

```
apple-leaf-disease-framework/
├── notebooks/                     # the experiments, at their latest executed state
│   ├── ADDC_final_pipeline.ipynb  #   leakage-free 4-class classification (5 CNN backbones)
│   ├── fgvc8_field_eval.ipynb     #   cross-dataset field evaluation (zero-shot vs CV)
│   ├── apple_yolo_colab.ipynb     #   YOLOv11 object detection (PlantDoc apple boxes)
│   └── data_efficiency.ipynb      #   data-efficiency & domain-adaptation study
├── scripts/                       # standalone, command-line versions of the experiments
│   ├── detection/{prepare_plantdoc_yolo.py, train_eval_yolo.py}
│   └── data_efficiency/data_efficiency.py
├── results/                       # small result summaries only (CSV + curve/figure PNGs)
│   ├── data_efficiency_fgvc8/     #   macro-F1 vs field-data fraction (ImageNet vs lab init)
│   ├── data_efficiency_plantdoc/  #   the same study replicated on PlantDoc
│   └── detection/                 #   YOLO metrics, PR/results curves, confusion matrix
├── docs/RUNNING.md                # exact commands for every experiment
├── paper/                         # the manuscript (PDF)
├── thesis/                        # the thesis (PDF + DOCX)
├── splits/                        # leakage-free split lists (add locally)
├── requirements.txt
└── LICENSE                        # MIT
```

## Datasets (public, not redistributed here)

* **PlantVillage** (apple subset) — de-duplicated, leakage-free; one image per physical leaf.
* **PlantDoc** — field images; object-detection release used for the YOLO experiment.
* **Plant Pathology 2021 / FGVC8** — four-class field set; frogeye leaf spot is mapped to
  black rot (same fungus, *Botryosphaeria obtusa*).

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then follow [`docs/RUNNING.md`](docs/RUNNING.md). A CUDA GPU is recommended.

## Headline results

* Leakage-free in-domain accuracy: all five backbones > 99.5 %; ResNet50 / InceptionV3 100 %.
* Honest field evaluation (zero-shot): 41.5 % (PlantDoc), 32.9 % (FGVC8) — the only true
  generalization figures; five-fold CV with field data added recovers to 81.9 % / 94.1 %.
* Detection: YOLOv11s at 93.5 % mAP@0.5 on the PlantDoc apple test split.
* **Data efficiency:** a leakage-free-lab backbone beats a generic ImageNet backbone at
  every field-data budget on **both** field datasets (up to +0.17 macro-F1 when field data
  is scarce) — the leakage-free protocol is not only honest but practically useful.

## Citation

Please cite the paper (BibTeX and DOI will be added here on publication / Zenodo archival).

## License

MIT — see [`LICENSE`](LICENSE).
