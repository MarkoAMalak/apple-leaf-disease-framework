# Data paths for `soic_missing_experiments.ipynb`

Run `prepare_data.ps1` once first (see below). It builds everything under
this `code soic` folder without duplicating your large datasets (it uses
NTFS junctions, which are transparent to Python/PyTorch). After it finishes,
use these exact paths in the notebook's CONFIG+RUN cells.

## How to run the prep script

In this folder, either right-click `prepare_data.ps1` -> "Run with PowerShell",
or from a terminal:
```
powershell -ExecutionPolicy Bypass -File "prepare_data.ps1"
```
It's safe to re-run -- it skips anything that already exists, and tells you
about anything it couldn't find.

## Path cheat-sheet (after running the script)

Assuming this folder is `H:\Master\co work\SOIC major revision\code soic`:

| Section | Argument | Path |
|---|---|---|
| 1. t-SNE | `plantvillage_root` | `code soic\data\plantvillage_leakfree\test` |
| 1. t-SNE | `fgvc8_root` | `code soic\data\fgvc8_frogeye_only` |
| 1. t-SNE | `checkpoint` | `code soic\checkpoints\cls_resnet50.pt` |
| 2. Leaky split | `raw_root` | **not resolved yet** -- see below |
| 3. Data efficiency | `fgvc8_root` | `code soic\data\fgvc8_by_class` |
| 3. Data efficiency | `lab_checkpoint` | `code soic\checkpoints\cls_<backbone>.pt` |
| 4. YOLOv8 vs YOLOv11 | `data_yaml` | `code soic\data\yolo_plantdoc_data.yaml` |
| 5. Grad-CAM per class | `plantvillage_test_root` | `code soic\data\plantvillage_leakfree\test` |
| 5. Grad-CAM per class | `checkpoint` | `code soic\checkpoints\cls_resnet50.pt` |
| 6. GAN oversampling | `minority_root` | `code soic\data\plantvillage_leakfree\train\cedar_apple_rust` |
| 6. GAN oversampling | `real_train_root` | `code soic\data\plantvillage_leakfree\train` |
| 6. GAN oversampling | `real_test_root` | `code soic\data\plantvillage_leakfree\test` |
| 6. GAN oversampling | `plantvillage_root` | `code soic\data\plantvillage_leakfree` |
| 6. GAN oversampling | `checkpoint` | `code soic\checkpoints\cls_resnet50.pt` |
| 7. Domain adaptation | `plantvillage_root` | `code soic\data\plantvillage_leakfree` |
| 7. Domain adaptation | `field_root` | `code soic\data\fgvc8_by_class` |
| 8. Manual annotations | `images_dir` | `code soic\manual_annotations\images` (put ~20-30 picked images here yourself) |
| 8. Manual annotations | `manual_masks_dir` | `code soic\manual_annotations\masks` (put your drawn masks here yourself) |
| 8. Manual annotations | `checkpoint` | `code soic\checkpoints\cls_resnet50.pt` |

## Notes / things that needed a real decision, not just a copy

- **Checkpoints**: found already-trained weights for all 5 backbones at
  `final paper 2\outputs\models\cls_*.pt` -- these are your actual leakage-free
  classifiers from the paper's Table 3 run, so results from Sections 1, 3, 5,
  6, 7, 8 will be directly comparable to the paper's numbers.
- **PlantVillage class folders were renamed**: the real data lives under
  `final\clean_apple_dataset\images\{train,val,test}\Apple___<Class>\`, but
  the notebook's code expects bare names (`apple_scab`, `black_rot`, ...). The
  script links `apple_scab` -> `Apple___Apple_scab`, etc., so this is now
  transparent.
- **FGVC8 has no `black_rot` label.** Per the manuscript's own Section 2.6
  discussion, `frog_eye_leaf_spot` is used as the closest visual proxy for
  field evaluation of that class in Sections 3 and 7 (folder `black_rot`
  actually contains frogeye images -- this is intentional, not a bug).
  Section 1's t-SNE keeps the real name (`frogeye_leaf_spot`) since it's
  testing exactly that assumption.
- **YOLO `data.yaml` was broken**: it pointed at `D:\ADDC new\...`, a path
  from a different machine/drive, and had no `test` split (only `train`/`val`
  -- the `val` images are actually named `TEST_*.jpg`, i.e. they already are
  your held-out test set). The script writes a corrected copy with `test:`
  pointing at the same folder as `val:`, matching the paper's protocol.
- **Section 2 (leaky split) is NOT resolved.** It needs the *original,
  un-deduplicated* PlantVillage folder with the on-disk augmented copies
  still present (`_180deg`, `_flipLR`, etc.). The only raw candidate found is
  `final\Plantvillage\archive.zip` (185 MB, the raw Kaggle download) -- the
  script lists its top-level folder names into
  `data\plantvillage_archive_top_level_listing.txt` without extracting
  everything. Open that file: if you see per-class apple folders with
  augmented-looking filenames, tell Claude and it'll finish wiring up
  Section 2. If not, that raw copy may not exist anymore and Section 2 would
  need to be regenerated from scratch (re-running the original augmentation
  step before the leak-free cleanup).
- **Section 8 needs images and masks you draw yourself** -- the script only
  creates the empty `manual_annotations\images` / `manual_annotations\masks`
  folders. `data\manual_gold_subset_reference` (linked from `manual gold
  subset\`) has your earlier color-based reference work, useful context but
  not a substitute for the new manual annotations Reviewer B asked for.
