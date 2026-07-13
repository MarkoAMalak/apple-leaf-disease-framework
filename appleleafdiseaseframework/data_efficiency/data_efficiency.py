#!/usr/bin/env python3
"""
data_efficiency.py
==================
Scientific contribution experiment: *How much field data, and does leakage-free
lab pre-training help, to close the laboratory -> field gap for apple-leaf
disease recognition?*

It trains a ResNet50 classifier on a **field** dataset using increasing
fractions of the available field training images (0, 5, 10, 25, 50, 100 %),
under two initialisations:

  * imagenet : backbone starts from generic ImageNet weights.
  * lab      : backbone starts from your leakage-free PlantVillage-trained
               checkpoint (--lab_ckpt). The 0 % point of this curve is exactly
               the zero-shot number already reported in the thesis.

Each (init, fraction) cell is repeated over several random seeds, and the
macro-F1 and accuracy on a fixed, held-out field TEST split (never used for
training) are recorded as mean +/- standard deviation. The result is a
data-efficiency curve with error bars, a CSV, and a summary table -- a genuine,
runnable contribution that quantifies the field-data budget and the value of
leakage-free pre-training.

Field data layout expected (torchvision ImageFolder):

    field_dir/
        apple_scab/   *.jpg
        black_rot/    *.jpg          (or frogeye_leaf_spot)
        cedar_apple_rust/ *.jpg
        healthy/      *.jpg

Usage (typical, on a GPU):
    python data_efficiency.py --field_dir /path/to/FGVC8_4class \
        --lab_ckpt /path/to/resnet50_plantvillage.pt \
        --epochs 15 --seeds 3 --out runs_de

If you do not have a saved lab checkpoint, omit --lab_ckpt: the script then
reports only the ImageNet-init curve (still informative), and you can add the
lab curve later.
"""

import argparse, json, random
from pathlib import Path

import numpy as np


def set_seed(s):
    random.seed(s); np.random.seed(s)
    import torch
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_model(n_classes, init, lab_ckpt, device):
    import torch, torch.nn as nn
    import torchvision
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if init == "imagenet" else None
    model = torchvision.models.resnet50(weights=weights)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, n_classes)
    if init == "lab":
        if not lab_ckpt or not Path(lab_ckpt).exists():
            raise FileNotFoundError(f"--lab_ckpt required for lab init: {lab_ckpt}")
        sd = torch.load(lab_ckpt, map_location="cpu")
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        sd = { k.replace("module.", ""): v for k, v in sd.items() }
        # Transfer the BACKBONE only; always reinitialise the classifier head.
        # This avoids any class-order mismatch between the lab and field label
        # spaces (which would silently corrupt results), and makes the lab curve
        # a clean test of the *features* learned on the leakage-free lab set.
        own = model.state_dict()
        filt = { k: v for k, v in sd.items()
                 if k in own and own[k].shape == v.shape and not k.startswith("fc.") }
        model.load_state_dict(filt, strict=False)
        print(f"[lab-init] transferred {len(filt)} backbone tensors from checkpoint "
              f"(classifier head reinitialised)")
    return model.to(device)


def loaders(field_dir, test_frac, frac, seed, batch, workers):
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf_train = transforms.Compose([
        transforms.Resize((256, 256)), transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(), norm])
    tf_eval = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(), norm])

    base = datasets.ImageFolder(field_dir)
    targets = np.array(base.targets)
    classes = base.classes
    rng = np.random.RandomState(seed)

    # fixed stratified test split (depends only on seed, NOT on frac)
    test_idx, pool_idx = [], []
    for c in range(len(classes)):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac)))
        test_idx += list(idx[:n_test]); pool_idx += list(idx[n_test:])
    # sample a stratified fraction of the remaining pool for training
    train_idx = []
    if frac > 0:
        rng2 = np.random.RandomState(seed + 999)
        pool = np.array(pool_idx)
        pt = targets[pool]
        for c in range(len(classes)):
            ci = pool[pt == c]; rng2.shuffle(ci)
            k = max(1, int(round(len(ci) * frac / 100.0)))
            train_idx += list(ci[:k])

    ds_train_full = datasets.ImageFolder(field_dir, transform=tf_train)
    ds_eval_full = datasets.ImageFolder(field_dir, transform=tf_eval)
    test_ds = Subset(ds_eval_full, test_idx)
    test_dl = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=workers)
    train_dl = None
    if train_idx:
        train_ds = Subset(ds_train_full, train_idx)
        train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=workers, drop_last=len(train_idx) > batch)
    return train_dl, test_dl, classes, len(train_idx), len(test_idx)


def evaluate(model, dl, device):
    import torch
    from sklearn.metrics import f1_score, accuracy_score
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in dl:
            out = model(x.to(device))
            ps += out.argmax(1).cpu().tolist(); ys += y.tolist()
    return accuracy_score(ys, ps), f1_score(ys, ps, average="macro")


def train_one(model, dl, device, epochs, lr):
    import torch, torch.nn as nn
    if dl is None:
        return model
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        for x, y in dl:
            opt.zero_grad()
            loss = crit(model(x.to(device)), y.to(device))
            loss.backward(); opt.step()
        sched.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field_dir", required=True, help="ImageFolder of the FIELD dataset (class subfolders)")
    ap.add_argument("--lab_ckpt", default=None, help="ResNet50 checkpoint trained on the leakage-free lab set")
    ap.add_argument("--fractions", default="0,5,10,25,50,100")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--out", default="runs_de")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fractions = [float(x) for x in args.fractions.split(",")]
    inits = ["imagenet"] + (["lab"] if args.lab_ckpt else [])

    rows = []  # (init, frac, seed, n_train, acc, f1)
    for init in inits:
        for frac in fractions:
            # frac==0 means "no field training data": with a freshly initialised
            # head this is a random classifier, so it is not evaluated here. The
            # true 0% zero-shot point comes from the thesis's existing lab-only
            # model applied to the field set, and is added when plotting/reporting.
            if frac == 0:
                continue
            for seed in range(args.seeds):
                set_seed(seed)
                tr, te, classes, n_tr, n_te = loaders(
                    args.field_dir, args.test_frac, frac, seed, args.batch, args.workers)
                model = build_model(len(classes), init, args.lab_ckpt, device)
                model = train_one(model, tr, device, args.epochs, args.lr)
                acc, f1 = evaluate(model, te, device)
                rows.append(dict(init=init, frac=frac, seed=seed, n_train=n_tr,
                                 n_test=n_te, acc=acc, macro_f1=f1))
                print(f"[{init:8s}] frac={frac:5.1f}% seed={seed} n_train={n_tr:4d} "
                      f"acc={acc:.4f} macroF1={f1:.4f}")

    (out / "raw_results.json").write_text(json.dumps(rows, indent=2))

    # aggregate mean +/- sd over seeds
    import csv
    agg = {}
    for r in rows:
        agg.setdefault((r["init"], r["frac"]), []).append(r["macro_f1"])
    with (out / "data_efficiency.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["init", "field_fraction_%", "macroF1_mean", "macroF1_sd", "n_seeds"])
        for (init, frac), vals in sorted(agg.items()):
            m = float(np.mean(vals)); s = float(np.std(vals))
            w.writerow([init, frac, f"{m:.4f}", f"{s:.4f}", len(vals)])
            print(f"SUMMARY {init:8s} {frac:5.1f}%  macroF1 = {m:.3f} +/- {s:.3f}")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        for init in inits:
            xs = sorted({fr for (i, fr) in agg if i == init})
            ms = [float(np.mean(agg[(init, fr)])) for fr in xs]
            ss = [float(np.std(agg[(init, fr)])) for fr in xs]
            plt.errorbar(xs, ms, yerr=ss, marker="o", capsize=4,
                         label=("Lab-initialised (leakage-free pre-train)"
                                if init == "lab" else "ImageNet-initialised"))
        plt.xlabel("Field training data used (%)")
        plt.ylabel("Macro-F1 on held-out field test set")
        plt.title("Data-efficiency: closing the laboratory -> field gap")
        plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout(); plt.savefig(out / "data_efficiency_curve.png", dpi=150)
        print(f"[ok] plot -> {out/'data_efficiency_curve.png'}")
    except Exception as e:
        print(f"[warn] plotting failed: {e}")

    print("\n[done] send back the whole", out, "folder (CSV + PNG + raw_results.json).")


if __name__ == "__main__":
    main()
