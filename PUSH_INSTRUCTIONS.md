# Publishing the repository (GitHub + Zenodo)

The repository is fully prepared. Pushing must be done from your own machine with your
own GitHub authentication.

## 1. Create the repository on GitHub

Go to https://github.com/new and create a **Public** repository named
`apple-leaf-disease-framework`. Do not add a README or .gitignore (they are already here).

## 2. Push from your machine

Open a terminal inside the unzipped `apple-leaf-disease-framework/` folder and run:

```bash
git init
git add .
git status          # verify no raw images/datasets/weights (.jpg .png raw or .pt) were added
git commit -m "Leakage-free apple-leaf disease framework: code, notebooks, results"
git branch -M main
git remote add origin https://github.com/MarkoAMalak/apple-leaf-disease-framework.git
git push -u origin main
```

The `.gitignore` automatically blocks datasets, raw images, and model weights (`.pt`).
If `git status` shows any data file, do not commit it.

## 3. Zenodo DOI

1. Sign in to https://zenodo.org/ with your GitHub account.
2. Go to https://zenodo.org/account/settings/github/ and switch the toggle ON next to
   `apple-leaf-disease-framework`.
3. Back on GitHub, create a Release (Releases → Draft a new release → tag `v1.0.0` → Publish).
4. Zenodo captures the release and mints a DOI. Copy it.

## 4. Add the DOI back to the manuscript

Send me the DOI and I will place it in the paper's Data Availability statement, the thesis
Reproducibility section, and as a badge in this README.
