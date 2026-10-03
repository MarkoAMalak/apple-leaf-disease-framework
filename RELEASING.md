# Releasing and archiving

Every GitHub release of this repository is archived on Zenodo through the
GitHub–Zenodo integration. The concept DOI
[10.5281/zenodo.21336556](https://doi.org/10.5281/zenodo.21336556) always resolves to
the latest release.

To publish a new version:

1. Commit the changes to `main`.
2. On GitHub, open **Releases → Draft a new release**, create a new tag (for example
   `v2.1.0`) and publish it.
3. Zenodo archives the release automatically and assigns it a version DOI. Cite the
   version DOI when a paper depends on an exact snapshot.

Do not commit datasets, raw images or model weights; `.gitignore` blocks the common
formats.
