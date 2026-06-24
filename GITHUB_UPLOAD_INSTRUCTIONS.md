# How to apply this update to the existing GitHub repository

This ZIP is an **update/merge package**, not a reason to delete the existing extraction scripts.

## Preserve from the current repository

Keep these existing files unless you have newer verified versions:

- `code/locked_validation_fast_v5.py`
- `code/extract_nonmcq_finitechoice_v2.py`
- `code/slm_q1_broad_nonmcq_extraction_validation.ipynb`

## Replace or add

1. Replace the root metadata files with the versions in this package:
   - `README.md`
   - `README_REPRODUCIBILITY.md`
   - `README_FINAL_PACKAGE.md`
   - `CITATION.cff`
   - `REPRODUCIBILITY_MANIFEST.md`
   - `RESULTS_AUDIT_BROAD_NONMCQ.md`
   - `ZENODO_UPLOAD_MANIFEST.md`
   - `requirements.txt`
   - `environment.yml`
2. Replace the existing `paper/` folder with the package's `paper/` folder.
3. Add `code/final_analysis/`.
4. Replace the existing `results/` folder with the package's `results/` folder, or merge carefully if it contains additional verified outputs.
5. Add `run_final_analysis.sh`, `RELEASE_NOTES_v1.0.2.md`, and `VERSION`.
6. Do not upload the 1.4 GB feature archive to GitHub; keep it on Zenodo.

## Recommended GitHub release

- Tag: `v1.0.2`
- Title: `MLWA Submission Reproducibility Release v1.0.2`
- Release text: copy `RELEASE_NOTES_v1.0.2.md`

## Repository About settings

Description:

`Code and reproducibility materials for validation-locked reliability-signal selection in finite-choice small language models.`

Website:

`https://doi.org/10.5281/zenodo.20732805`

Topics:

- `selective-prediction`
- `small-language-models`
- `uncertainty-estimation`
- `hidden-state-probing`
- `abstention`
- `reproducible-research`
- `machine-learning`

## Final check

After committing, open the repository while logged out and verify that the title, eight-author metadata, DOI links, final paper PDFs, and release tag are visible and consistent.
