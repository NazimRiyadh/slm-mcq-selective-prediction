# Zenodo upload manifest

Upload this full package to Zenodo after creating a GitHub release.

## Include in Zenodo now

- Full revised manuscript PDF and LaTeX source.
- Code scripts and notebook.
- Final locked validation outputs.
- Manuscript figures and LaTeX tables.
- Result audit notes.
- `README_REPRODUCIBILITY.md`, `CITATION.cff`, `LICENSE`, and environment files.

## Strongly recommended manual addition

If available from Kaggle/local storage, add the large feature archive before Zenodo upload:

```text
features/features_combined_q1_broad_nonmcq.zip
```

Expected content of that archive:

```text
38 feature CSV files
38 hidden NPZ files
38 metadata JSON files
```

This allows reviewers to rerun locked validation without rerunning model inference. The current package already allows reviewers to verify the reported tables from locked validation CSV outputs and to regenerate features from public models/datasets, but it does not include the large hidden-state NPZ archive unless you add it manually.
