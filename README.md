# Cost-Aware Reliability Signal Selection for Finite-Choice Small Language Models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20732805.svg)](https://doi.org/10.5281/zenodo.20732805)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the code, processed results, manuscript sources, figures, tables, and reproducibility documentation for:

**Cost-Aware Reliability Signal Selection for Finite-Choice Small Language Models: A Locked Validation Study**

## Study overview

The study evaluates whether PCA-compressed full hidden-vector augmentation provides stable incremental selective-prediction value beyond lower-cost reliability summaries in finite-choice small language models. The primary evaluation spans six instruction-tuned models and four multiple-choice benchmarks under validation-locked model selection. The broad 24-condition analysis compares compact reliability summaries with compact summaries plus full hidden vectors; a separate eight-condition Gemma/Llama sensitivity provides the strict output-only comparison.

The paper reports selective-prediction discrimination and operating metrics, crossed model/dataset sensitivity, leave-one-cluster-out analyses, feature-schema provenance, validation-search sensitivity, and operational diagnostics. The results support a staged policy: begin with the lowest-cost admissible signal family and add full hidden-vector probes only when target-domain validation demonstrates a stable, practically meaningful benefit.

## Repository structure

- `paper/` — final manuscript, supplementary material, LaTeX sources, tables, and figures
- `code/` — original extraction/validation code retained in the repository, plus `final_analysis/`
- `code/final_analysis/` — crossed-bootstrap, sensitivity, schema-audit, and cost-utility analyses
- `results/locked_validation_outputs/` — final locked CSV outputs
- `results/final_analysis/` — derived final-analysis summaries
- `README_REPRODUCIBILITY.md` — step-by-step reproduction instructions
- `REPRODUCIBILITY_MANIFEST.md` — evidence and table/figure provenance
- `CITATION.cff` — citation metadata

## Archived feature data

The per-example feature archive is hosted on Zenodo rather than GitHub because of its size.

- Stable all-versions DOI: **10.5281/zenodo.20732805**
- Published v1.0.1 record: **10.5281/zenodo.20732806**

The Zenodo record includes `features_combined_q1_broad_nonmcq.zip`, containing option-score CSV files, hidden-state NPZ arrays, and metadata JSON files. These files allow the validation analyses to be rerun without repeating model inference. Model weights are not redistributed; gated checkpoints remain subject to their original providers' access conditions.

## Quick reproduction

Create the environment:

```bash
conda env create -f environment.yml
conda activate finite-choice-slm-reliability
```

Reproduce the final dependence, selection-sensitivity, schema-audit, and cost-utility analyses from the included locked outputs:

```bash
bash run_final_analysis.sh
```

Detailed requirements and expected outputs are documented in `README_REPRODUCIBILITY.md`.

## Paper compilation

```bash
cd paper
latexmk -pdf -interaction=nonstopmode main.tex
latexmk -pdf -interaction=nonstopmode supplementary_material.tex
```

The LaTeX source is also compatible with Overleaf using pdfLaTeX.

## Citation

Citation metadata are available in `CITATION.cff`. Please cite the Zenodo archive and, after publication, the journal article.

## License

Code and repository documentation are released under the MIT License. The Zenodo data record is governed by the license displayed on that record.
