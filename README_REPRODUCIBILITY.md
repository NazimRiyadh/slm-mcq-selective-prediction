# Reproducibility package

This package accompanies the manuscript:

**When Do Hidden-State Probes Add Deployable Value? A Cost-Aware Locked Validation Study of Finite-Choice Small Language Models**

## What is included

- `SLM_FiniteChoice_Q1_Final_BroadNonMCQ.pdf`: final revised manuscript PDF.
- `SLM_FiniteChoice_Q1_Final_BroadNonMCQ.tex`: final revised LaTeX source.
- `figures/`: manuscript figures in PDF/PNG/SVG where available.
- `tables/`: LaTeX tables used by the manuscript.
- `code/locked_validation_fast_v5.py`: locked validation script.
- `code/extract_nonmcq_finitechoice_v2.py`: broad non-MCQ extraction script for AG News, TREC, and DBPedia.
- `code/slm_q1_broad_nonmcq_extraction_validation.ipynb`: Kaggle notebook for broad non-MCQ extraction and validation, if present.
- `locked_validation_outputs/`: final CSV outputs used to build the paper tables.
- `uploaded_validation_zips/USED_slm_locked_validation_outputs_q1_broad_nonmcq.zip`: uploaded validation ZIP used for the final broad non-MCQ results.
- `RESULTS_AUDIT_BROAD_NONMCQ.md`: audit note stating which result artifact was used.

## Reproduction levels

### Level 1: Reproduce paper tables from locked outputs

Use the CSV files in `locked_validation_outputs/`. These are the final locked validation outputs used in the paper. This level does not require GPU inference or model access.

### Level 2: Rerun locked validation from feature files

This requires the large feature archive `features_combined_q1_broad_nonmcq.zip`, containing one feature CSV, one hidden-state NPZ, and one metadata JSON per model-dataset condition. That archive is not included here unless manually added before Zenodo upload, because the chat upload only provided the locked validation outputs. If you have the Kaggle checkpoint, add it under:

```text
features/features_combined_q1_broad_nonmcq.zip
```

Then extract it and run:

```bash
python code/locked_validation_fast_v5.py
```

with environment variables:

```bash
export SLM_FEATURE_INPUT=/path/to/features_combined_q1_broad_nonmcq
export SLM_LOCKED_OUTPUT=/path/to/output_dir
export SLM_BOOTSTRAP_B=300
```

### Level 3: Regenerate broad non-MCQ features

This requires GPU resources, Hugging Face access for gated models where applicable, and public datasets. Example:

```bash
export MODEL_ID=Qwen/Qwen2.5-7B-Instruct
export MODEL_SHORT=qwen25_7b
export TASKS=agnews,trec6,dbpedia14
export OUTPUT_DIR=/kaggle/working/features_nonmcq_finitechoice
export USE_4BIT=1
python code/extract_nonmcq_finitechoice_v2.py
```

Repeat for:

```text
meta-llama/Llama-3.1-8B-Instruct -> llama31_8b_it
mistralai/Mistral-7B-Instruct-v0.3 -> mistral7b_v03
```

## Important limitation

The package does not redistribute model weights. Some checkpoints may require user-side license acceptance or Hugging Face access approval. The large hidden-state feature archive should be uploaded to Zenodo if available; otherwise, features are regenerable from the extraction scripts and model identifiers.
