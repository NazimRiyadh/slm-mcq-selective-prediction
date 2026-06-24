# Reproducibility guide

This guide accompanies the manuscript **Cost-Aware Reliability Signal Selection for Finite-Choice Small Language Models: A Locked Validation Study**.

## Reproduction levels

### Level 1 — Reproduce final tables and sensitivity summaries from locked outputs

This level requires no GPU and no model access. The required CSV files are included in:

- `code/final_analysis/input_data/`
- `results/locked_validation_outputs/`

Run:

```bash
bash run_final_analysis.sh
```

The script executes:

1. `build_additional_analyses.py`
   - seed-averaged condition effects;
   - 50,000-replicate two-way model/dataset bootstrap;
   - leave-one-dataset-out and leave-one-model-out analyses;
   - failure-prevalence summaries;
   - equal-condition and test-size-weighted summaries;
   - alternative validation-objective and equal-search-budget sensitivity;
   - validation-to-test degradation;
   - practical-effect and model-level cost-utility summaries.
2. `strict_feature_audit.py`
   - broad compact-summary versus strict output-only schema audit;
   - eight-condition Gemma/Llama strict output-only sensitivity;
   - 50,000-replicate crossed sensitivity for that subset.

Expected derived outputs are written to `code/final_analysis/analysis_outputs/` and mirrored in `results/final_analysis/`.

### Level 2 — Rerun locked validation from archived feature files

Download the large feature archive from Zenodo:

- Stable DOI: `10.5281/zenodo.20732805`
- Current published version: `10.5281/zenodo.20732806`

Extract `features_combined_q1_broad_nonmcq.zip`, then use the repository's existing locked-validation script:

```bash
export SLM_FEATURE_INPUT=/absolute/path/to/extracted/features
export SLM_LOCKED_OUTPUT=/absolute/path/to/new/output
export SLM_BOOTSTRAP_B=50000
python code/locked_validation_fast_v5.py
```

The archived manuscript results were generated from five repeated stratified splits with seeds 0–4. Do not change the feature schema or candidate grid when attempting exact reproduction.

### Level 3 — Regenerate auxiliary finite-choice features

GPU resources and Hugging Face access are required. Example:

```bash
export MODEL_ID=Qwen/Qwen2.5-7B-Instruct
export MODEL_SHORT=qwen25_7b
export TASKS=agnews,trec6,dbpedia14
export OUTPUT_DIR=/absolute/path/to/features_nonmcq_finitechoice
export USE_4BIT=1
python code/extract_nonmcq_finitechoice_v2.py
```

Repeat with the model identifiers documented in the manuscript and supplement. Some checkpoints require user-side license acceptance or access approval.

## Exact final-analysis commands

The supplied runner executes the following commands:

```bash
python code/final_analysis/build_additional_analyses.py \
  --data-dir code/final_analysis/input_data \
  --out-dir code/final_analysis/analysis_outputs \
  --tables-dir paper/tables \
  --figures-dir paper/figures

python code/final_analysis/strict_feature_audit.py \
  --input code/final_analysis/input_data/locked_primary_selected.csv \
  --out-dir code/final_analysis/analysis_outputs \
  --bootstrap-reps 50000 \
  --seed 20260623
```

## Interpretation guardrail

The broad 24-condition analysis is **compact reliability summaries versus compact summaries plus PCA-compressed full hidden vectors**. Only the eight Gemma-2/Llama-3.1 conditions are used for the strict output-only sensitivity. The released machine-readable family labels are preserved to maintain exact correspondence with the archived outputs.

## Verification

After running the scripts, compare generated files against `results/final_analysis/`. Small textual differences in generated LaTeX can occur across environments; numerical CSV outputs should agree up to normal floating-point precision.
