# Reproducibility manifest

## Primary evidence inputs

- `code/final_analysis/input_data/locked_primary_selected.csv` — selected primary results across five repeated splits.
- `code/final_analysis/input_data/locked_all_candidate_details.csv` — candidate-level validation/test results used for selection-objective and search-budget sensitivity.
- `code/final_analysis/input_data/locked_deepseek_stress_selected.csv` — separate DeepSeek interface stress test.
- `results/locked_validation_outputs/` — full locked-output collection archived with the study.

## Final analysis scripts

- `code/final_analysis/build_additional_analyses.py`
  - 50,000-replicate two-way model/dataset bootstrap;
  - leave-one-dataset-out and leave-one-model-out summaries;
  - dataset failure prevalence and sparse-failure counts;
  - macro and test-size-weighted summaries;
  - alternative validation objectives and equal search budgets;
  - validation-to-test degradation;
  - practical-effect and model-level cost-utility summaries.
- `code/final_analysis/strict_feature_audit.py`
  - feature-schema provenance;
  - broad compact-summary versus strict output-only distinction;
  - eight-condition strict output-only sensitivity.
- `code/final_analysis/strict_output_feature_whitelist.py`
  - conservative output-only whitelist for any future uniform all-model rerun.

## Main manuscript mapping

- Primary family means and paired differences: `locked_primary_selected.csv` and `condition_mean_deltas.csv`.
- Crossed dependence and leave-one-group-out results: `crossed_dependence_summary.csv` plus `lodo_*` and `lomo_*` outputs.
- Dataset accuracy and sparse failures: `dataset_accuracy_failure_summary.csv`.
- Macro and test-size-weighted summaries: `macro_testsize_weighted_summary.csv`.
- Selection-objective and equal-budget sensitivity: `selection_criterion_and_budget_sensitivity.csv`.
- Validation-to-test degradation: `validation_to_test_auroc_gap.csv`.
- Practical-effect sensitivity: `practical_effect_sensitivity.csv`.
- DeepSeek compatibility boundary: `deepseek_base_accuracy_vs_chance.csv` and `deepseek_delta_summary.csv`.
- Feature-schema audit and strict output-only results: `feature_family_schema_audit.csv`, `feature_schema_counts.csv`, and `strict_output_only_*` outputs.
- Condition-level AUROC figure: generated from `condition_mean_deltas.csv`.

## Interpretation guardrail

The broad 24-condition result is compact-summary versus full hidden-vector augmentation. Only the eight Gemma/Llama conditions are used for the strict output-only sensitivity. Family labels in archived CSV files are retained for provenance and should be interpreted using the schema audit.
