# Final results audit

The final manuscript uses the locked-output files under `results/locked_validation_outputs/` and the derived analyses under `results/final_analysis/`.

The broad 24-condition lower-dimensional family is not uniformly output-only. The feature-schema audit identifies strict output-only features for Gemma-2 and Llama-3.1 and compact output-plus-layer-summary features for Falcon3, Mistral, OLMo, and Qwen. The manuscript therefore interprets the broad comparison as compact summaries versus compact summaries plus PCA-compressed full hidden vectors and reports the eight-condition Gemma/Llama block as the strict output-only sensitivity.

No model inference is repeated by the final-analysis scripts. The scripts operate on the archived locked outputs and preserve the original machine-readable family labels.
