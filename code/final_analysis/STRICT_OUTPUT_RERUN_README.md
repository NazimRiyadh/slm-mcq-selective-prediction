# Strict output-only rerun specification

The archived family name `confidence_option` is not schema-identical across all
six primary models. For a future all-six-model output-only rerun, use
`strict_output_feature_whitelist.py` before fitting any reliability selector.

The whitelist permits only output-derived probability, logit, score, margin,
entropy, and option-count columns. It rejects any column whose name indicates a
layer, hidden state, activation, embedding, residual stream, MLP, or attention
quantity.

Required workflow:

1. Generate the same output-derived columns for every model.
2. Run the whitelist audit and save selected/rejected columns per condition.
3. Fit preprocessing and selectors on the training split only.
4. Select configuration on validation data only.
5. Evaluate once on the corresponding held-out split.
6. Report the schema manifest with every result table.

This patch is supplied for transparency and future work. The manuscript does
not substitute hypothetical rerun values for archived experiment outputs.
