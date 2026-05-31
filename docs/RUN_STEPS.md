# Reproduction run steps

## Quick debug run

Use this to verify that gated model access, GPU memory, datasets, and analysis code work.

1. Open `notebooks/slm_gemma_llama_inline_full_experiments.ipynb` in Kaggle.
2. Enable GPU and internet.
3. Add `HF_TOKEN` as a Kaggle secret and enable it for the notebook.
4. Set:

```python
DEBUG_N = 20
```

5. Run all cells.

The debug run is not used for paper numbers because some model-dataset pairs may have too few failures/correct examples for AUROC.

## Full paper-scale run

Restart the Kaggle session after the debug run. Then set:

```python
DEBUG_N = None
OVERWRITE = False
RUN_FEATURE_EXTRACTION = True
RUN_ANALYSIS = True
RUN_LATENCY = True
```

Run the notebook. The resulting ZIP can be archived in Zenodo as a large raw output.

## Notes

- Gemma and Llama are gated models on Hugging Face. Access must be accepted on the model pages before running.
- The notebook includes a NumPy compatibility patch for `np.trapz` / `np.trapezoid`.
- Hidden-state NPZ files can be large. Store them in Zenodo or GitHub Releases, not the main GitHub repository.
