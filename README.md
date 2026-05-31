# Cost-Aware Selective Prediction for Multiple-Choice Small Language Models

This repository contains the reproducibility materials for the manuscript:

**Cost-Aware Selective Prediction for Multiple-Choice Small Language Models: When Do Hidden-State Probes Add Value Beyond Confidence and Option Scores?**

## Scope

This is a controlled study of **answerable multiple-choice question answering (MCQ)** under forced-choice scoring. It evaluates selective prediction and failure detection for small/open-weight language models. It is **not** a general benchmark for open-ended hallucination detection, retrieval-augmented generation, tool use, long-form summarization, or false-premise abstention.

## What is included

- `notebooks/`: the inline Kaggle notebook used for Gemma-2-9B-IT and Llama-3.1-8B-Instruct plus the supervised contrastive hidden-state probe.
- `configs/`: experiment configuration summary, model list, datasets, seeds, and metrics.
- `results/final_tables/`: processed CSV tables used to build the final manuscript results.
- `results/latex_tables/`: LaTeX table fragments included in the manuscript.
- `figures/`: final paper figures in PDF/PNG format.
- `manuscript/`: final LaTeX source and compiled PDF draft.
- `supplementary/`: self-critique and revision audit.

Model weights are **not redistributed**. The experiments download model checkpoints from their official Hugging Face repositories, subject to their licenses and gated-access requirements.

## Reproducing the core Gemma/Llama extension

The main inline notebook is:

```text
notebooks/slm_gemma_llama_inline_full_experiments.ipynb
```

Recommended environment: Kaggle GPU with internet enabled. For 7B/9B-class models, use 4-bit quantized loading and batch size 1.

1. Create/install the environment:

```bash
pip install -r requirements.txt
```

2. Add a Hugging Face token. In Kaggle, create a secret named:

```text
HF_TOKEN
```

The same Hugging Face account must have accepted the access conditions for gated models such as:

- `google/gemma-2-9b-it`
- `meta-llama/Llama-3.1-8B-Instruct`

3. Run a quick check by setting `DEBUG_N = 20` in the notebook.

4. For the paper-scale reproduction, set:

```python
DEBUG_N = None
OVERWRITE = False
RUN_FEATURE_EXTRACTION = True
RUN_ANALYSIS = True
RUN_LATENCY = True
```

5. The notebook writes outputs under a working output folder and produces zipped results containing tables, figures, features, and diagnostics.

## Main results contained in this package

The final processed tables include model base accuracy, feature-family AUROC summaries, supervised contrastive hidden-probe results, fixed-coverage selective prediction metrics, AURC/E-AURC, failure capture, PCA dimension sensitivity, layer localization, redundancy/group-permutation diagnostics, and latency/VRAM diagnostics.

## Large raw outputs

GitHub should not store very large hidden-state or feature archives. For full archival, upload large raw output ZIPs to Zenodo or GitHub Releases. The manuscript can cite the GitHub repository for code and the Zenodo DOI for the full reproducibility archive.

## Citation

After creating a Zenodo archive, update `CITATION.cff` and the manuscript `Data and code availability` section with the final DOI.

## License

Code is released under the MIT License. Processed result tables and figures may be reused with attribution, subject to the licenses of the underlying datasets and model checkpoints.
