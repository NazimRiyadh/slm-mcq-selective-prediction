# Self-critique and revision audit - Gemma/Llama final update

## What changed

- Added successful Gemma-2-9B-IT and Llama-3.1-8B-Instruct runs after gated Hugging Face token access was fixed.
- Updated the scaling audit from five model families to seven 7B/9B-class families.
- Added a supervised contrastive hidden-state probe on Gemma and Llama to answer the reviewer concern that PCA/logistic hidden probes may be too weak.
- Regenerated core tables: base accuracy, feature-family AUROC, hidden gains, selective fixed-coverage metrics, layer localization, PCA sensitivity, redundancy, and latency.
- Updated the abstract, methods, results, discussion, conclusion, and scope table.

## Main updated numbers

- Final audit: 28 model-dataset cases across seven 7B/9B-class model families.
- Confidence/option AUROC: 0.772 over all cases; 0.800 over the competent subset.
- Confidence/option + PCA hidden AUROC: 0.777 over all cases; 0.807 over the competent subset.
- Mean incremental gain of confidence/option + PCA hidden: +0.004 over all cases; +0.008 over the competent subset.
- Gemma/Llama contrastive probe: confidence/option = 0.826 AUROC; contrastive hidden-only = 0.746; confidence/option + contrastive = 0.772.

## Remaining weaknesses

- The paper is still an empirical audit, not a new algorithm or theory. The novelty must be framed as cost-aware signal-selection and redundancy diagnosis.
- The task setting remains answerable MCQ. The paper should not claim general hallucination detection or RAG reliability.
- DeepSeek-R1-Distill-Qwen-7B remains a forced-choice stress test, not a fair general reasoning estimate.
- Code repository URL, environment file, author email, and Zenodo/OSF DOI remain required before submission.

## Recommendation

This version is substantially stronger than the previous V6 and the earlier five-model revision. It is now defensible as a Q1-style empirical reliability audit, especially for an Elsevier venue such as Applied Soft Computing or Machine Learning with Applications. It still cannot be guaranteed to pass Q1 because methodological novelty remains limited, but the main reviewer objections on scale, selective metrics, and hidden-probe strength have been addressed.
