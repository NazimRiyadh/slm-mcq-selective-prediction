# Model access and licensing notes

This repository does not redistribute model weights.

Some models used in the paper require gated access on Hugging Face. Before reproducing the full experiments, log in to Hugging Face and accept the relevant terms for each gated model, then create a read token with access to public gated repositories.

Common gated models:

- `google/gemma-2-9b-it`
- `meta-llama/Llama-3.1-8B-Instruct`

If Kaggle reports `401 Unauthorized` or `403 Forbidden`, check:

1. The model terms were accepted under the same Hugging Face account used for the token.
2. The token is named `HF_TOKEN` in Kaggle secrets.
3. The token has read permission and access to public gated repositories.
4. Internet is enabled in the Kaggle notebook.

The model licenses should be reviewed before public release of outputs.
