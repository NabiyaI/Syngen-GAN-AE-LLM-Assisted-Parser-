# Architecture

## End-to-end flow

1. User prompt arrives from CLI or GUI.
2. Parser layer (`rules`, `llm`, or `hybrid`) converts prompt to `PromptSpec`.
3. Domain routing resolves a target profile and GAN+AE checkpoint.
4. Generator backend produces synthetic rows.
5. Rows are written to timestamped CSV.
6. Optional validation compares generated data vs source datasets.

## Components

- `prompt_parser/`
  - Controlled-NL parser and key/value override handling.
  - Conflict detection and strict-mode validation.
  - Optional LLM rewrite (Gemini/OpenAI) before rules parse.
- `generator.py`
  - Backend abstraction (`baseline` and `gan_ae`).
- `gan_ae_backend.py`
  - Checkpoint loading and inference dispatch.
- `gan_ae_full/`
  - Train/infer for full tabular AutoEncoder + latent GAN.
- `checkpoint_registry.py`
  - Domain/profile to checkpoint resolution.
- `main.py`
  - CLI entrypoint, JSON response output, CSV write.
- `gui_app.py`
  - Desktop UI with parse preview and generation flow.
- `kaggle_eval/`
  - Benchmark pipeline and similarity reports.

## Data and control diagram

```text
Prompt --> Parse Router --> PromptSpec --> Checkpoint Resolver --> Backend
   ^            |                                 |                 |
   |            +--(optional LLM rewrite)         |                 v
GUI/CLI -------------------------------------------+--> GAN+AE Inference --> CSV
```

## Checkpoint contract

Required files for full GAN+AE checkpoints:

- `metadata.json`
- `preprocessor.pkl`
- `encoder.pt`
- `decoder.pt`
- `generator.pt`

