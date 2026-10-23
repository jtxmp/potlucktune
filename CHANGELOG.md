# Changelog

## 0.1.0 (2026-10-23)

First version.

- `try`, `id`, `team init`, `team add`, `init`, `train`, `status`, `eval`, `export`.
- FFA-LoRA: frozen A derived from the team seed, B trained locally, merged by example-weighted mean.
- Rounds recorded in an Autobase; adapters in Hyperblobs; joins signed by members' identity keys; merges compared by hash.
- Eval on a 12-intent Banking77 split: federated 0.907, local-only 0.381, central 0.942 (mean of 3 seeds).
