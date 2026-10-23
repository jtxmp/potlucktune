# Evaluation

Does merging adapters from members who each hold part of the data give a model close to one trained on all of it? `eval/run.py` compares three ways to spend the same number of training steps:

| Setting | What trains | Steps |
|---|---|---|
| local | each member alone on their own data | rounds × steps |
| fed | PotluckTune: each round every member trains `steps` from the merged adapter, then the adapters are averaged | rounds × steps per member |
| central | one model on the union of everyone's data | rounds × steps |

All three are scored on the same held-out test set. The merge in `run.py` uses the same rule as the daemon, and `test/merge.test.js` checks that the two produce the same bytes.

## Setup

- Data: 12 intents from [Banking77](https://huggingface.co/datasets/PolyAI/banking77) (PolyAI, CC-BY-4.0). Three members with four intents each and 40 examples per intent (160 per member). Test set: 20 held-out examples per intent (240). Built by `scripts/make_demo_data.py`.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`, LoRA rank 8 on `q_proj` and `v_proj`, frozen A from the team seed, B trained with AdamW at lr 0.002, 40 steps of batch 8 per round, 3 rounds. Full config: `eval/config.json`.
- Scoring: for each test message, the model scores every label as a continuation of the prompt (mean log-probability per label token), and the highest score is the prediction.
- Hardware: one RTX 3050 Ti laptop GPU (4 GB). Peak memory during training was 2.4 GB.
- Seeds 1, 2 and 3. The seed changes the frozen A and the order of training batches.

Thresholds were written down before the first run: [THRESHOLDS.md](THRESHOLDS.md).

## Results

From `eval/results/run-1.json` (2026-10-23):

| Seed | local (mean of 3 members) | fed | central |
|---|---|---|---|
| 1 | 0.370 | 0.904 | 0.938 |
| 2 | 0.381 | 0.879 | 0.954 |
| 3 | 0.392 | 0.938 | 0.933 |
| **mean** | **0.381** | **0.907** | **0.942** |

Federated accuracy by round (mean over seeds): 0.775, 0.868, 0.907. The base model with no adapter scores 0.238 (seed 1234, one run).

Against the thresholds:

1. Federated minus local: 0.526 (per seed 0.535, 0.499, 0.546). Threshold 0.20: **met**.
2. Central minus federated: 0.035 (per seed 0.033, 0.075, −0.004). Threshold 0.10: **met**.

Reproduce:

```sh
.venv/bin/python scripts/make_demo_data.py
.venv/bin/python eval/run.py --rounds 3 --seeds 1,2,3 --out eval/results/run-1.json
```

A run takes about 10 minutes on the GPU above.

## Limits

- The split is the extreme case: each member's intents are disjoint from the others'. Local-only models cannot predict intents they never saw, so their ceiling is near one third plus whatever the base model gets right. Splits where members share labels with different distributions will show a smaller difference between local and federated.
- One dataset, one model, three seeds, 240 test examples. A difference of one test example is 0.004.
- Seed 3's federated score above central is within that noise.
- The eval runs every member in one process on one machine. The network path (Autobase, Hyperblobs, hash checks) is covered by the tests and by `scripts/rehearse.sh`, not by this eval.
