# Contributing to PotluckTune

Contributions are welcome, from a docs fix to a robust merge rule. For security problems, follow [SECURITY.md](SECURITY.md) and don't open a public issue.

## Setup

You need Node 22 or later and pnpm. The Node tests use a stand-in trainer, so they need no Python, GPU or internet.

```sh
git clone https://github.com/jtxmp/potlucktune && cd potlucktune
pnpm install
pnpm test                             # local DHT testnet and the stand-in trainer
node bin/potlucktune.js try           # three members over the real DHT
git config core.hooksPath .githooks   # optional: scan each commit for secrets (needs gitleaks)
```

For the real trainer, the eval and `scripts/rehearse.sh`, create the Python environment in the clone:

```sh
python3 -m venv .venv
.venv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # or the CPU build
.venv/bin/pip install -r trainer/requirements.txt
```

With `.venv` present, `pnpm test` also checks that the JavaScript merge and `eval/run.py`'s merge give the same bytes.

## Code map

| File | What it does |
|---|---|
| `bin/potlucktune.js` | CLI: `try`, `id`, `team`, `init`, `train`, `status`, `eval`, `export` |
| `src/state.js` | The run's rules: which operations are accepted (`decide`), how the view becomes state (`reduce`), when a round is complete (`roundPosts`), merge agreement |
| `src/node.js` | Corestore, Autobase, Hyperblobs and Hyperswarm for one member: firewall, join, adapter put and fetch with hash checks, lookup retry |
| `src/run.js` | `createRun` (owner) and `runRounds`: train, post, wait for the round, fetch, merge, report |
| `src/merge.js` | Deterministic example-weighted mean of B |
| `src/safetensors.js` | Strict F32 safetensors reader and deterministic writer |
| `src/config.js` | Default training config and its validation |
| `src/trainer.js` | Runs `trainer/trainer.py` as a subprocess; the stand-in trainer for tests and `try` |
| `trainer/trainer.py` | Frozen A from the seed, B training, label scoring, PEFT export |
| `src/team.js`, `src/identity.js`, `src/log.js` | Shared with PotluckLLM and PotluckRAG: team file, Ed25519 identity, event logs |
| `eval/` | Local vs federated vs central comparison, pre-registered thresholds, results |
| `spike/` | Measurements behind design choices |

## House rules

- **Plain JavaScript with types in comments** on the Node side: ES modules, `// @ts-check`, JSDoc, no build step. The trainer is plain Python with no framework beyond torch, transformers and peft.
- **No new runtime dependency without an issue first.**
- **Every fix and feature comes with a test** that fails without it. Node tests must not need Python, a GPU or the internet; use `fakeTrainer` and `hyperdht/testnet`.
- **Keep `decide` pure and order-independent.** Members must reach the same state from the same operations in any order. A change to `src/state.js` needs a test that applies operations in more than one order.
- **Never log training data.** Logs hold event names, key and hash prefixes, round numbers, losses and timings.
- **Training or merge changes come with an eval run.** Include the before and after tables from `eval/run.py`.

## Pull requests

- Keep each pull request to one change, and link the issue it closes.
- Say how you verified it: tests added, commands run, hardware used.
- CI runs the tests on Linux and macOS, `pnpm audit` and a secrets scan. All must pass.
- Contributions are licensed under the [MIT License](LICENSE).
