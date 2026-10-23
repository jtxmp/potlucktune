# Spike results

Go/no-go checks run on 2026-10-23 before building the CLI. Machine: Fedora 43, Node 22.22.2, RTX 3050 Ti laptop GPU (4 GB), Python 3.12 with torch 2.11.0+cu128, transformers 5.20.2, peft 0.21.2.

| # | Question | Result |
|---|---|---|
| 1 | Does LoRA training of a 0.5B model fit in 4 GB? | Yes. Qwen2.5-0.5B-Instruct, rank 8 on `q_proj`/`v_proj`, batch 8, sequences up to 96 tokens, bf16 base with fp32 B: peak 2,437 MB allocated. |
| 2 | Is a round short enough for a demo? | 40 steps took 3.8 s on the GPU; the whole trainer call, including loading the model, about 10 s. Scoring 240 test messages against 12 labels took 13 s. |
| 3 | Does averaging B with a shared frozen A match the mean of the updates? | Yes by construction (B·A is linear in B); `test/merge.test.js` checks it numerically. Accuracy against central training: [eval/README.md](../eval/README.md). |
| 4 | Do two members reach the same merged bytes? | Yes. The merge sorts by member key, sums in float64 and casts once; the JS merge and the Python merge in `eval/run.py` give identical bytes (`test/merge.test.js`), and every member reported the same hash in `potlucktune try` and the integration test. |

Adapter size for the config above: 791,448 bytes per member per round.

## Autobase (spike/autobase.mjs)

Owner bootstraps an Autobase; a member joins with a signed optimistic append; an outsider tries the same. Local testnet, three runs:

| Step | Time |
|---|---|
| Member writable after its join request | 164, 301, 580 ms |
| Owner sees the member's first post | 101, 104, 103 ms |
| Owner fetches the member's 2 MB Hyperblob | 51, 188, 161 ms |

An outsider's join request (valid signature, key not in the team) was ignored: it never became a writer.

Findings that shaped the code:

- **Replicate with `base.replicate(conn)`, not `store.replicate(conn)`.** With the store's stream the member's optimistic join never reached the owner (owner's view stayed at 2 entries for 30 s). With the base's stream it arrived in about 100 ms. The README's example uses `store.replicate`.
- **Never call `host.addWriter` for the bootstrap writer.** Calling it on the bootstrap core with `{ indexer: false }` throws `ERR_ASSERTION: Invalid checkout ... for batch`. The owner's writer is now bound by the team entry itself, and only other members send join requests.
- The first version used a Hyperbee view and hit the same assertion, so I switched to a plain Hypercore of JSON entries. The assertion was later traced to the `addWriter` call above; Hyperbee was not retested. The plain log is enough: state is rebuilt by reading it, and it holds a few entries per member per round.

## Discovery

Three members started at the same moment on one machine on the public DHT often never connected: three of five runs (of `potlucktune try` or an equivalent script) stalled for 120 to 180 s with no connection. PotluckRAG found the cause: Hyperswarm repeats a topic lookup only every 10 minutes. With the same fix (look up again with backoff from 5 s to 60 s while a member is missing), three of three `try` runs completed in 11 to 14 s.

One earlier `try` run finished all rounds but never exited while closing three nodes at once. Closing them one at a time exited normally in every later run. The cause was not found.
