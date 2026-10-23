# Threat model

PotluckTune lets team members train one LoRA adapter on data that stays on each member's machine. This file lists what it protects, from whom, and what it does not protect.

## Assets

- Each member's training data (the JSONL file passed to `train`).
- Each member's adapters, which are derived from that data.
- The merged adapter, which every member uses.
- Each member's machine: the Python trainer runs with the member's privileges.

## Trust assumptions

- Team members trust each other with adapters. Every member receives every other member's adapter each round.
- The team owner is trusted to set the training config and the member list. Members validate the config's form (see below), not whether its choices are good.
- The team file reaches members over a private channel. It holds no secrets: an outsider who reads it learns the topic, the member keys and the run's key, but cannot connect or write without a member's private key.
- Hugging Face and the base model's publisher are trusted for the base model's weights and tokenizer.

## Outsiders on the internet

| Threat | Control | Test |
|---|---|---|
| A key not in the team file connects | Hyperswarm firewall closes the connection before any data flows | run.test.js: outsider rejected, no connection |
| An outsider who reached the shared log writes to it | Writers are added only through a join request signed by a key in the run's member list, bound to the writer key and the run key. Operations from unbound writers are ignored. | state.test.js: forged, replayed and outside-key joins; run.test.js: outsider never becomes a writer |
| Traffic is read or altered in transit | Noise encryption bound to each Ed25519 key (Hyperswarm); every adapter is checked against the SHA-256 in its post | merge and integration tests; transport encryption provided by `hyperswarm` |
| PotluckLLM or PotluckRAG peers of the same team connect | PotluckTune joins a topic derived from the team topic, so the tools don't meet | run.test.js: topic |

## Members who misbehave or run buggy software

| Threat | Control | Test |
|---|---|---|
| A member impersonates another in the log | A writer is bound to one member by that member's signature; a member can bind only one writer | state.test.js |
| A member changes the member list | Only the owner's writer can append a member list, and it must keep the owner | state.test.js |
| A member posts an adapter trained with a different frozen A | Posts must carry the run's A hash; each member also checks that its own trainer reproduces the hash before training | state.test.js; `runRounds` |
| A member posts a malformed or oversized adapter | Posts larger than 64 MB are ignored. The safetensors reader checks header size, dtype, shapes, offsets, overlaps and trailing bytes. The merge refuses different tensor names or shapes and NaN or Inf values. | merge.test.js |
| Members posting in different orders end with different merges | A round's posts are chosen independently of arrival order; the merge is deterministic; members publish their merge hashes and `train` reports any difference | state.test.js: order; run.test.js: three members agree |
| The owner sets a config that reads local files or runs code | The model must be a Hugging Face id (`org/name`), not a path; numbers and module names are range- and pattern-checked; transformers loads without `trust_remote_code`; adapters are read with safetensors, never pickle | config.test.js |
| **A member sends an adapter trained to degrade or backdoor the model** | **Not prevented.** The merge is an example-weighted mean, so one member's update moves the result in proportion to their claimed example count, which is not verified. Robust merging (median, trimmed mean, norm clipping) is an open issue. | |
| **A member infers facts about another member's data from their adapter** | **Not prevented.** Adapter updates can carry membership signals ([LoRA as Oracle, 2026](https://arxiv.org/abs/2601.11207)). Secure aggregation, where members see only the sum, is an open issue. | |
| A member never posts and stalls the round | Not prevented; the owner removes them from `team.json`. Rounds can't close without every member, by design, so the merge never silently drops someone. | |

## Local data

- `~/.potluck/identity.json` holds the private seed (mode 0600, shared with PotluckLLM and PotluckRAG). Startup refuses a group- or world-writable identity folder.
- `~/.potlucktune/` holds the Autobase and blob stores, every round's adapters and merges, the run's config and `status.json`. It is created with mode 0700, and commands refuse to run if other users can read it.
- Logs on stderr hold event names, round numbers, 8-character key or hash prefixes, losses and timings. They never hold training text or labels. The trainer's error messages name a line number for bad input, not its content.

## Not covered

- Hosts on the path seeing that two IP addresses exchange traffic. Hyperswarm does not hide IP addresses from peers or the DHT.
- A compromised member machine.
- The quality of the merged model. `eval` measures it; nothing enforces it.
