# Security policy

## Reporting a vulnerability

Report vulnerabilities privately through GitHub: [open a private advisory](https://github.com/jtxmp/potlucktune/security/advisories/new). Please don't open a public issue for a security problem.

Include what you found, how to reproduce it, and what an attacker gains. This is a personal project, so reports are handled on a best-effort basis. Fixes land on `main`, which is the only supported version.

## What PotluckTune defends against

The full threat model is in [docs/threat-model.md](docs/threat-model.md). In short: PotluckTune keeps outsiders out of a team's training run, makes members agree on one merged adapter, and keeps each member's data on their machine. It does not protect members from each other: every member receives every other member's adapter, and a member can send an adapter that harms the merged model.

In scope:

- Connecting to a member, or writing to a run, without a key listed in `team.json`
- Posting as another member, or changing the member list without the owner's key
- Getting a member's merge to include an adapter other than the one whose hash was posted
- Making members end a round with different merged adapters without that being reported
- A training config that makes a member's trainer read local files, run code or load pickled data
- Crashing a member with crafted log entries or adapter files
- Training data or private keys appearing in logs

Out of scope:

- A member who keeps or analyses other members' adapters
- A member who sends a poisoned or degraded adapter (listed as an open issue, not a vulnerability)
- Attacks that need write access to the user's home directory
- Vulnerabilities in PyTorch, transformers, peft, or the base model
