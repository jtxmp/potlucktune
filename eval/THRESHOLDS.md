# Pre-registered thresholds

Written on 2026-10-23, before the first full run of `eval/run.py`.

Setup: `eval/config.json`, the demo split from `scripts/make_demo_data.py` (12 Banking77 intents, three members with four intents each, 160 training examples per member, 240 held-out test examples), 3 rounds, seeds 1, 2 and 3.

Federated training counts as working if, averaged over the seeds:

1. Federated accuracy beats the mean local-only accuracy by at least 0.20.
2. Federated accuracy is within 0.10 of central accuracy.

A result that misses either threshold is reported as a miss, not tuned until it passes.
