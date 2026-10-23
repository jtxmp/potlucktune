"""Builds the demo split from Banking77 (PolyAI, CC-BY-4.0).

Twelve intents, three members, four intents each, so no member sees the
others' intents. Writes data/demo/{alice,bob,carol}.jsonl, data/demo/all.jsonl
(the union, for the central baseline) and data/demo/test.jsonl (held-out,
all twelve intents). Data is downloaded, never committed.

  .venv/bin/python scripts/make_demo_data.py [--per-intent 40] [--test-per-intent 20]
"""

import argparse
import json
import os
import random

from huggingface_hub import hf_hub_download

# Mirror of Banking77 in plain JSONL. Original dataset: PolyAI, CC-BY-4.0.
REPO = "mteb/banking77"
REVISION = "main"

INTENTS = {
    "alice": ["card_arrival", "lost_or_stolen_card", "card_not_working", "activate_my_card"],
    "bob": ["exchange_rate", "top_up_failed", "transfer_not_received_by_recipient", "cash_withdrawal_charge"],
    "carol": ["pin_blocked", "change_pin", "edit_personal_details", "terminate_account"],
}


def load(split):
    path = hf_hub_download(REPO, f"{split}.jsonl", repo_type="dataset", revision=REVISION)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-intent", type=int, default=40)
    p.add_argument("--test-per-intent", type=int, default=20)
    p.add_argument("--out", default="data/demo")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    rng = random.Random(args.seed)
    train, test = load("train"), load("test")
    wanted = {i for intents in INTENTS.values() for i in intents}

    def by_label(rows):
        groups = {}
        for r in rows:
            label = r.get("label_text") or r.get("label")
            if label in wanted:
                groups.setdefault(label, []).append(r["text"])
        return groups

    tr, te = by_label(train), by_label(test)
    missing = wanted - set(tr)
    if missing:
        raise SystemExit(f"intents not found in {REPO}: {sorted(missing)}")

    os.makedirs(args.out, exist_ok=True)
    union = []
    for member, intents in INTENTS.items():
        rows = []
        for intent in intents:
            texts = sorted(tr[intent])
            rng.shuffle(texts)
            rows += [{"text": t, "label": intent} for t in texts[: args.per_intent]]
        rng.shuffle(rows)
        union += rows
        write(os.path.join(args.out, f"{member}.jsonl"), rows)
    rng.shuffle(union)
    write(os.path.join(args.out, "all.jsonl"), union)

    rows = []
    for intent in sorted(wanted):
        texts = sorted(te[intent])
        rng.shuffle(texts)
        rows += [{"text": t, "label": intent} for t in texts[: args.test_per_intent]]
    write(os.path.join(args.out, "test.jsonl"), rows)
    print(json.dumps({"members": {m: len(i) for m, i in INTENTS.items()}, "train": len(union), "test": len(rows), "labels": sorted(wanted)}))


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
