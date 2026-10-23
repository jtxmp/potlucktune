#!/usr/bin/env bash
# Runs the full CLI flow on one machine with the real trainer: three members
# (alice, bob, carol), each with their own HOME (identity and run data), the
# demo split from scripts/make_demo_data.py, connected over the public DHT.
#
#   scripts/rehearse.sh [rounds]
#
# Needs: the Python environment in .venv (see trainer/requirements.txt) and
# data/demo/. One GPU can't hold three trainers, so bob and carol train on CPU.
# Set BOB_DEVICE / CAROL_DEVICE to change that.
set -euo pipefail

ROUNDS="${1:-3}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CLI="$REPO/bin/potlucktune.js"
export POTLUCKTUNE_PYTHON="${POTLUCKTUNE_PYTHON:-$REPO/.venv/bin/python}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/potlucktune-rehearse.XXXXXX")"
echo "work dir: $WORK"

as() { local who="$1"; shift; HOME="$WORK/$who" node "$CLI" --team "$WORK/team.json" "$@"; }

for who in alice bob carol; do mkdir -p "$WORK/$who"; chmod 700 "$WORK/$who"; done
as alice team init rehearsal --member alice >/dev/null
as alice team add bob "$(as bob id 2>/dev/null)" >/dev/null
as alice team add carol "$(as carol id 2>/dev/null)" >/dev/null
as alice init --labels-from "$REPO/data/demo/test.jsonl"

start=$(date +%s)
as alice train --data "$REPO/data/demo/alice.jsonl" --rounds "$ROUNDS" --device "${ALICE_DEVICE:-auto}" 2>"$WORK/alice.log" &
a=$!
as bob train --data "$REPO/data/demo/bob.jsonl" --rounds "$ROUNDS" --device "${BOB_DEVICE:-cpu}" 2>"$WORK/bob.log" &
b=$!
as carol train --data "$REPO/data/demo/carol.jsonl" --rounds "$ROUNDS" --device "${CAROL_DEVICE:-cpu}" 2>"$WORK/carol.log" &
c=$!
wait $a $b $c
echo "training took $(( $(date +%s) - start ))s"

for who in alice bob carol; do
  echo "--- $who"
  grep -E ' (posted|merged|agree|DISAGREE) ' "$WORK/$who.log" | sed 's/^[^ ]* //'
done

echo "--- eval (bob's copy of the final merge, on the shared test set)"
as bob eval --data "$REPO/data/demo/test.jsonl" --device "${ALICE_DEVICE:-auto}"
