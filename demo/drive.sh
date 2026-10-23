#!/usr/bin/env bash
# Plays the README demo in tmux. Recorded by demo/demo.tape (vhs); can also be
# run by hand from the repo root to watch it live.
#
# Three members, each with their own HOME (identity, run data) and their own
# slice of the demo data. alice and carol are on this machine's network; bob is
# in a simulated separate network (scripts/simnet.sh: own NAT, LAN blocked,
# +80 ms, 1% loss). They train three rounds, every member merges, and bob
# scores the result. One GPU is shared through scripts/gpu-python.sh, which
# makes the members take turns on it.
#
# Needs Linux, tmux, pasta, flock, .venv with the trainer's packages, the base
# model in the Hugging Face cache, and data/demo/ (scripts/make_demo_data.py).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
W=$(mktemp -d /tmp/potlucktune-demo.XXXXXX)
for who in alice bob carol; do mkdir -p "$W/$who"; chmod 700 "$W/$who"; done
cp "$REPO/data/demo/alice.jsonl" "$W/alice/tickets.jsonl"
cp "$REPO/data/demo/bob.jsonl" "$W/bob/tickets.jsonl"
cp "$REPO/data/demo/carol.jsonl" "$W/carol/tickets.jsonl"
cp "$REPO/data/demo/test.jsonl" "$W/bob/test.jsonl"
cp "$REPO/data/demo/test.jsonl" "$W/alice/labels.jsonl"
TM="tmux -S $W/tmux.sock"
INNER="tmux -S $W/inner.sock"
HF=${HF_HOME:-$HOME/.cache/huggingface}

cleanup () {
  set +e
  [ -n "${PLAYER:-}" ] && kill "$PLAYER" 2>/dev/null
  $INNER kill-server 2>/dev/null
  $TM kill-server 2>/dev/null
  rm -rf "$W"
}
trap cleanup EXIT HUP INT TERM

cat > "$W/tmux.conf" <<'EOF'
set -g status off
set -g pane-border-status top
set -g pane-border-format ' #{pane_title} '
set -g pane-border-style fg=colour244
set -g pane-active-border-style fg=colour244
set -g default-terminal screen-256color
EOF

# `train` logs to stderr with ISO timestamps. The demo keeps the time of day
# and the events that show a round's progress; nothing else is changed.
for who in alice bob carol; do
  cat > "$W/$who.rc" <<EOF
export HOME=$W/$who NODE_OPTIONS=--no-warnings HF_HOME=$HF HF_HUB_OFFLINE=1
export POTLUCKTUNE_PYTHON=$REPO/scripts/gpu-python.sh POTLUCKTUNE_GPU_LOCK=$W/gpu.lock
cd $W/$who
PS1='\[\e[1;36m\]$who \$\[\e[0m\] '
potlucktune () {
  if [ "\$1" = train ]; then
    node $REPO/bin/potlucktune.js "\$@" 2>&1 | grep --line-buffered -E ' (connect|ready|posted|merged|agree|DISAGREE) |^Round' \
      | sed -uE 's/^[0-9-]+T([0-9:]+)\.[0-9]+Z /\1 /; s/ (loss|seconds|bytes|examples|team)=[^ ]+//g'
  else
    node $REPO/bin/potlucktune.js "\$@"
  fi
}
EOF
done

# Setup off camera: identities, team file with three members.
as () { local who=$1; shift; (cd "$W/$who" && HOME=$W/$who NODE_OPTIONS=--no-warnings node "$REPO/bin/potlucktune.js" "$@"); }
as alice team init support --member alice >/dev/null
as alice team add bob "$(as bob id 2>/dev/null)" >/dev/null
as alice team add carol "$(as carol id 2>/dev/null)" >/dev/null

$TM -f "$W/tmux.conf" new-session -d -s demo -x 200 -y 44 "bash --rcfile $W/alice.rc"
$TM select-pane -t demo:0.0 -T 'alice · home network'
$TM split-window -h -t demo:0.0 \
  "$REPO/scripts/simnet.sh --profile hotspot -- tmux -S $W/inner.sock -f $W/tmux.conf new-session -s b 'bash --rcfile $W/bob.rc' 2>/dev/null"
$TM select-pane -t demo:0.1 -T 'bob · simulated separate network: own NAT, +80 ms, 1% loss'
$TM split-window -h -t demo:0.1 "bash --rcfile $W/carol.rc"
$TM select-pane -t demo:0.2 -T 'carol · home network'
$TM select-layout -t demo even-horizontal

type () {
  local tm=$1 pane=$2 text=$3 i
  for ((i = 0; i < ${#text}; i++)); do $tm send-keys -t "$pane" -l "${text:i:1}"; sleep 0.02; done
  sleep 0.3
  $tm send-keys -t "$pane" Enter
}
wait_for () {
  local i
  for i in $(seq 1500); do
    $1 capture-pane -p -J -t "$2" | grep -qE "$3" && return 0
    sleep 0.2
  done
  echo "timed out waiting for /$3/ in $2" >&2
  return 1
}

play () {
  for i in $(seq 50); do [ -S "$W/inner.sock" ] && break; sleep 0.1; done
  sleep 1.5
  $TM select-layout -t demo even-horizontal
  A=demo:0.0 B=b:0.0 C=demo:0.2

  type "$TM" $A '# alice, bob and carol each hold support tickets for 4 of 12 intents'
  type "$TM" $A 'wc -l tickets.jsonl'
  sleep 1
  type "$TM" $A 'potlucktune init --labels-from labels.jsonl'
  wait_for "$TM" $A 'Send the updated'
  sleep 1
  cp "$W/alice/team.json" "$W/bob/team.json"
  cp "$W/alice/team.json" "$W/carol/team.json"
  type "$TM" $A 'clear'

  type "$TM" $A 'potlucktune train --data tickets.jsonl --rounds 3'
  type "$INNER" $B 'potlucktune train --data tickets.jsonl --rounds 3'
  type "$TM" $C 'potlucktune train --data tickets.jsonl --rounds 3'
  wait_for "$INNER" $B '^Round 3 merged'
  wait_for "$TM" $A '^Round 3 merged'
  wait_for "$TM" $C '^Round 3 merged'
  sleep 2

  type "$INNER" $B 'potlucktune eval --data test.jsonl --baseline'
  wait_for "$INNER" $B 'merged adapter: accuracy'
  sleep 2
  type "$TM" $C '# every member holds the same adapter; no tickets left anyone'"'"'s machine'
  sleep 6
  $TM kill-server 2>/dev/null || true
}

play &
PLAYER=$!
$TM attach -t demo || true
