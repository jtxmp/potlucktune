#!/usr/bin/env bash
# Runs the trainer with exclusive use of the GPU, so several members on one
# machine (demo, rehearsal) take turns instead of running out of memory.
exec flock "${POTLUCKTUNE_GPU_LOCK:-/tmp/potlucktune-gpu.lock}" "$(dirname "$0")/../.venv/bin/python" "$@"
