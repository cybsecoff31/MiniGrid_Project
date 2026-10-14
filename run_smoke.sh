#!/usr/bin/env bash
# One-command smoke test: MiniGrid-Empty-5x5-v0 on CPU, about 45 seconds.
# The return should climb from roughly 0.1 to about 0.95.
#
# Anything you pass here is forwarded to ppo_minigrid.py as a Hydra override:
#   ./run_smoke.sh seed=7 ppo.ent_coef=0.02
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"
# Fail with a useful message rather than a traceback when the env is not active.
if ! "$PY" -c "import torch, gymnasium, minigrid" 2>/dev/null; then
  echo "error: '$PY' cannot import torch, gymnasium and minigrid." >&2
  echo "       Run 'conda activate cap6671' first, or set PYTHON=/path/to/python." >&2
  exit 1
fi

exec "$PY" ppo_minigrid.py \
  env=empty5x5 \
  exp_name=smoke \
  seed=1 \
  device=cpu \
  "$@"
