#!/usr/bin/env bash
# Midterm "first comparison" on 8x8: control vs intervention(s), 3 seeds each.
#
# Run this inside the cap6671 conda env from the repo root:
#     conda activate cap6671
#     bash run_comparison.sh
#
# Budget is the handout's 1,000,000 steps per run (same as the baselines). On the
# machine that produced the baselines each 8x8 run took ~7 min of wall time
# (runs/baseline_doorkey8x8_seed1/env_info.json: wall_time_s ~427), so the full
# matrix below is a few hours. To do a faster preliminary pass, lower the budget
# identically for every group, e.g. append  env.total_timesteps=400000  to every
# training line, and say so in the report (a similar, reduced budget is fine as
# long as it is the SAME for control and intervention).
set -euo pipefail

SEEDS="1 2 3"
PY=python          # the cap6671 interpreter

# Commit the code first: every run records git.json, and a dirty tree means the
# recorded commit does not describe the code that ran.
if [ -n "$(git status --porcelain -- '*.py' conf)" ]; then
  echo "Commit your code changes before running the comparison (git.json would be dirty)." >&2
  exit 1
fi

# Sanity test of the two intervention wrappers on the real MiniGrid (seconds).
$PY tests/test_interventions.py

# --- Group 1: control (partial view + sparse reward) ------------------------
# All three control seeds are trained at the SAME commit as the interventions.
# Seed 1 is also a cross-check: it must reproduce runs/baseline_doorkey8x8_seed1
# exactly (same seed, same machine, 4 threads), which proves the encoder's new
# AdaptiveAvgPool2d is the identity on the 7x7 view.
for s in $SEEDS; do
  $PY ppo_minigrid.py env=doorkey8x8 seed=$s device=cpu \
     exp_name=cmp_control run_name=cmp_control_8x8_seed$s
done

# --- Group 2: shaped variant (partial view + milestone reward) --------------
for s in $SEEDS; do
  $PY ppo_minigrid.py env=doorkey8x8_shaped seed=$s device=cpu \
     exp_name=cmp_shaped run_name=cmp_shaped_8x8_seed$s
done

# --- Group 3: global ablation (full view + sparse reward) -------------------
# Optional for the midterm (one intervention is required), but it is the other
# half of the diagnostic. Comment the block out to skip it.
for s in $SEEDS; do
  $PY ppo_minigrid.py env=doorkey8x8_fullobs seed=$s device=cpu \
     exp_name=cmp_fullobs run_name=cmp_fullobs_8x8_seed$s
done

# --- Evaluate every run on the protocol (200 eps, seed 12345, sparse) --------
# For the shaped group this builds the BARE env (no MilestoneReward), so all
# groups are scored on the identical sparse objective and identical layouts.
eval_one () {  # $1 = run dir
  $PY eval_checkpoint.py model_path="$1/model.pt" policy.also_random=true
}
for s in $SEEDS;   do eval_one runs/cmp_control_8x8_seed$s; done
for s in $SEEDS;   do eval_one runs/cmp_shaped_8x8_seed$s;  done
for s in $SEEDS;   do eval_one runs/cmp_fullobs_8x8_seed$s; done

# --- Aggregate the spread across seeds --------------------------------------
$PY aggregate_eval.py \
  --group "Group1_control"  runs/cmp_control_8x8_seed1 runs/cmp_control_8x8_seed2 runs/cmp_control_8x8_seed3 \
  --group "Group2_shaped"   runs/cmp_shaped_8x8_seed1 runs/cmp_shaped_8x8_seed2 runs/cmp_shaped_8x8_seed3 \
  --group "Group3_fullobs"  runs/cmp_fullobs_8x8_seed1 runs/cmp_fullobs_8x8_seed2 runs/cmp_fullobs_8x8_seed3 \
  > runs/comparison_8x8_summary.txt
cat runs/comparison_8x8_summary.txt

# --- Determinism / no-regression cross-check --------------------------------
$PY compare_runs.py runs/baseline_doorkey8x8_seed1 runs/cmp_control_8x8_seed1 \
  | tee runs/control_vs_baseline_check.txt || true
echo
echo "Done. Per-seed numbers are in each run's eval.json; the across-seed spread"
echo "is in runs/comparison_8x8_summary.txt."
