# MiniGrid DoorKey: why PPO fails at 8×8, and what fixes it

**CAP 6671, Fall 2026 · Hubert Yakpé Trolindo** · repository `ucf-cap6671/fall2026-doorkey-cybsecoff31`

This repository holds the course PPO baseline for Farama MiniGrid and the first part of my DoorKey project (up to the midterm). It contains:

1. a **baseline replication** on DoorKey 5×5, 6×6, 8×8 and 16×16;
2. a **failure analysis** of the 8×8 collapse;
3. a **controlled comparison** on 8×8: control against two interventions (milestone reward shaping, full observability), three seeds each.

The full write-up is [`MiniGrid_DoorKey_Midterm_Report-1.pdf`](MiniGrid_DoorKey_Midterm_Report-1.pdf). Every number below is read from the run records committed under `runs/` (`eval.json`, `metrics.csv`).

---

## Results at a glance

### 1. Baseline replication (unchanged PPO, seed 1)

All runs use the course evaluation protocol: 200 episodes, evaluation seed 12345 (episode *i* is played on layout seed 12345 + *i*), sampled actions, and success means return > 0. "Random" is a uniform-random policy on the same layouts.

| Grid | Budget | Trained success | Trained return | Trained length | Random success | Random return | Random length | Run (`runs/…`) |
|---|---|---|---|---|---|---|---|---|
| 5×5 | 500k | **100%** | 0.959 | 11.5 | 6.0% | 0.025 | 244.7 | `baseline_doorkey5x5_seed1` |
| 6×6 | 1.0M | **100%** | 0.968 | 12.9 | 4.0% | 0.017 | 354.9 | `baseline_doorkey6x6_seed1` |
| 8×8 | 1.0M | **0%** | 0.000 | 640.0 | 1.5% | 0.007 | 636.4 | `baseline_doorkey8x8_seed1` |
| 16×16 | 1.0M | **0%** | 0.000 | 2560.0 | 1.5% | 0.005 | 2549.7 | `baseline_doorkey16x16_seed1` |

Success rate and return collapse at 8×8. Episode length hits the truncation cap (10·N² steps: 640 at 8×8, 2,560 at 16×16), so the trained agent times out on every evaluation episode and does *worse* than random.

### 2. Why it fails at 8×8 (summary of report §2)

- **Behaviour.** I decoded the recorded evaluation GIFs (`cell_visits_from_gifs.py`). The trained agent picks up the key in 5/5 episodes but never unlocks the door (0/5). It stays on 6–12 cells and spends on average 42% of the 640 steps on a single cell next to the key's starting position. The random policy covers 15–20 cells.
- **Training.** There are only 6 successes in 1M steps, all before step 202k, and none in the last ≈800k steps, where chance alone would predict about 19 (`episodes_over_training.py`). Entropy stays high (1.66 at the end, against 0.02 at 6×6), and final explained variance is −2.5 (0.81 at 6×6).
- **Diagnosis.** This is a **sparse-reward credit-assignment failure, with partial observability making it worse**. The bottleneck is the door step. With γ = 0.99 and λ = 0.95, credit decays by 0.94 per step (it halves every ≈11 steps). Six rewards cannot anchor the critic, so the advantages are noise and the policy never commits to a behaviour.

### 3. Controlled comparison on 8×8 (3 seeds, 1M steps, same PPO and hyperparameters)

| Group | Change | Success s1 / s2 / s3 | Mean ± sd [min–max] | Return / length | Stays > 0.9 from | Final entropy | Final expl. var. |
|---|---|---|---|---|---|---|---|
| **G1 control** `cmp_control_8x8_seed*` | none | 0 / 0 / 100% | 33.3 ± 47.1% [0–100] | 0.321 / 435.5 | — / — / 677k | 1.7 / 1.8 / 0.2 | −2.50 / −3.51 / 0.65 |
| **G2 shaped** `cmp_shaped_8x8_seed*` | +0.2 key, +0.3 door (training only) | 100 / 100 / 100% | 100.0 ± 0.0% | 0.974 / 18.7 | 131k / 103k / 123k | 0.02 / 0.01 / 0.02 | 0.95 / 0.95 / 0.96 |
| **G3 global view** `cmp_fullobs_8x8_seed*` | full N×N grid, sparse reward | 7 / 5.5 / 3.5% | 5.3 ± 1.4% [3.5–7.0] | 0.024 / 626.6 | — / — / — | 1.9 / 1.8 / 1.8 | 0.51 / 0.21 / 0.62 |

The random policy reaches 1.5% on the same layouts. sd is the population standard deviation across seeds (`aggregate_eval.py`, saved in `runs/comparison_8x8_summary.txt`). Entropy and explained variance are the mean of the last 20 updates in `metrics.csv`.

**How to read it.** The decision rule was fixed before the runs: one group is "better" only if its worst seed beats the other group's best seed. Under that rule neither comparison is decisive yet, because control seed 3 also solved the task. Descriptively, though, G2 solves 3/3 seeds against 1/3 for the control, and does so 5–6× faster. G3 makes the critic informative but the policy still never commits. This is Case 2 of the proposal: **temporal credit deficit dominates**. With three seeds, the result is provisional.

**Validity checks.**
- `cmp_control_8x8_seed1` reproduces `baseline_doorkey8x8_seed1` bit for bit: `metrics.csv` and `episodes.csv` are identical, and only the `exp_name` label differs (`runs/control_vs_baseline_check.txt`).
- Shaped runs log only sparse returns.
- Every group is evaluated without shaping (`reward_shaping_at_eval: false` in `eval.json`).

### Next steps (final report)

- ≥ 8 seeds per group, to test the gap and estimate how often the control escapes.
- A shaping ablation (door-only against key-only bonus).
- G2 at 16×16.
- G2 + G3 combined (Case 3 check).

---

## Repository contents

| File | What it is |
|---|---|
| `ppo_minigrid.py` | Training: the whole PPO algorithm (CleanRL style), the `Agent` network, run recording, and the `fully_obs` / `reward_shaping` switches in `make_env` |
| `wrappers.py` | Intervention wrappers: `MilestoneReward` (G2) and `FullObsStateFix` (G3) |
| `eval_checkpoint.py` | Evaluates a checkpoint under the protocol, writes `eval.json`, can draw episodes as GIFs. Never applies shaping; rebuilds the full-grid observation for G3 checkpoints |
| `aggregate_eval.py` | Mean / sd / min / max of success across seeds, per group |
| `compare_runs.py` | Checks that two runs produced identical `metrics.csv` and `episodes.csv` |
| `run_smoke.sh` | Short end-to-end test on `MiniGrid-Empty-5x5-v0` |
| `run_comparison.sh` | The whole 8×8 comparison: tests → 9 training runs → evaluation → aggregation → determinism check |
| `tests/test_interventions.py` | Checks the intervention rules on the real MiniGrid, plus one end-to-end PPO iteration per group |
| `sb3_sanity.py` | Independent Stable-Baselines3 PPO reference, for debugging |
| `protocol_success_rate_by_grid.py` | Report Fig. 1: success by grid size, trained against random → `fig1_success.png` |
| `cell_visits_from_gifs.py` | Report Fig. 2: cell-visit heat maps and key/door events decoded from evaluation GIFs → `fig2_visits.png` |
| `episodes_over_training.py` | Report Fig. 5: every training episode's length and successes → `fig4_episodes8x8.png` |
| `comparison_curves.py` | Report Fig. 6: training return and entropy for the nine comparison runs → `fig6_comparison.png` |
| `conf/` | Hydra configuration (see below) |
| `environment.yml` | Conda environment, every package pinned |
| `runs/` | Text records of the 4 baseline and 9 comparison runs, plus the two summary files |
| `MiniGrid_DoorKey_Midterm_Report-1.pdf` | Midterm report |

---

## Setup

```bash
git clone <this repository> && cd <this repository>
conda env create -f environment.yml
conda activate cap6671
./run_smoke.sh
```

The environment is Python 3.11.15, gymnasium 1.3.0, minigrid 3.1.0, torch 2.13.0 (from PyPI, which includes Apple MPS support), hydra-core 1.3.2 and stable-baselines3 2.9.0. Transitive dependencies are pinned too, and the environment was rebuilt and checked on macOS / Apple silicon. `conda-forge` is named explicitly in the file because the build machine has no default channel configured.

`run_smoke.sh` trains on `MiniGrid-Empty-5x5-v0` for 100k steps on CPU (about 45 s). The return should pass 0.90 well before the end. If the script cannot import torch, activate the environment first or run `PYTHON=/path/to/python ./run_smoke.sh`. Any arguments you add are passed on as Hydra overrides, for example `./run_smoke.sh seed=7 ppo.ent_coef=0.02`.

To watch training live: `tensorboard --logdir runs`.

---

## Reproducing the results

### Baselines (report §1)

```bash
for g in 5x5 6x6 8x8 16x16; do
  python ppo_minigrid.py env=doorkey$g seed=1 device=cpu exp_name=baseline run_name=baseline_doorkey${g}_seed1
  python eval_checkpoint.py model_path=runs/baseline_doorkey${g}_seed1/model.pt policy.also_random=true
done
```

These were run at commit `97c4f89`. The exact command of each run is in `runs/<run>/command.txt`.

### Failure analysis (report §2)

```bash
python eval_checkpoint.py model_path=runs/baseline_doorkey8x8_seed1/model.pt policy.also_random=true render.enabled=true
python cell_visits_from_gifs.py                 # decodes render/{sampled,random}/ep000-004.gif
python episodes_over_training.py                # reads episodes.csv, prints the §2 numbers
```

### 8×8 comparison (report §3)

```bash
bash run_comparison.sh
```

The script refuses to start if the working tree is not clean, because every run's `git.json` would then be marked dirty. It runs `tests/test_interventions.py` first, then trains G1, G2 and G3 for seeds 1–3, evaluates all nine runs, writes `runs/comparison_8x8_summary.txt`, and checks `cmp_control_8x8_seed1` against the baseline. Each 8×8 run takes about 7 minutes on the original machine, so the full matrix takes a few hours. For a quicker preliminary pass, lower `env.total_timesteps` by the **same** amount for every group and say so in the report.

G1 and G2 were run at commit `55d359a`; its only untracked files were three figure PNGs. G3 was run at `5caf2f9`, a clean tree that adds a rollout-buffer shape fix for full-grid inputs. That fix has no effect on 7×7 runs, as the exact seed-1 reproduction shows.

### Figures

```bash
python protocol_success_rate_by_grid.py   # Fig. 1  (needs eval.json, committed)
python comparison_curves.py               # Fig. 6  (needs metrics.csv, committed)
python cell_visits_from_gifs.py           # Fig. 2  (needs render/*.gif, regenerate locally)
python episodes_over_training.py          # Fig. 5  (needs episodes.csv, regenerate locally)
```

Only the text records of each run are committed (`command.txt`, `git.json`, `config.json`, `metrics.csv`, `eval.json`). `episodes.csv`, `model.pt`, `tb/` and the GIFs come from re-running training or evaluation. Figure PNGs are gitignored (`/fig*.png`).

---

## The interventions

The rules come from the proposal and are enforced in `wrappers.py` and checked by `tests/test_interventions.py`. Across all three groups, the PPO algorithm, the hyperparameters and the 7-action space are unchanged.

| Group | Config | Observation | Training reward | Evaluation |
|---|---|---|---|---|
| G1 control | `env=doorkey8x8` | 7×7 egocentric view | sparse | sparse |
| G2 shaped | `env=doorkey8x8_shaped` | 7×7 egocentric view | sparse + `r_key: 0.2` + `r_door: 0.3` | sparse (no wrapper) |
| G3 global view | `env=doorkey8x8_fullobs` | full 8×8 grid | sparse | sparse, full grid rebuilt from the checkpoint |

**`MilestoneReward` (G2).** It pays +0.2 on the first step the agent holds the key, and +0.3 on the first step the door is open, but only after the key bonus has been paid. Each bonus is paid at most once per episode, so dropping and picking up the key again earns nothing. The environment's own sparse reward still applies. Episode statistics are recorded **before** the bonus is added (the order of wrappers in `make_env`), so `episodes.csv` and `metrics.csv` hold the sparse return for every group and the curves can be compared directly.

**`FullyObsWrapper` + `FullObsStateFix` (G3).** MiniGrid's `FullyObsWrapper` writes the agent's facing direction (0–3) into the *state* channel of the agent's cell. The encoder one-hot encodes that channel with only 3 classes, so facing up crashed the run, and directions 1–2 were silently read as door states. `FullObsStateFix` sets that one cell's state channel to 0. The direction still reaches the network through the `direction` input.

**Encoder change (all groups).** I added `nn.AdaptiveAvgPool2d((1, 1))` before the flatten. It has no parameters and does nothing on the 7×7 view, where the conv stack already ends at 1×1, so G1 and G2 are bit-identical to the baseline. On the 8×8 full view, the conv stack leaves a 2×2 map, which the pooling averages into the same 64 features. The rollout buffer now takes its image shape from the environment instead of a hard-coded 7×7×3.

---

## Configuration

All parameters are YAML files under `conf/`, merged by [Hydra](https://hydra.cc):

| File | What it holds |
|---|---|
| `conf/config.yaml` | Run-level settings: `exp_name`, `seed` (1), `device` (cpu), `torch_threads` (4), `runs_dir`, `run_name`, `save_model` |
| `conf/env/<name>.yaml` | The task: `env_id`, `total_timesteps`, and optionally `fully_obs`, `reward_shaping`, `r_key`, `r_door` |
| `conf/ppo/default.yaml` | PPO hyperparameters, each with a comment |
| `conf/eval.yaml`, `conf/sb3.yaml` | Options for `eval_checkpoint.py` and `sb3_sanity.py` |

### Environments

| Selected with | Env id | Budget | Note |
|---|---|---|---|
| `env=empty5x5` (default) | `MiniGrid-Empty-5x5-v0` | 100k | Smoke test |
| `env=doorkey5x5` | `MiniGrid-DoorKey-5x5-v0` | 500k | Budget fixed by handout |
| `env=doorkey6x6` | `MiniGrid-DoorKey-6x6-v0` | 1M | Budget fixed by handout |
| `env=doorkey8x8` | `MiniGrid-DoorKey-8x8-v0` | 1M | Budget fixed by handout; G1 control |
| `env=doorkey8x8_shaped` | `MiniGrid-DoorKey-8x8-v0` | 1M | G2: `reward_shaping: true`, `r_key: 0.2`, `r_door: 0.3` |
| `env=doorkey8x8_fullobs` | `MiniGrid-DoorKey-8x8-v0` | 1M | G3: `fully_obs: true` |
| `env=doorkey16x16` | `MiniGrid-DoorKey-16x16-v0` | 1M | Largest registered size; subject of the final report |

The number of steps actually run is rounded down to a whole number of rollout batches: 976 × 1,024 = 999,424 for a 1M budget.

### PPO hyperparameters (`conf/ppo/default.yaml`)

These are the CleanRL defaults for small discrete tasks, not tuned for DoorKey:

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| `num_envs` | 8 | `gamma` | 0.99 |
| `num_steps` | 128 (batch 1,024) | `gae_lambda` | 0.95 |
| `learning_rate` | 2.5e-4, `anneal_lr: true` | `clip_coef` | 0.2, `clip_vloss: true` |
| `num_minibatches` | 4 (256 samples) | `ent_coef` | 0.01 (constant) |
| `update_epochs` | 4 | `vf_coef` | 0.5 |
| `max_grad_norm` | 0.5 | `norm_adv` / `target_kl` | true / null |

Truncated episodes are bootstrapped: the value of the final observation, multiplied by γ, is added to the reward, so the critic does not learn that the time limit is worth zero.

### Overrides

```bash
python ppo_minigrid.py env=doorkey8x8 seed=2 ppo.ent_coef=0.02 ppo.anneal_lr=false
python ppo_minigrid.py env=doorkey8x8_shaped env.r_door=0.5        # change a shaping bonus
python ppo_minigrid.py env=doorkey8x8 --cfg job                     # print the resolved config and exit
python ppo_minigrid.py -m env=doorkey8x8 seed=1,2,3                 # sweep, one run directory per job
```

Booleans are written `true` / `false`, and unset is `null`. A key that does not exist is an error, so a typo cannot silently become an unused option. To add a hyperparameter, add it to the right YAML file and read it as `cfg.<group>.<name>`; the run record picks it up automatically.

---

## Observation and network

MiniGrid returns a symbolic dict, not pixels:

- **`image`**: `(7, 7, 3)` `uint8`, the patch in front of the agent (object id, colour id, state id). The shape is the same at every grid size, except under G3, where it is `(N, N, 3)`.
- **`direction`**: `Discrete(4)`.
- **`mission`**: a constant string. The `DropMission` wrapper removes it, which also makes vectorised envs work.

The `Agent` does the following:
- one-hot encodes the image into 20 channels (11 objects + 6 colours + 3 states);
- passes it through three 3×3 convolutions (32 → 64 → 64), `AdaptiveAvgPool2d(1)` and a flatten, giving 64 features;
- concatenates a one-hot direction (4);
- uses a 128-unit Tanh trunk, then an actor head (7 logits) and a critic head (1 value).

---

## Run records

Every run writes to `runs/<run_name>/`. By default the name is `<env_id>__<exp_name>__seed<N>__<utc timestamp>`.

| File | Content |
|---|---|
| `command.txt` | Exact interpreter and command line |
| `git.json` | Commit, branch, `git describe`, dirty flag and `git status --porcelain` |
| `config.json` | Resolved config plus derived batch sizes |
| `env_info.json` | Package versions, platform, threads/device, timing, steps/s |
| `pip_freeze.txt`, `conda_env_export.yml` | Full environment |
| `metrics.csv` | One row per PPO update |
| `episodes.csv` | Return and length of every finished training episode (sparse return, for every group) |
| `tb/` | TensorBoard events |
| `model.pt` | Final weights and config |
| `eval.json` / `eval-argmax.json` | Written by `eval_checkpoint.py` |
| `render/<policy>/ep<i>.gif` | Evaluation GIFs, when rendering is on |

If the working tree has uncommitted or untracked changes, the run prints a warning and marks `git.json` dirty. `runs/` is gitignored; to commit a run's text records, force-add them:

```bash
git add -f runs/<name>/{command.txt,git.json,config.json,metrics.csv,eval.json}
```

### Determinism

Two runs with the same seed, machine and thread count give byte-identical `metrics.csv` and `episodes.csv`:

```bash
python ppo_minigrid.py env=empty5x5 env.total_timesteps=20000 seed=3 run_name=repro_a
python ppo_minigrid.py env=empty5x5 env.total_timesteps=20000 seed=3 run_name=repro_b
python compare_runs.py runs/repro_a runs/repro_b
```

Determinism relies on:
- one seed for `random`, numpy and torch, with sub-environment *i* reset with `seed + i`;
- `torch.use_deterministic_algorithms(True, warn_only=True)`;
- a fixed `torch_threads: 4`.

`compare_runs.py` ignores `wall_time_s` and `sps`, and reports any difference in `config.json`. That is why `runs/control_vs_baseline_check.txt` says "DO NOT match": only the `exp_name` label differs, and both CSVs are identical.

### Device

CPU is the default, and it is the device the determinism guarantee covers. `device=mps` and `device=cuda` also work, and `device=auto` picks CUDA, then MPS, then CPU.

---

## Metrics

`metrics.csv` and TensorBoard record:
- episode return and length;
- policy loss, value loss, entropy;
- approximate KL, clip fraction, explained variance;
- learning rate, steps/s.

How to read the main ones, as they showed up in this project:

- **`episodic_return_mean`**: 0 until successes appear. At 8×8 it stays flat for the failed seeds.
- **`entropy`**: starts near ln 7 ≈ 1.95. At 6×6 it collapses to ≈0.02 once the task is learned. At 8×8 it stays high (≈1.7) with no reward to commit to; this is the "high-entropy repetitive loop" failure.
- **`explained_variance`**: near or below 0 means the critic explains nothing (−2.5 in the failed 8×8 runs). With shaping it reaches 0.95.
- **`approx_kl`** / **`clipfrac`**: how much each update moves the policy.

---

## Evaluating a checkpoint

```bash
python eval_checkpoint.py model_path=runs/<run>/model.pt policy.also_random=true
```

The defaults in `conf/eval.yaml` are the course protocol: `episodes: 200`, `seed: 12345`, `policy.mode: sampled`. Leave them unchanged for any number you report.

| Option | Default | What it does |
|---|---|---|
| `model_path` | required | The `model.pt` to evaluate |
| `env_id` | `null` | `null` uses the environment recorded in the checkpoint |
| `episodes` / `seed` | `200` / `12345` | Episode *i* is played on layout `seed + i` |
| `device` | `cpu` | Where the policy runs |
| `policy.mode` | `sampled` | `argmax` writes `eval-argmax.json` instead, so it never overwrites the protocol result |
| `policy.also_random` | `false` | Also score a uniform-random policy on the same layouts (needed for the report) |
| `render.enabled` | `false` | Save GIFs under `runs/<run>/render/{sampled,argmax,random}/ep<i>.gif` |
| `render.first` / `render.only` | `5` / `null` | First N episodes, or exact indices such as `'render.only=[0,17,199]'` |
| `render.highlight` | `true` | Shade the 7×7 patch the agent sees |
| `render.tile_size` / `render.frame_ms` | `32` / `200` | Pixels per cell / ms per frame (`cell_visits_from_gifs.py` assumes these values) |

`eval.json` also records `fully_obs` and `reward_shaping_at_eval` (always `false`). Rendering always runs the full 200-episode evaluation. Use `render.only` to pick particular episodes; do not shorten `episodes`. If you change the network, the observation encoding or the action space, update the loading code in `eval_checkpoint.py`.

## Independent sanity check

`sb3_sanity.py` runs Stable-Baselines3 PPO with a different observation pipeline (`ImgObsWrapper` + MLP policy). If SB3 learns and `ppo_minigrid.py` does not, the bug is in our code.

```bash
python sb3_sanity.py env_id=MiniGrid-Empty-5x5-v0 total_timesteps=100000
```

---

## AI use

As disclosed in the report, Claude (Anthropic) was used to parse run records and recorded episodes, compute the §1–§2 statistics and figures, review and fix the intervention code and tests, and draft the report. Gemini (Google) was used to rephrase and proofread the report. Every number is read from run artifacts.

## References

- Chevalier-Boisvert, M. et al. (2023). *Minigrid & Miniworld: Modular & customizable reinforcement learning environments for goal-oriented tasks.* NeurIPS Datasets and Benchmarks.
- Huang, S. et al. (2022). *CleanRL: High-quality single-file implementations of deep RL algorithms.* JMLR 23(274).
- Ng, A. Y., Harada, D., Russell, S. (1999). *Policy invariance under reward transformations: Theory and application to reward shaping.* ICML.
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
