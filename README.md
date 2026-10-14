# CAP 6671 reinforcement learning baseline

This is the course baseline for CAP 6671. No measured results are provided here.

The baseline is a single-file PPO implementation for Farama MiniGrid. The algorithm is in `ppo_minigrid.py`, following the CleanRL style: no framework, no class hierarchy. The hyperparameters live in `conf/`, one YAML file per group, read by [Hydra](https://hydra.cc); every parameter is defined and explained where it is declared. The code is meant to be read and modified.

## Repository contents

| File | What it is |
|---|---|
| `ppo_minigrid.py` | Training: the whole PPO algorithm, the `Agent` network, and run recording |
| `eval_checkpoint.py` | Evaluates a saved checkpoint, writes `eval.json`, optionally draws episodes as GIFs |
| `sb3_sanity.py` | Independent Stable-Baselines3 PPO reference, for debugging |
| `compare_runs.py` | Checks that two runs produced identical metrics |
| `run_smoke.sh` | Short end-to-end test on `MiniGrid-Empty-5x5-v0` |
| `conf/` | Hydra configuration |
| `environment.yml` | Conda environment with pinned package versions |

## From a fresh clone to a running smoke test

```bash
git clone <your repository> && cd <your repository>
conda env create -f environment.yml
conda activate cap6671
./run_smoke.sh
```

`run_smoke.sh` trains on `MiniGrid-Empty-5x5-v0` for 100k steps on CPU. The return column should pass 0.90 well before the run ends. Everything produced by the run is saved under `runs/<run name>/`. If the script says it cannot import torch, activate the environment first or run it as `PYTHON=/path/to/python ./run_smoke.sh`.

To watch the curves live:

```bash
tensorboard --logdir runs
```

Arguments passed to `run_smoke.sh` are forwarded to `ppo_minigrid.py`, so this works:

```bash
./run_smoke.sh seed=7 ppo.ent_coef=0.02
```

## Configuration

All parameters are YAML files under `conf/`, merged by Hydra:

| File | What it holds | In the code |
|---|---|---|
| `conf/config.yaml` | Run-level settings: `exp_name`, `seed`, `device`, `torch_threads`, `runs_dir`, `run_name`, `save_model` | `cfg.seed` |
| `conf/env/<name>.yaml` | The task: `env_id` and its `total_timesteps` budget. One file per task, `env=doorkey8x8` selects it | `cfg.env.env_id` |
| `conf/ppo/default.yaml` | The PPO hyperparameters, each with a comment saying what it does | `cfg.ppo.clip_vloss` |
| `conf/eval.yaml`, `conf/sb3.yaml` | Options of `eval_checkpoint.py` and `sb3_sanity.py` | |

Override any value on the command line as `key=value`, with the group as prefix:

```bash
python ppo_minigrid.py env=doorkey8x8 seed=2 ppo.ent_coef=0.02 ppo.anneal_lr=false ppo.target_kl=0.02
python ppo_minigrid.py env=doorkey8x8 --cfg job     # print the resolved config and exit
python ppo_minigrid.py --help
```

Booleans are `true` / `false`, unset is `null`. A key that does not exist is an error (`Key 'ent_coeff' is not in struct`), so a typo cannot silently become an unused option. To add a hyperparameter, add it to the YAML file it belongs to and read it as `cfg.ppo.<name>` in the code; the run record picks it up automatically.

`-m` runs a sweep, one job after another, each with its own run directory:

```bash
python ppo_minigrid.py -m env=doorkey8x8 seed=1,2,3
```

## Environments

| Env id | Selected with | Note |
|---|---|---|
| `MiniGrid-Empty-5x5-v0` | `env=empty5x5` | Empty room, walk to the green square. Smoke test: if this does not learn, something is broken |
| `MiniGrid-DoorKey-5x5-v0` | `env=doorkey5x5` | |
| `MiniGrid-DoorKey-6x6-v0` | `env=doorkey6x6` | |
| `MiniGrid-DoorKey-8x8-v0` | `env=doorkey8x8` | |
| `MiniGrid-DoorKey-16x16-v0` | `env=doorkey16x16` | The largest registered DoorKey size |

Each config file sets the `env_id` and the default `total_timesteps` for that task:

```bash
python ppo_minigrid.py env=doorkey5x5 seed=1
python ppo_minigrid.py env=doorkey6x6 seed=1
python ppo_minigrid.py env=doorkey8x8 seed=1
python ppo_minigrid.py env=doorkey16x16 seed=1
```

## Observation format and baseline encoding

MiniGrid does not return pixels. Each step returns a dict:

- **`image`**: a `(7, 7, 3)` `uint8` symbolic view of the 7x7 patch in front of the agent. Channel 0 is object id, channel 1 colour id, and channel 2 state id (`open`, `closed`, `locked`). The shape is the same at every grid size.
- **`direction`**: `Discrete(4)`, giving the direction the agent faces.
- **`mission`**: a string such as `"use the key to open the door and then get to the goal"`.

The baseline handles them as follows:

- **`image`**: one-hot + CNN. The three categorical channels are one-hot encoded into a 20-channel 7x7 tensor (11 object ids + 6 colours + 3 states), then passed through three 3x3 convolutions to produce a 64-dimensional vector. The ids are categorical, so their raw integer ordering should not be treated as meaningful.
- **`direction`**: one-hot. The four-dimensional encoding is concatenated with the CNN features. Direction matters because the observation is egocentric.
- **`mission`**: dropped by the `DropMission` wrapper, since the string is constant within one DoorKey environment id.

`DropMission` also makes the observation compatible with Gymnasium vector environments, which cannot batch MiniGrid's `MissionSpace`. If you add a text encoder, you will need another way around this.

## Run records

Every run creates

```
runs/<env>__<exp>__seed<N>__<utc timestamp>/
```

with the following files:

| File | What it is |
|---|---|
| `command.txt` | Exact interpreter and command line |
| `git.json` | Commit hash, branch, `git describe`, and dirty-tree status |
| `config.json` | Resolved config (`env`, `ppo`, run-level), plus the derived batch sizes |
| `env_info.json` | Package versions, platform, CPU/thread/device information, timing, and final steps/s |
| `pip_freeze.txt` | Full package list |
| `conda_env_export.yml` | Full conda environment export |
| `metrics.csv` | PPO update metrics |
| `episodes.csv` | Return and length of each completed episode |
| `tb/` | TensorBoard event files |
| `model.pt` | Final weights and configuration |
| `eval.json` | Written later by `eval_checkpoint.py`, see below. `eval-argmax.json` if the greedy policy was evaluated |
| `render/` | GIFs of evaluation episodes, when rendering was on |

If the working tree has uncommitted changes when a run starts, the run prints a warning and `git.json` marks the tree as dirty, because the recorded commit no longer fully describes the code that ran.

`runs/` is gitignored. To put a run's text records into git, force-add them and leave the binaries (`model.pt`, `tb/`) out:

```bash
git add -f runs/<name>/{command.txt,git.json,config.json,metrics.csv,eval.json}
```

### Checking determinism

On the same machine, two runs with the same seed should produce byte-identical `metrics.csv` and `episodes.csv`:

```bash
python ppo_minigrid.py env=empty5x5 env.total_timesteps=20000 seed=3 run_name=repro_a
python ppo_minigrid.py env=empty5x5 env.total_timesteps=20000 seed=3 run_name=repro_b
python compare_runs.py runs/repro_a runs/repro_b
```

`compare_runs.py` ignores `wall_time_s` and `sps` and compares the remaining values cell by cell.

The implementation uses:

- one `seed` for random, numpy, and torch, with each sub-environment reset using `seed + i`;
- `torch.use_deterministic_algorithms(True, warn_only=True)`;
- `torch_threads: 4` by default, recorded with each run.

CPU thread count can change floating-point reduction order, so use the same thread count when checking determinism. Exact numerical agreement is only expected on the same software and hardware environment.

## Device

CPU is the default. You can also use `device=mps`, while `device=auto` selects CUDA, then MPS, then CPU.

For this small network, device speed depends on the machine. Measure it if runtime matters.

## Metrics

`metrics.csv` and TensorBoard record episode return and length, policy and value loss, entropy, steps per second, approximate KL, clip fraction, explained variance, and learning rate.

A few useful quantities:

- **`episodic_return_mean`**: stays at 0 until successful episodes appear, then rises as success becomes more frequent and efficient.
- **`entropy`**: starts near `ln(7) = 1.95` for a nearly uniform seven-action policy. Very low entropy while return remains at 0 can indicate that the policy committed before finding reward.
- **`explained_variance`**: near 0 means the value function explains little of the variation in returns. This is common early in sparse-reward training.
- **`approx_kl`** and **`clipfrac`**: measure how strongly a PPO update changes the policy. Large values are a reason to inspect the update size and learning rate.

## Independent sanity check

`sb3_sanity.py` runs Stable-Baselines3 PPO on the same environment. You should not need to modify it.

Use it when the baseline does not learn and you want to distinguish an implementation problem from an environment/setup problem. It deliberately uses a different observation pipeline (`ImgObsWrapper` with an MLP policy), so it is independent of the course PPO implementation.

```bash
python sb3_sanity.py env_id=MiniGrid-Empty-5x5-v0 total_timesteps=100000
```

## Evaluating a checkpoint

```bash
python eval_checkpoint.py model_path=runs/<run>/model.pt policy.also_random=true
```

This reports mean return, mean episode length, and success rate for the trained policy and, with `policy.also_random=true`, for a uniform-random policy on the same layouts. It writes `eval.json` next to the checkpoint.

The options are in `conf/eval.yaml`, in three levels. The top level is the protocol: which checkpoint plays how many episodes on which layouts. `policy` says which policies play them. `render` says which of those episodes are also drawn. `python eval_checkpoint.py --cfg job` prints the merged options; every value is overridden as `key=value`, exactly like training.

| Option | Default | What it does |
|---|---|---|
| `model_path` | required | The `model.pt` to evaluate |
| `env_id` | `null` | Environment to evaluate on; `null` uses the one recorded in the checkpoint |
| `episodes` | `200` | Number of evaluation episodes |
| `seed` | `12345` | Episode `i` is played on layout seed `seed + i`; also seeds torch, so the sampled policy is reproducible |
| `device` | `cpu` | Where the policy runs |
| `policy.mode` | `sampled` | `sampled` draws actions from the policy's distribution and writes `eval.json`. `argmax` takes the most likely action and writes `eval-argmax.json` instead, so it never overwrites the protocol result |
| `policy.also_random` | `false` | Also play a uniform-random policy on the same layouts and report it |
| `render.enabled` | `false` | Save the chosen episodes as GIFs, one file per episode |
| `render.first` | `5` | Render episodes `0` to `first - 1` |
| `render.only` | `null` | Render exactly these episode indices instead of the first ones, e.g. `'render.only=[0,17,199]'` (quote it, the shell treats brackets specially) |
| `render.highlight` | `true` | Shade the 7x7 patch the agent sees |
| `render.tile_size` | `32` | Pixels per grid cell |
| `render.frame_ms` | `200` | Milliseconds per frame |

The defaults for `episodes`, `seed`, and `policy.mode` are the course's fixed evaluation settings. Leave them unchanged for any number you report. `policy.also_random` is off by default because the random policy does not depend on your checkpoint; turn it on whenever you need the random baseline on the same layouts, which the report does.

A GIF is one evaluation episode from reset to its end, success or timeout, one frame per step. The frame is the whole grid seen from above: the agent is the red triangle, the lighter cells are the 7x7 patch it can see (`render.highlight`), so you can watch it act on what is in view and miss what is not. Rendering reads frames and changes no number. Files land under the run directory by policy and episode index, so different modes never overwrite each other:

```
runs/<run>/render/sampled/ep000.gif     # policy.mode=sampled
runs/<run>/render/argmax/ep017.gif      # policy.mode=argmax
runs/<run>/render/random/ep000.gif      # policy.also_random=true
```

```bash
python eval_checkpoint.py model_path=runs/<run>/model.pt render.enabled=true                            # first five episodes of the trained policy
python eval_checkpoint.py model_path=runs/<run>/model.pt render.enabled=true policy.also_random=true    # the same five, trained and random
python eval_checkpoint.py model_path=runs/<run>/model.pt render.enabled=true 'render.only=[3,42]'       # these two episodes
python eval_checkpoint.py model_path=runs/<run>/model.pt render.enabled=true policy.mode=argmax         # the greedy policy, into render/argmax/
```

Rendering runs the full evaluation and rewrites `eval.json` (or `eval-argmax.json`) every time. Do not shorten `episodes` for a quick look at a GIF: `render.only` picks the episode without changing the protocol, and the 200 episodes take about a minute on the largest grid.

The script loads the checkpoint into the stock `Agent`. If you change the network, the observation encoding, or the action space, adapt the loading code in `eval_checkpoint.py` to build your agent instead.
