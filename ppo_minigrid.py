"""
Single-file PPO for Farama MiniGrid, CleanRL style.

CAP 6671 baseline. Everything the algorithm does is in this file, on purpose:
students are asked to read it, form a hypothesis about what is holding it back,
change it, and measure the difference. Nothing here is a library call into an
RL framework.

Run:
    python ppo_minigrid.py env=empty5x5
    python ppo_minigrid.py env=doorkey8x8 seed=2 ppo.ent_coef=0.02

Every parameter is declared and explained in conf/ (conf/config.yaml, conf/env/,
conf/ppo/default.yaml); Hydra merges those files and the key=value overrides
from the command line into `cfg`. `python ppo_minigrid.py --cfg job` prints the
result. Every run writes runs/<name>/ with the command line, the git commit, the
resolved config, the package list, and the metric log. See README.md.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf, open_dict

import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv, AutoresetMode

import minigrid  # noqa: F401  (registers the MiniGrid-* env ids)
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
# MiniGrid hands back a dict with three entries:
#
#   image     (7, 7, 3) uint8. NOT pixels. It is a symbolic view of the 7x7
#             patch in front of the agent. Channel 0 is an object id, channel 1
#             a colour id, channel 2 a state id (a door being open/closed/locked).
#   direction Discrete(4), which way the agent faces. The image is egocentric
#             but the goal position is not, so this matters.
#   mission   a string, e.g. "use the key to open the door and then get to the
#             goal".
#
# What this baseline does with them:
#
#   image     one-hot each of the three channels and stack them into a
#             (11 + 6 + 3) = 20 channel 7x7 tensor, then a 3-layer CNN. One-hot
#             rather than feeding the raw integers, because the ids are
#             categorical: object id 9 (lava) is not "three more than" id 6
#             (ball), and a conv over the raw ints has to waste capacity
#             unlearning that ordering.
#   direction one-hot to 4 dims, concatenated onto the flattened CNN features.
#   mission   DROPPED. Each env id here has exactly one mission string, so it
#             carries zero bits and any encoder for it would be dead weight.
#             This is a real limitation, not a free lunch: it is why this
#             baseline cannot be pointed at a multi-task MiniGrid set such as
#             BabyAI without adding a text encoder first. Doing that is a
#             legitimate project.

NUM_OBJECT = len(OBJECT_TO_IDX)   # 11
NUM_COLOR = len(COLOR_TO_IDX)     # 6
NUM_STATE = len(STATE_TO_IDX)     # 3
IMAGE_CHANNELS = NUM_OBJECT + NUM_COLOR + NUM_STATE
NUM_DIRECTIONS = 4


class DropMission(gym.ObservationWrapper):
    """Keep `image` and `direction`, drop the mission string.

    Also needed mechanically: gymnasium's vector envs cannot batch MiniGrid's
    MissionSpace, so the string has to go before SyncVectorEnv sees it.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Dict(
            {
                "image": env.observation_space["image"],
                "direction": env.observation_space["direction"],
            }
        )

    def observation(self, obs):
        return {"image": obs["image"], "direction": obs["direction"]}


def encode_obs(image: torch.Tensor, direction: torch.Tensor):
    """(B,7,7,3) uint8 and (B,) int -> (B,20,7,7) float and (B,4) float."""
    idx = image.long()
    onehot = torch.cat(
        [
            F.one_hot(idx[..., 0], NUM_OBJECT),
            F.one_hot(idx[..., 1], NUM_COLOR),
            F.one_hot(idx[..., 2], NUM_STATE),
        ],
        dim=-1,
    ).float()
    img = onehot.permute(0, 3, 1, 2).contiguous()
    dirs = F.one_hot(direction.long(), NUM_DIRECTIONS).float()
    return img, dirs

"""
def make_env(env_id: str, idx: int, record_stats: bool = True):
    def thunk():
        env = gym.make(env_id)
        env = DropMission(env)
        if record_stats:
            env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

This is the initial make_env function
"""

"""
The following is the complete  new version of make_env 
"""

def make_env(env_id: str, idx: int, record_stats: bool = True,
             fully_obs: bool = False, reward_shaping: bool = False,
             r_key: float = 0.2, r_door: float = 0.3):
    """Build one DoorKey environment.

    The two interventions from the project proposal enter here, and nowhere else,
    so the PPO code stays a pure control:
      fully_obs       Group 3. Wrap with MiniGrid's FullyObsWrapper so `image`
                      becomes the whole (N,N,3) grid instead of the 7x7 patch.
                      Rewards stay sparse. The CNN handles the larger input
                      because Agent pools to 1x1 before the linear trunk.
      reward_shaping  Group 2. Wrap with MilestoneReward (+r_key, +r_door),
                      applied at training time only. Evaluation builds the bare
                      env, so the agent is always scored on the sparse objective.
    Both default off, so the untouched call reproduces the baseline exactly.
    """
    def thunk():
        env = gym.make(env_id)
        if fully_obs:
            from minigrid.wrappers import FullyObsWrapper
            from wrappers import FullObsStateFix
            env = FullyObsWrapper(env)
            # FullyObsWrapper stores the agent's direction (0..3) in the state
            # channel, which the encoder one-hots with only 3 classes. Zero it.
            env = FullObsStateFix(env)
        env = DropMission(env)
        # Episode statistics are recorded BEFORE the shaping bonus is added, so
        # episodes.csv / metrics.csv log the environment's true sparse return
        # for every group and the learning curves stay comparable. PPO itself
        # still trains on the shaped reward (the outer wrapper).
        if record_stats:
            env = gym.wrappers.RecordEpisodeStatistics(env)
        if reward_shaping:
            from wrappers import MilestoneReward
            env = MilestoneReward(env, r_key=r_key, r_door=r_door)
        return env

    return thunk


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Shared conv encoder, separate linear heads for policy and value.

    Small on purpose. The whole point of the assignment is that this is the
    thing you get to change: width, depth, whether the heads share a trunk,
    whether direction enters at the conv or at the MLP.
    """

    def __init__(self, num_actions: int, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(IMAGE_CHANNELS, 32, kernel_size=3)),  # 7 -> 5
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3)),              # 5 -> 3
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3)),              # 3 -> 1
           
	    # nn.ReLU(),
            # nn.Flatten(),
	    
	    # The previous two lines nn.ReLU(), and nn.Flatten(), are replaced by the following lines
	   
	    nn.ReLU(),
            # Collapse whatever spatial size the conv stack leaves to 1x1. On the
            # 7x7 partial view the stack already outputs 1x1, so this is the
            # identity and Groups 1 and 2 are bit-for-bit the baseline. On the
            # Group 3 full view (e.g. 8x8 -> 6 -> 4 -> 2) it averages the 2x2 map
            # to 64 features, so the SAME trunk/heads and parameter count are
            # reused with no change to the action space. This is the only
            # observation-encoding accommodation the global-view group requires.
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
 
        )
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(64 + NUM_DIRECTIONS, hidden)),
            nn.Tanh(),
        )
        self.actor = layer_init(nn.Linear(hidden, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)

    def features(self, image, direction):
        img, dirs = encode_obs(image, direction)
        h = self.encoder(img)
        return self.trunk(torch.cat([h, dirs], dim=1))

    def get_value(self, image, direction):
        return self.critic(self.features(image, direction)).squeeze(-1)

    def get_action_and_value(self, image, direction, action=None):
        h = self.features(image, direction)
        logits = self.actor(h)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# There is no Config class. The parameters live in conf/ and arrive here as an
# OmegaConf DictConfig with three levels:
#
#   cfg.env.*   task: env_id, total_timesteps            (conf/env/<name>.yaml)
#   cfg.ppo.*   algorithm hyperparameters                (conf/ppo/default.yaml)
#   cfg.*       run-level: exp_name, seed, device, ...   (conf/config.yaml)
#
# finalize() checks the values and adds the derived sizes at the top level.

DEVICES = ("cpu", "mps", "cuda", "auto")


def finalize(cfg: DictConfig) -> DictConfig:
    if cfg.device not in DEVICES:
        raise SystemExit(f"device={cfg.device!r} is not one of {DEVICES}")
    batch_size = cfg.ppo.num_envs * cfg.ppo.num_steps
    num_iterations = cfg.env.total_timesteps // batch_size
    if num_iterations < 1:
        raise SystemExit(
            f"env.total_timesteps={cfg.env.total_timesteps} is smaller than one rollout batch "
            f"({cfg.ppo.num_envs} envs x {cfg.ppo.num_steps} steps = {batch_size}). "
            "Nothing would train. Raise the budget or lower ppo.num_envs / ppo.num_steps."
        )
    with open_dict(cfg):  # the composed config is closed to new keys; these are derived, not options
        cfg.batch_size = batch_size
        cfg.minibatch_size = batch_size // cfg.ppo.num_minibatches
        cfg.num_iterations = num_iterations
    return cfg


def config_dict(cfg: DictConfig) -> dict:
    """The resolved config as plain nested dicts, for config.json and model.pt."""
    return OmegaConf.to_container(cfg, resolve=True)


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


# ---------------------------------------------------------------------------
# Reproducibility record
# ---------------------------------------------------------------------------


def _run(cmd, cwd=None):
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120)
        return out.stdout.strip() if out.returncode == 0 else f"<failed: {out.stderr.strip()[:400]}>"
    except Exception as exc:  # noqa: BLE001
        return f"<failed: {exc}>"


def git_record(repo_dir: str) -> dict:
    porcelain = _run(["git", "status", "--porcelain"], cwd=repo_dir)
    return {
        "commit": _run(["git", "rev-parse", "HEAD"], cwd=repo_dir),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir),
        "describe": _run(["git", "describe", "--always", "--dirty"], cwd=repo_dir),
        "dirty": bool(porcelain) and not porcelain.startswith("<failed"),
        "status_porcelain": porcelain,
    }


def write_record(run_dir: str, cfg: DictConfig, argv: list[str], device: str) -> dict:
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    git = git_record(repo_dir)

    with open(os.path.join(run_dir, "command.txt"), "w") as f:
        f.write(shlex.join([sys.executable, os.path.abspath(__file__)] + argv) + "\n")

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_dict(cfg), f, indent=2, sort_keys=True)

    with open(os.path.join(run_dir, "git.json"), "w") as f:
        json.dump(git, f, indent=2, sort_keys=True)

    with open(os.path.join(run_dir, "pip_freeze.txt"), "w") as f:
        f.write(_run([sys.executable, "-m", "pip", "freeze"]) + "\n")

    conda_export = _run(["conda", "env", "export", "-p", sys.prefix])
    with open(os.path.join(run_dir, "conda_env_export.yml"), "w") as f:
        f.write(conda_export + "\n")

    env_info = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gymnasium": gym.__version__,
        "minigrid": getattr(minigrid, "__version__", "unknown"),
        "device_requested": cfg.device,
        "device_resolved": device,
        "torch_threads": torch.get_num_threads(),
        "mps_available": bool(torch.backends.mps.is_available()),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(run_dir, "env_info.json"), "w") as f:
        json.dump(env_info, f, indent=2, sort_keys=True)

    return git


def resolve_device(name: str) -> str:
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("device=mps requested but torch.backends.mps.is_available() is False")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("device=cuda requested but no CUDA device is visible")
    return name


def seed_everything(seed: int):
    """Seed every generator this program draws from.

    Python's `random`, numpy, torch. Env seeds are set at reset(), one per
    sub-env, derived from this seed. `use_deterministic_algorithms(warn_only)`
    picks deterministic kernels where torch has one and warns instead of
    crashing where it does not.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Note: PYTHONHASHSEED is deliberately not set here. It only takes effect
    # before the interpreter starts, so setting it now would look like a
    # guarantee without being one. Nothing in this file depends on hash order.


METRIC_FIELDS = [
    "iteration",
    "global_step",
    "wall_time_s",
    "sps",
    "learning_rate",
    "episodes_this_iter",
    "episodic_return_mean",
    "episodic_length_mean",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clipfrac",
    "explained_variance",
]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(cfg: DictConfig, argv: list[str]) -> str:
    device_name = resolve_device(cfg.device)
    device = torch.device(device_name)
    if cfg.torch_threads > 0:
        torch.set_num_threads(cfg.torch_threads)

    if not cfg.run_name:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        cfg.run_name = f"{cfg.env.env_id}__{cfg.exp_name}__seed{cfg.seed}__{stamp}"
    run_dir = os.path.join(cfg.runs_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)

    for label, value in (
        ("env", cfg.env.env_id),
        ("device", device_name),
        ("seed", cfg.seed),
        ("steps", f"{cfg.num_iterations} iterations x {cfg.batch_size} = "
                  f"{cfg.num_iterations * cfg.batch_size} run, {cfg.env.total_timesteps} requested"),
        ("run directory", run_dir),
    ):
        print(f"{label:<14}{value}", flush=True)
    print(flush=True)

    git = write_record(run_dir, cfg, argv, device_name)
    if git["dirty"]:
        print(
            "WARNING: the working tree is dirty. The commit recorded in "
            f"{run_dir}/git.json does not fully describe this run. Commit first.",
            file=sys.stderr,
        )

    """
    seed_everything(cfg.seed)

    envs = SyncVectorEnv(
        [make_env(cfg.env.env_id, i) for i in range(cfg.ppo.num_envs)],
    """
    """ This is the previous seed_everything(cfg.seed) that is replaced by the following which ends with r_key=r_key, r_door=r_door) for i in range(cfg.ppo.num_envs)],
    """

    seed_everything(cfg.seed)

    # Intervention flags come from the env config group (conf/env/*.yaml). They
    # default to the baseline (partial view, sparse reward) when absent, so the
    # original env files keep working unchanged.
    fully_obs = bool(OmegaConf.select(cfg, "env.fully_obs", default=False))
    reward_shaping = bool(OmegaConf.select(cfg, "env.reward_shaping", default=False))
    r_key = float(OmegaConf.select(cfg, "env.r_key", default=0.2))
    r_door = float(OmegaConf.select(cfg, "env.r_door", default=0.3))
    print(f"{'intervention':<14}fully_obs={fully_obs} reward_shaping={reward_shaping} "
          f"(r_key={r_key}, r_door={r_door})", flush=True)

    envs = SyncVectorEnv(
        [make_env(cfg.env.env_id, i, fully_obs=fully_obs, reward_shaping=reward_shaping,
                  r_key=r_key, r_door=r_door) for i in range(cfg.ppo.num_envs)],

        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    assert isinstance(envs.single_action_space, spaces.Discrete)
    num_actions = int(envs.single_action_space.n)

    agent = Agent(num_actions).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.ppo.learning_rate, eps=1e-5)

    writer = SummaryWriter(os.path.join(run_dir, "tb"))
    writer.add_text(
        "config",
        "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in sorted(flatten(config_dict(cfg)).items())),
    )
    csv_path = os.path.join(run_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=METRIC_FIELDS)
    csv_writer.writeheader()

    ep_path = os.path.join(run_dir, "episodes.csv")
    ep_file = open(ep_path, "w", newline="")
    ep_writer = csv.DictWriter(ep_file, fieldnames=["global_step", "episodic_return", "episodic_length"])
    ep_writer.writeheader()

    # rollout storage. The image stays uint8 until the forward pass, which is
    # where it gets one-hotted; storing 20 float channels here would be 80x the
    # memory for nothing.
    # S, N = cfg.ppo.num_steps, cfg.ppo.num_envs
    # obs_image = torch.zeros((S, N, 7, 7, 3), dtype=torch.uint8, device=device)
    S, N = cfg.ppo.num_steps, cfg.ppo.num_envs
    # Image shape from the env, not hard-coded: (7, 7, 3) for the partial view
    # (baseline, Groups 1-2: identical to before), (N, N, 3) for the Group 3 full view.
    img_shape = tuple(envs.single_observation_space["image"].shape)
    obs_image = torch.zeros((S, N, *img_shape), dtype=torch.uint8, device=device)
    obs_dir = torch.zeros((S, N), dtype=torch.long, device=device)
    actions = torch.zeros((S, N), dtype=torch.long, device=device)
    logprobs = torch.zeros((S, N), device=device)
    rewards = torch.zeros((S, N), device=device)
    dones = torch.zeros((S, N), device=device)
    values = torch.zeros((S, N), device=device)

    global_step = 0
    start_time = time.time()
    raw_obs, _ = envs.reset(seed=[cfg.seed + i for i in range(cfg.ppo.num_envs)])
    next_image = torch.as_tensor(np.asarray(raw_obs["image"]), dtype=torch.uint8, device=device)
    next_dir = torch.as_tensor(np.asarray(raw_obs["direction"]), dtype=torch.long, device=device)
    next_done = torch.zeros(N, device=device)

    # one row per iteration, every field labelled so a single pasted line reads
    # on its own: return is the mean over the episodes that ended in the
    # iteration (n/a when none did), eps their count, ent the policy entropy
    for iteration in range(1, cfg.num_iterations + 1):
        if cfg.ppo.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
            lr_now = frac * cfg.ppo.learning_rate
            optimizer.param_groups[0]["lr"] = lr_now
        else:
            lr_now = cfg.ppo.learning_rate

        iter_returns: list[float] = []
        iter_lengths: list[float] = []

        for step in range(S):
            global_step += N
            obs_image[step] = next_image
            obs_dir[step] = next_dir
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_image, next_dir)
            values[step] = value
            actions[step] = action
            logprobs[step] = logprob

            raw_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            reward = np.asarray(reward, dtype=np.float32)

            # Truncation is not termination. A MiniGrid episode cut off at
            # max_steps still has value ahead of it, so bootstrap that value
            # into the reward and then treat the step as done for GAE. Without
            # this the critic learns that the horizon is worth zero.
            trunc_only = np.logical_and(truncations, np.logical_not(terminations))
            if trunc_only.any():
                idx = np.nonzero(trunc_only)[0]
                final_obs = infos["final_obs"]
                f_img = torch.as_tensor(
                    np.stack([final_obs[i]["image"] for i in idx]), dtype=torch.uint8, device=device
                )
                f_dir = torch.as_tensor(
                    np.array([final_obs[i]["direction"] for i in idx]), dtype=torch.long, device=device
                )
                with torch.no_grad():
                    bootstrap = agent.get_value(f_img, f_dir).cpu().numpy()
                reward[idx] += cfg.ppo.gamma * bootstrap

            rewards[step] = torch.as_tensor(reward, device=device)
            done = np.logical_or(terminations, truncations)
            next_done = torch.as_tensor(done.astype(np.float32), device=device)
            next_image = torch.as_tensor(np.asarray(raw_obs["image"]), dtype=torch.uint8, device=device)
            next_dir = torch.as_tensor(np.asarray(raw_obs["direction"]), dtype=torch.long, device=device)

            final_info = infos.get("final_info")
            if final_info and "episode" in final_info:
                mask = np.asarray(final_info["_episode"], dtype=bool)
                if mask.any():
                    ep_r = np.asarray(final_info["episode"]["r"], dtype=np.float64)[mask]
                    ep_l = np.asarray(final_info["episode"]["l"], dtype=np.float64)[mask]
                    for r_, l_ in zip(ep_r, ep_l):
                        iter_returns.append(float(r_))
                        iter_lengths.append(float(l_))
                        writer.add_scalar("charts/episodic_return", float(r_), global_step)
                        writer.add_scalar("charts/episodic_length", float(l_), global_step)
                        ep_writer.writerow(
                            {
                                "global_step": global_step,
                                "episodic_return": f"{float(r_):.6f}",
                                "episodic_length": int(l_),
                            }
                        )

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_image, next_dir)
            advantages = torch.zeros_like(rewards)
            lastgaelam = torch.zeros(N, device=device)
            for t in reversed(range(S)):
                if t == S - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg.ppo.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + cfg.ppo.gamma * cfg.ppo.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values

        # b_image = obs_image.reshape(-1, 7, 7, 3)
        b_image = obs_image.reshape(-1, *img_shape)
        b_dir = obs_dir.reshape(-1)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(cfg.batch_size)
        clipfracs = []
        approx_kl = torch.tensor(0.0)
        pg_loss = v_loss = entropy_loss = torch.tensor(0.0)
        for epoch in range(cfg.ppo.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                mb_inds = b_inds[start : start + cfg.minibatch_size]
                mb = torch.as_tensor(mb_inds, dtype=torch.long, device=device)

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_image[mb], b_dir[mb], b_actions[mb]
                )
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > cfg.ppo.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb]
                if cfg.ppo.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.ppo.clip_coef, 1 + cfg.ppo.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                if cfg.ppo.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb]) ** 2
                    v_clipped = b_values[mb] + torch.clamp(
                        newvalue - b_values[mb], -cfg.ppo.clip_coef, cfg.ppo.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ppo.ent_coef * entropy_loss + cfg.ppo.vf_coef * v_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.ppo.max_grad_norm)
                optimizer.step()

            if cfg.ppo.target_kl is not None and approx_kl.item() > cfg.ppo.target_kl:
                break

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = float(np.var(y_true))
        explained_var = float("nan") if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)

        elapsed = time.time() - start_time
        sps = global_step / max(elapsed, 1e-9)
        row = {
            "iteration": iteration,
            "global_step": global_step,
            "wall_time_s": round(elapsed, 3),
            "sps": round(sps, 2),
            "learning_rate": lr_now,
            "episodes_this_iter": len(iter_returns),
            "episodic_return_mean": float(np.mean(iter_returns)) if iter_returns else "",
            "episodic_length_mean": float(np.mean(iter_lengths)) if iter_lengths else "",
            "policy_loss": float(pg_loss.item()),
            "value_loss": float(v_loss.item()),
            "entropy": float(entropy_loss.item()),
            "approx_kl": float(approx_kl.item()),
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else "",
            "explained_variance": explained_var,
        }
        csv_writer.writerow(row)
        csv_file.flush()
        ep_file.flush()

        writer.add_scalar("charts/learning_rate", lr_now, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("losses/policy_loss", row["policy_loss"], global_step)
        writer.add_scalar("losses/value_loss", row["value_loss"], global_step)
        writer.add_scalar("losses/entropy", row["entropy"], global_step)
        writer.add_scalar("losses/approx_kl", row["approx_kl"], global_step)
        if clipfracs:
            writer.add_scalar("losses/clipfrac", row["clipfrac"], global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        if iter_returns:
            writer.add_scalar("charts/episodic_return_mean", row["episodic_return_mean"], global_step)
            writer.add_scalar("charts/episodic_length_mean", row["episodic_length_mean"], global_step)

        ret_str = f"{row['episodic_return_mean']:.3f}" if iter_returns else "  n/a"
        print(
            f"iter {iteration:4d}/{cfg.num_iterations}  step {global_step:8d}  "
            f"return {ret_str}  eps {len(iter_returns):3d}  "
            f"ent {row['entropy']:.3f}  sps {sps:7.0f}",
            flush=True,
        )

    envs.close()
    csv_file.close()
    ep_file.close()
    writer.close()

    if cfg.save_model:
        torch.save(
            {"model_state_dict": agent.state_dict(), "config": config_dict(cfg), "global_step": global_step},
            os.path.join(run_dir, "model.pt"),
        )

    with open(os.path.join(run_dir, "env_info.json")) as f:
        info = json.load(f)
    info["finished_utc"] = datetime.now(timezone.utc).isoformat()
    info["wall_time_s"] = round(time.time() - start_time, 3)
    info["final_sps"] = round(global_step / max(time.time() - start_time, 1e-9), 2)
    with open(os.path.join(run_dir, "env_info.json"), "w") as f:
        json.dump(info, f, indent=2, sort_keys=True)

    print(f"\nrun directory: {run_dir}")
    return run_dir


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # sys.argv[1:] holds exactly the overrides the user typed; command.txt keeps them.
    train(finalize(cfg), sys.argv[1:])


if __name__ == "__main__":
    main()
