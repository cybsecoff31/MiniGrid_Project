"""Evaluate a saved model.pt from a run directory.

    python eval_checkpoint.py model_path=runs/<run>/model.pt policy.also_random=true
    python eval_checkpoint.py model_path=runs/<run>/model.pt render.enabled=true

Every option is declared and explained in conf/eval.yaml. The defaults are the
course evaluation protocol (200 episodes, seed 12345, sampled actions), so the
bare command already conforms and only a deliberate departure needs an override.

Writes the result next to the checkpoint as eval.json, so the "does it learn"
numbers in a report have the same audit trail as the training curve. A number
that exists only in terminal scrollback is not evidence. With render.enabled the
chosen episodes are also saved as GIFs under runs/<run>/render/<policy>/.
"""

import json
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from ppo_minigrid import Agent, make_env
import minigrid  # noqa: F401

POLICY_MODES = ("sampled", "argmax")


# ---------------------------------------------------------------------------
# Playing episodes
# ---------------------------------------------------------------------------

def play(env, act, episodes, seed, renderer=None):
    """Play `episodes` episodes, episode i on layout seed + i.

    `act(obs) -> int` is the policy. Returns the per-episode returns and lengths.
    The renderer, if any, only reads frames; it never touches the environment
    or the policy, so the numbers are the same with and without it.
    """
    returns, lengths = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        rec = renderer.start(ep) if renderer else None
        done = False
        total, steps = 0.0, 0
        while not done:
            obs, r, term, trunc, _ = env.step(act(obs))
            total += float(r)
            steps += 1
            done = term or trunc
            if rec is not None:
                rec.append(renderer.frame())
        if rec is not None:
            renderer.save(ep, rec)
        returns.append(total)
        lengths.append(steps)
    return np.array(returns), np.array(lengths)


def trained_policy(agent, mode, device):
    def act(obs):
        img = torch.as_tensor(obs["image"][None], dtype=torch.uint8, device=device)
        d = torch.as_tensor([obs["direction"]], dtype=torch.long, device=device)
        with torch.no_grad():
            if mode == "argmax":
                logits = agent.actor(agent.features(img, d))
                return int(logits.argmax(-1).item())
            a, _, _, _ = agent.get_action_and_value(img, d)
            return int(a.item())
    return act


def random_policy(n_actions, seed):
    rng = np.random.default_rng(seed)
    return lambda obs: int(rng.integers(n_actions))


def evaluate(model_path, episodes=200, seed=12345, device="cpu", mode="sampled", env_id=None, renderer=None):
    # torch has to be seeded too, or the stochastic path samples differently on
    # every invocation and `seed` only controls the layouts.
    torch.manual_seed(seed)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    env_id = env_id or checkpoint_env_id(ckpt)
    """
    env = make_env(env_id, 0, record_stats=False)()
    This section is commented and replaced with its following equivalent
    """
    env = make_env(env_id, 0, record_stats=False, fully_obs=checkpoint_fully_obs(ckpt))()
    agent = Agent
    agent = Agent(int(env.action_space.n)).to(device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()
    if renderer:
        renderer.attach(env)
    out = play(env, trained_policy(agent, mode, device), episodes, seed, renderer)
    env.close()
    return out

"""    
def random_baseline(env_id, episodes=200, seed=12345, renderer=None):
    env = make_env(env_id, 0, record_stats=False)()
Another comment from the previous version
"""
def random_baseline(env_id, episodes=200, seed=12345, renderer=None):
    # The random policy ignores observations, so the observation wrapper cannot
    # change its actions or the layouts; the bare env is used as before.
    env = make_env(env_id, 0, record_stats=False)()
    if renderer:
        renderer.attach(env)
    out = play(env, random_policy(env.action_space.n, seed), episodes, seed, renderer)
    env.close()
    return out


def checkpoint_env_id(ckpt) -> str:
    """model.pt stores the resolved training config; the task id is under `env`.

    Checkpoints written before the config moved to conf/ kept it at the top level.
    """
    c = ckpt["config"]
    return c["env"]["env_id"] if "env" in c else c["env_id"]


def checkpoint_fully_obs(ckpt) -> bool:
    """Group 3 models were trained on the full-grid observation, so they must be
    evaluated on it too (a 7x7 patch is a different input). Only the OBSERVATION
    is rebuilt from the checkpoint: reward shaping is never applied here, so
    every group is scored on the environment's sparse objective. Checkpoints
    without the flag (all baselines) evaluate exactly as before."""
    env_cfg = ckpt["config"].get("env", {})
    return bool(env_cfg.get("fully_obs", False))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class Renderer:
    """Saves the chosen episodes as GIFs, one per episode, in `out_dir`.

    Frames come from MiniGrid's own full-grid drawing (`get_frame`), with the
    agent's 7x7 view shaded when `highlight` is set. Which episodes are drawn
    is decided by index alone: the first `first`, or exactly `only`.
    """

    def __init__(self, out_dir, first=5, only=None, highlight=True, tile_size=32, frame_ms=200):
        self.out_dir = out_dir
        self.only = None if only is None else set(int(i) for i in only)
        self.first = int(first)
        self.highlight = bool(highlight)
        self.tile_size = int(tile_size)
        self.frame_ms = int(frame_ms)
        self.env = None
        self.written = []

    def attach(self, env):
        self.env = env.unwrapped

    def wants(self, ep):
        return ep in self.only if self.only is not None else ep < self.first

    def start(self, ep):
        if not self.wants(ep):
            return None
        return [self.frame()]

    def frame(self):
        return self.env.get_frame(highlight=self.highlight, tile_size=self.tile_size)

    def save(self, ep, frames):
        from PIL import Image
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"ep{ep:03d}.gif")
        ims = [Image.fromarray(f) for f in frames]
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=self.frame_ms, loop=0, optimize=True)
        self.written.append(path)


def make_renderer(a, run_dir, policy_name):
    if not a.render.enabled:
        return None
    only = None if a.render.only is None else OmegaConf.to_container(a.render.only)
    return Renderer(os.path.join(run_dir, "render", policy_name), a.render.first, only,
                    a.render.highlight, a.render.tile_size, a.render.frame_ms)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def summary(r, l):
    return {
        "return_mean": float(r.mean()), "return_std": float(r.std()),
        "length_mean": float(l.mean()), "success_rate": float((r > 0).mean()),
    }


def report(rows, episodes, seed):
    """One aligned row per policy, all over the same layouts."""
    print(f"{episodes} episodes, evaluation seed {seed}")
    print(f"{'policy':<18}  {'return':<15}  {'length':>6}  {'success':>7}")
    for name, r, l in rows:
        print(f"{name:<18}  {r.mean():5.3f} +/- {r.std():.3f}  {l.mean():6.1f}  {(r > 0).mean():7.2%}")


@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(a: DictConfig):
    if OmegaConf.is_missing(a, "model_path"):
        raise SystemExit("model_path is required:  python eval_checkpoint.py model_path=runs/<run>/model.pt")
    if a.policy.mode not in POLICY_MODES:
        raise SystemExit(f"policy.mode={a.policy.mode!r} is not one of {POLICY_MODES}")
    ckpt = torch.load(a.model_path, map_location="cpu", weights_only=False)
    env_id = a.env_id or checkpoint_env_id(ckpt)
    run_dir = os.path.dirname(os.path.abspath(a.model_path))

    renderer = make_renderer(a, run_dir, a.policy.mode)
    r, l = evaluate(a.model_path, a.episodes, a.seed, a.device, a.policy.mode, a.env_id, renderer)
    rows = [(f"trained ({a.policy.mode})", r, l)]
    written = list(renderer.written) if renderer else []

    record = {
        "model_path": os.path.abspath(a.model_path),
        "env_id": env_id,
        "episodes": a.episodes,
        "eval_seed": a.seed,
        "device": a.device,
        "policy_mode": a.policy.mode,
        "fully_obs": checkpoint_fully_obs(ckpt),
        "reward_shaping_at_eval": False,
        "command": " ".join([sys.executable, os.path.abspath(__file__)] + sys.argv[1:]),
        "trained": summary(r, l),
    }

    if a.policy.also_random:
        renderer = make_renderer(a, run_dir, "random")
        rr, ll = random_baseline(env_id, a.episodes, a.seed, renderer)
        rows.append(("random", rr, ll))
        record["random"] = summary(rr, ll)
        written += renderer.written if renderer else []
    report(rows, a.episodes, a.seed)

    # The protocol result is eval.json. The greedy variant gets its own file so
    # that looking at it never overwrites the number a report cites.
    out = os.path.join(run_dir, "eval.json" if a.policy.mode == "sampled" else "eval-argmax.json")
    with open(out, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    print(f"wrote {out}")
    if written:
        print(f"wrote {len(written)} GIFs under {os.path.join(run_dir, 'render')}")


if __name__ == "__main__":
    main()
