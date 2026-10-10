"""Independent sanity reference: Stable-Baselines3 PPO on the same task.

This is NOT the deliverable. It exists to answer one question when the
single-file PPO in ppo_minigrid.py is not learning: is the bug in the
algorithm, or in the environment plumbing? If SB3 learns here and
ppo_minigrid.py does not, the bug is ours.

    python sb3_sanity.py env_id=MiniGrid-Empty-5x5-v0 total_timesteps=100000

The options are declared in conf/sb3.yaml.
"""

import json
import os
import sys
import time

import gymnasium as gym
import hydra
from omegaconf import DictConfig
import minigrid  # noqa: F401
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


def make(env_id):
    # MiniGrid's own ImgObsWrapper: keep the image, drop direction and mission.
    # The observation is then Box(0, 255, (7,7,3), uint8), which SB3 classifies
    # as an image, so with an MlpPolicy it divides by 255 and flattens. The
    # symbolic object ids therefore reach the network as ordinal floats, which
    # is exactly the encoding ppo_minigrid.py argues against.
    #
    # That is deliberate and it is the point. A reference that shared our
    # observation treatment would not be independent of it: if both failed we
    # would learn nothing about whether the encoding was the problem.
    return ImgObsWrapper(gym.make(env_id))


@hydra.main(version_base=None, config_path="conf", config_name="sb3")
def main(a: DictConfig):
    venv = make_vec_env(lambda: make(a.env_id), n_envs=a.n_envs, seed=a.seed)
    model = PPO(
        "MlpPolicy",
        venv,
        seed=a.seed,
        device=a.device,
        verbose=1,
        n_steps=128,
        batch_size=256,
        ent_coef=0.01,
        learning_rate=2.5e-4,
    )

    eval_env = make_vec_env(lambda: make(a.env_id), n_envs=4, seed=a.seed + 1000)
    before, before_std = evaluate_policy(model, eval_env, n_eval_episodes=40)

    t0 = time.time()
    model.learn(total_timesteps=a.total_timesteps, progress_bar=False)
    dt = time.time() - t0

    after, after_std = evaluate_policy(model, eval_env, n_eval_episodes=40)
    print(f"[sb3] {a.env_id}, {a.total_timesteps} steps in {dt:.1f} s, {a.total_timesteps / dt:.0f} steps/s")
    print(f"[sb3] {'':<17}return")
    print(f"[sb3] {'before training':<17}{before:5.3f} +/- {before_std:.3f}")
    print(f"[sb3] {'after training':<17}{after:5.3f} +/- {after_std:.3f}")

    # Persist it. A reference number that lives only in scrollback cannot be
    # checked against later, which defeats the purpose of having a reference.
    out = a.out or os.path.join("runs", f"sb3_{a.env_id}_seed{a.seed}", "sb3_sanity.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "env_id": a.env_id, "seed": a.seed, "n_envs": a.n_envs, "device": a.device,
            "total_timesteps": a.total_timesteps,
            "command": " ".join([sys.executable, os.path.abspath(__file__)] + sys.argv[1:]),
            "return_before": float(before), "return_before_std": float(before_std),
            "return_after": float(after), "return_after_std": float(after_std),
            "wall_time_s": round(dt, 3), "steps_per_second": round(a.total_timesteps / dt, 2),
            "eval_episodes": 40,
            "note": "SB3 reference only. Not the deliverable. Different observation treatment "
                    "from ppo_minigrid.py on purpose, see the comment in make().",
        }, f, indent=2, sort_keys=True)
    print(f"[sb3] wrote {out}")


if __name__ == "__main__":
    main()
