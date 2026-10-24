"""Sanity tests for the two midterm interventions, on the real MiniGrid.

    python tests/test_interventions.py        # prints PASS lines, exits 1 on failure

Runs in seconds, needs only the cap6671 environment. It checks the rules the
proposal commits to, before any compute is spent on the 8x8 comparison:

  Group 2 (MilestoneReward)
    * no bonus on an ordinary step
    * +0.2 the first step the key is held, never again (drop / re-pick is not paid)
    * +0.3 for the open door only AFTER the key bonus, never again
    * an open door without the key bonus pays nothing (order is enforced)
    * reset() re-arms both milestones
    * the logged episode return (RecordEpisodeStatistics) is the SPARSE return
  Group 3 (FullyObsWrapper + FullObsStateFix)
    * image is the full N x N grid
    * the state channel stays in {0,1,2} for all four facing directions
      (without the fix, facing up writes 3 there and the one-hot crashes)
    * the unchanged Agent accepts the full-grid input and outputs 7 logits
  # Baseline
  #  * on the 7x7 view the conv stack already ends at 1x1, so the added
  #    AdaptiveAvgPool2d is the identity
  Baseline
    * on the 7x7 view the conv stack already ends at 1x1, so the added
      AdaptiveAvgPool2d is the identity
  End to end (about 10-20 s)
    * one real PPO iteration per group trains and saves a model
    * the full-view model evaluates
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import minigrid  # noqa: F401  (registers the envs)
import numpy as np
import torch
from minigrid.core.world_object import Door, Key
from minigrid.wrappers import FullyObsWrapper

from ppo_minigrid import Agent, NUM_STATE, encode_obs, make_env
from wrappers import MilestoneReward

DONE = 6  # MiniGrid "done" action: a no-op in DoorKey
FAILED = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILED.append(name)


def find_door(env):
    return next(o for o in env.unwrapped.grid.grid if isinstance(o, Door))


def test_milestones():
    env = make_env("MiniGrid-DoorKey-8x8-v0", 0, record_stats=False, reward_shaping=True)()
    env.reset(seed=0)
    u = env.unwrapped

    _, r, *_ = env.step(DONE)
    check("G2 no bonus on an ordinary step", r == 0.0)

    door = find_door(env)
    door.is_locked, door.is_open = False, True
    _, r, *_ = env.step(DONE)
    check("G2 open door BEFORE key pays nothing (order enforced)", r == 0.0)
    door.is_locked, door.is_open = True, False

    u.carrying = Key("yellow")
    _, r, *_ = env.step(DONE)
    check("G2 key bonus +0.2 on first pick-up", abs(r - 0.2) < 1e-9)
    _, r, *_ = env.step(DONE)
    check("G2 key bonus not repeated while holding", r == 0.0)
    u.carrying = None
    env.step(DONE)
    u.carrying = Key("yellow")
    _, r, *_ = env.step(DONE)
    check("G2 drop and re-pick is not paid again (anti-hacking)", r == 0.0)

    door.is_locked, door.is_open = False, True
    _, r, *_ = env.step(DONE)
    check("G2 door bonus +0.3 after key", abs(r - 0.3) < 1e-9)
    door.is_open = False
    env.step(DONE)
    door.is_open = True
    _, r, *_ = env.step(DONE)
    check("G2 door bonus not repeated (close / reopen)", r == 0.0)

    env.reset(seed=1)
    env.unwrapped.carrying = Key("yellow")
    _, r, *_ = env.step(DONE)
    check("G2 reset re-arms the milestones", abs(r - 0.2) < 1e-9)
    env.close()


def test_logged_return_is_sparse():
    env = make_env("MiniGrid-DoorKey-5x5-v0", 0, record_stats=True, reward_shaping=True)()
    env.reset(seed=0)
    env.unwrapped.carrying = Key("yellow")
    shaped_total, info, done = 0.0, {}, False
    while not done:
        _, r, term, trunc, info = env.step(DONE)
        shaped_total += float(r)
        done = term or trunc
    logged = float(info["episode"]["r"])
    check("G2 PPO sees the bonus (shaped total = 0.2)", abs(shaped_total - 0.2) < 1e-9)
    check("G2 logged episode return is sparse (0.0, no bonus)", logged == 0.0)
    env.close()


def test_full_observation():
    # The bug the fix addresses: raw FullyObsWrapper puts direction in channel 2.
    raw = FullyObsWrapper(gym.make("MiniGrid-DoorKey-8x8-v0"))
    obs, _ = raw.reset(seed=0)
    raw_max = []
    for _ in range(4):
        obs, *_ = raw.step(0)  # turn left: visits all four directions
        raw_max.append(int(obs["image"][..., 2].max()))
    check("G3 (bug reproduced) raw FullyObsWrapper writes 3 into the state channel",
          max(raw_max) >= NUM_STATE)
    raw.close()

    env = make_env("MiniGrid-DoorKey-8x8-v0", 0, record_stats=False, fully_obs=True)()
    obs, _ = env.reset(seed=0)
    check("G3 image is the full 8x8 grid", obs["image"].shape == (8, 8, 3))
    agent = Agent(int(env.action_space.n))
    ok_states, ok_forward = True, True
    for _ in range(4):
        obs, *_ = env.step(0)
        ok_states &= int(obs["image"][..., 2].max()) < NUM_STATE
        img = torch.as_tensor(obs["image"][None], dtype=torch.uint8)
        d = torch.as_tensor([obs["direction"]], dtype=torch.long)
        try:
            encode_obs(img, d)
            with torch.no_grad():
                logits = agent.actor(agent.features(img, d))
            ok_forward &= tuple(logits.shape) == (1, 7)
        except Exception as e:  # noqa: BLE001
            print("   ", type(e).__name__, e)
            ok_forward = False
    check("G3 state channel in {0,1,2} for all four directions", ok_states)
    check("G3 unchanged Agent runs on the full grid, 7 logits", ok_forward)
    env.close()


def test_pool_is_identity_on_7x7():
    agent = Agent(7)
    img = torch.zeros(1, 7, 7, 3, dtype=torch.uint8)
    d = torch.zeros(1, dtype=torch.long)
    x, _ = encode_obs(img, d)
    convs = agent.encoder[:-2]  # everything before AdaptiveAvgPool2d and Flatten
    with torch.no_grad():
        spatial = convs(x).shape[-2:]
    check("baseline conv stack ends at 1x1 on 7x7 (pool is the identity)", tuple(spatial) == (1, 1))

def test_one_iteration_each_group():
    """End-to-end: one real PPO iteration (1,024 steps) per group, then a short
    evaluation of the full-view model. Catches shape bugs in the training loop
    (rollout buffer) and in evaluation that the unit checks above cannot."""
    import subprocess
    import tempfile
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory() as tmp:
        for env in ("doorkey8x8", "doorkey8x8_shaped", "doorkey8x8_fullobs"):
            cmd = [sys.executable, "ppo_minigrid.py", f"env={env}", "env.total_timesteps=1024",
                   "seed=1", f"runs_dir={tmp}", f"run_name=t_{env}"]
            r = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
            ok = r.returncode == 0 and os.path.exists(os.path.join(tmp, f"t_{env}", "model.pt"))
            if not ok:
                print(r.stderr[-1500:])
            check(f"end-to-end: one training iteration on {env}", ok)
        model = os.path.join(tmp, "t_doorkey8x8_fullobs", "model.pt")
        r = subprocess.run([sys.executable, "eval_checkpoint.py", f"model_path={model}", "episodes=2"],
                           cwd=root, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-1500:])
        check("end-to-end: evaluate the full-view model", r.returncode == 0)


if __name__ == "__main__":
    test_milestones()
    test_logged_return_is_sparse()
    test_full_observation()
    # test_pool_is_identity_on_7x7()
    # print()
    test_pool_is_identity_on_7x7()
    test_one_iteration_each_group()
    print()
    print("ALL PASSED" if not FAILED else f"{len(FAILED)} FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)
