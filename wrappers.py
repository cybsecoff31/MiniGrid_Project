"""
Intervention wrappers for the CAP6671 DoorKey project (midterm comparison).

These implement the three-group diagnostic framework from the proposal without
touching the PPO algorithm itself:

    Group 1  control baseline   partial 7x7 view  +  sparse reward   (no wrapper)
    Group 2  shaped variant     partial 7x7 view  +  milestone reward (MilestoneReward)
    Group 3  global ablation     full grid view    +  sparse reward   (FullyObsWrapper
                                                                       + FullObsStateFix)

Design rules the proposal commits to, enforced here:
  * The PPO code, hyperparameters and the native 7-action space are untouched.
  * Group 2 only changes the TRAINING reward. Evaluation must remove the wrapper:
    eval_checkpoint.py never applies MilestoneReward, so every group is scored on
    the environment's sparse objective. (For Group 3 it rebuilds only the
    full-grid OBSERVATION from the checkpoint's config.)
  * Episode statistics are logged BEFORE the bonus (see make_env in
    ppo_minigrid.py), so episodes.csv / metrics.csv hold the sparse return for
    every group and the learning curves stay comparable.
  * Milestone bonuses are paid at most once per episode and are ordered: the
    door bonus cannot fire before the key bonus. This blocks the reward-hacking
    loop (drop key / pick key repeatedly) the proposal calls out.

Everything here uses only the public MiniGrid API:
  * env.unwrapped.carrying        -> the object the agent holds (a Key or None)
  * env.unwrapped.grid.grid       -> the tiles; a Door has .is_open / .is_locked
"""

from __future__ import annotations

import gymnasium as gym
from minigrid.core.world_object import Key, Door


class MilestoneReward(gym.Wrapper):
    """Add shaped bonuses for the two sub-goals of DoorKey, on top of the
    environment's own sparse goal reward.

        R_key  = +0.2   paid the first step the agent is carrying the key
        R_door = +0.3   paid the first step the (previously locked) door is open,
                        and only after the key bonus has already been paid

    The environment's native reward (≈1 - 0.9*steps/max_steps on success, 0
    otherwise) is left untouched and still flows through. At evaluation time this
    wrapper is simply not applied, so the agent is scored on the real objective.
    """

    def __init__(self, env: gym.Env, r_key: float = 0.2, r_door: float = 0.3):
        super().__init__(env)
        self.r_key = float(r_key)
        self.r_door = float(r_door)
        self._key_paid = False
        self._door_paid = False

    def reset(self, **kwargs):
        self._key_paid = False
        self._door_paid = False
        return self.env.reset(**kwargs)

    # --- detectors (read-only; they never mutate the environment) -------------
    def _holding_key(self) -> bool:
        carrying = self.env.unwrapped.carrying
        return isinstance(carrying, Key)

    def _door_open(self) -> bool:
        grid = self.env.unwrapped.grid
        for obj in grid.grid:                 # flat list of tiles, None for empty
            if isinstance(obj, Door) and obj.is_open:
                return True
        return False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        bonus = 0.0

        # key milestone: once, the first step the key is in hand
        if not self._key_paid and self._holding_key():
            bonus += self.r_key
            self._key_paid = True

        # door milestone: once, after the key, the first step the door is open
        if self._key_paid and not self._door_paid and self._door_open():
            bonus += self.r_door
            self._door_paid = True

        if bonus:
            reward = float(reward) + bonus
            info = dict(info)
            info["shaping_bonus"] = bonus
        return obs, reward, terminated, truncated, info


class FullObsStateFix(gym.ObservationWrapper):
    """Make MiniGrid's FullyObsWrapper output compatible with the baseline encoder.

    FullyObsWrapper writes the agent into the grid as
        [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], agent_dir]
    i.e. it stores the facing direction (0..3) in channel 2, the STATE channel.
    The baseline encoder one-hot encodes channel 2 with NUM_STATE = 3 classes
    (open / closed / locked), so whenever agent_dir == 3 (facing up) the one-hot
    call raises "Class values must be smaller than num_classes" and training
    crashes. Directions 1-2 would not crash but would be silently mis-read as
    door states.

    The direction is already given to the network through the separate
    `direction` input, so the fix is simply to zero the state channel on the
    agent's own cell. Every other cell is untouched (doors keep their state).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        from minigrid.core.constants import OBJECT_TO_IDX
        self._agent_id = OBJECT_TO_IDX["agent"]

    def observation(self, obs):
        img = obs["image"].copy()
        agent_cells = img[..., 0] == self._agent_id
        img[..., 2][agent_cells] = 0
        obs = dict(obs)
        obs["image"] = img
        return obs
