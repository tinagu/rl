import numpy as np


from pg_agent import MLPPolicy
import gym
from typing import Dict


def sample_trajectory(
        env: gym.Env, policy: MLPPolicy, max_length: int,
) -> Dict[str, np.ndarray]:
    ob, _ = env.reset()
    obs, acs, rewards, next_obs, terminals = [], [], [], [], []
    steps = 0

    while True:
        ac: np.ndarray = policy.get_action(ob)
        next_ob, rew, done, _, _ = env.step(ac)

        steps += 1
        rollout_done: bool = done or steps > max_length

        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob

        if rollout_done:
            break

    return {
        "observation": np.array(obs, dtype=np.float32),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminals": np.array(terminals, dtype=np.float32)
    }