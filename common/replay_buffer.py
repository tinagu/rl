import torch
from numpy import ndarray
import numpy as np
from .utils import to_tensor


class EnvReplayBuffer(object):

    def __init__(self, observation_shape, action_shape, max_size):
        self.max_size = max_size

        self.obs_buf = np.empty((self.max_size, *observation_shape), dtype=np.float32)
        self.ac_buf  = np.empty((self.max_size, *action_shape), dtype=np.float32)
        self.rew_buf = np.empty(self.max_size, dtype=np.float32)

    def store(self, obs: ndarray, action: ndarray, rewards: ndarray,
              ptr: int) -> None:
        self.obs_buf[ptr] = obs
        self.ac_buf[ptr] = action
        self.rew_buf[ptr] = rewards

    def sample_batch(self, idxs: ndarray):
        batch = {
            "obs": to_tensor(self.obs_buf[idxs], dtype=torch.float32),
            "actions": to_tensor(self.ac_buf[idxs]),
            "rewards": to_tensor(self.rew_buf[idxs], dtype=torch.float32),
        }
        return batch


class OffPolicyReplayBuffer:

    def __init__(self, observation_shape, action_shape, max_size=int(1e6)):
        self.env_replay_buffer = EnvReplayBuffer(observation_shape, action_shape, max_size)

        self.ptr = 0
        self.sz = 0
        self.max_size = max_size

        self.nxt_obs_buf = np.empty((self.max_size, *observation_shape), dtype=np.float32)
        self.done_buf    = np.empty(self.max_size, dtype=np.int32)

    def store(self, obs: ndarray, action: ndarray, rewards: ndarray, nxt_obs: ndarray, done: ndarray):
        self.env_replay_buffer.store(
            obs, action, rewards, self.ptr
        )

        self.nxt_obs_buf[self.ptr] = nxt_obs
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.sz = min(self.sz+1, self.max_size)

    def sample_batch(self, batch_size=32):
        assert self.sz >= batch_size, f'sample size: {self.sz}, batch size: {batch_size}'
        idxs = np.random.randint(0, self.sz, batch_size)

        batch = self.env_replay_buffer.sample_batch(idxs)
        batch["nxt_obs"] = to_tensor(self.nxt_obs_buf[idxs])
        batch["dones"] = to_tensor(self.done_buf[idxs])
        return batch


class OnPolicyReplayBuffer:

    def __init__(self, observation_shape, action_shape, max_size,
                 gamma=0.99, tau=0.95):
        self.env_replay_buffer = EnvReplayBuffer(observation_shape, action_shape, max_size)

        self.max_size = max_size
        self.gamma = gamma
        self.tau = tau
        self.ptr = 0
        self.path_start_idx = 0

        self.logp_buf    = np.empty(self.max_size, dtype=np.float32)
        self.val_buf     = np.empty(self.max_size, dtype=np.float32)
        self.adv_buf     = np.empty(self.max_size, dtype=np.float32)
        self.ret_buf     = np.empty(self.max_size, dtype=np.float32)
        self.ptr = 0
        self.path_start_idx = 0

    def store(self, ob: ndarray, action: ndarray, val: np.float32,
              logp: np.float32, reward: np.float32) -> None:
        self.env_replay_buffer.store(
            ob, action, reward, self.ptr
        )
        self.logp_buf[self.ptr] = logp
        self.val_buf[self.ptr] = val
        self.ret_buf[self.ptr] = reward

        self.ptr = self.ptr + 1
        if self.ptr == self.max_size:
            print("warning: buffer is full")

    def end_episode(self, last_val=0):
        prev_adv = 0
        prev_ret = last_val
        prev_val = last_val
        for i in range(self.ptr - 1, self.path_start_idx - 1, -1):
            delta_t = self.env_replay_buffer.rew_buf[i] + self.gamma * prev_val - self.val_buf[i]
            prev_val = self.val_buf[i]

            self.adv_buf[i] = prev_adv * self.gamma * self.tau + delta_t
            prev_adv = self.adv_buf[i]

            self.ret_buf[i] += prev_ret * self.gamma
            prev_ret = self.ret_buf[i]
        self.path_start_idx = self.ptr

    def sample_batch(self, batch_size):
        assert batch_size == self.max_size, f'sample size: {batch_size}, batch_size: {self.max_size}'
        assert self.ptr == self.max_size, f'available sample size: {self.ptr}, batch_size: {self.max_size}'
        self.ptr = 0
        self.path_start_idx = 0

        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        batch = self.env_replay_buffer.sample_batch(np.arange(self.max_size))
        batch["logp"] = to_tensor(self.logp_buf)
        batch["returns"] = to_tensor(self.ret_buf)
        batch["adv"] = to_tensor(self.adv_buf)

        return batch


