import torch
import torch.nn as nn
import numpy as np
import gym
import os
from common.replay_buffer import OffPolicyReplayBuffer
from common.logger import logger
import common.utils as utils
from .agent import Agent


class DQNAgent(Agent):

    def __init__(self, configs):
        super().__init__(configs)

        self.dqn_config = self.configs["dqn"]
        obs_dim = self.env.observation_space.low.size
        action_dim = self.env.action_space.n
        hidden_layer_sizes = self.dqn_config["hidden_sizes"]

        self.q_net = utils.build_net(obs_dim, action_dim, hidden_layer_sizes, nn.ReLU)
        self.q_net_target = utils.build_net(obs_dim, action_dim, hidden_layer_sizes, nn.ReLU)
        self.q_net_target.load_state_dict(self.q_net.state_dict())

        self.q_loss_func = nn.MSELoss()

        self._replay_buffer = OffPolicyReplayBuffer(self.env.observation_space.shape, self.env.action_space.shape)

        self.learning_rate = self.dqn_config["learning_rate"]
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=self.learning_rate)

        self.epsilon = self.dqn_config["epsilon"]
        self.polyak = self.dqn_config["polyak"]
        self.gamma = self.dqn_config["gamma"]
        self.target_update_period = self.dqn_config["target_update_period"]

        self.register_networks(self.q_net)
        self.load_models()

    def load_models(self):
        if self.model_saving_path:
            fname = f"{self.model_saving_path}/q_net"
            if os.path.exists(fname):
                self.q_net.load_state_dict(torch.load(fname))
            fname = f"{self.model_saving_path}/q_net_target"
            if os.path.exists(fname):
                self.q_net_target.load_state_dict(torch.load(fname))

    def save_models(self):
        if self.model_saving_path:
            torch.save(self.q_net.state_dict(), f"{self.model_saving_path}/q_net")
            torch.save(self.q_net_target.state_dict(), f"{self.model_saving_path}/q_net_target")

    def get_explore_action(self, ob):
        epsilon = self.epsilon
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            return self.get_exploit_action(ob)

    def get_exploit_action(self, ob):
        observation = torch.tensor(ob)
        with torch.no_grad():
            logits = self.q_net(observation)
            action = torch.argmax(logits, dim=-1)
            return action.item()

    @property
    def replay_buffer(self):
        return self._replay_buffer

    def collect_data(self, max_ep_len, num_samples, get_action_func):
        ob = self.env.reset()
        count = 0
        ep_rewards = []
        while count < num_samples:
            ep_len = 0
            done = False
            ep_reward = 0
            while not done and ep_len < min(max_ep_len, num_samples-count):
                action = get_action_func(ob)
                nxt_ob, reward, done, _ = self.env.step(action)

                ep_rewards.append(reward)
                ep_len += 1
                ep_reward += 1

                self.replay_buffer.store(ob, action, reward, nxt_ob, done)
                ob = nxt_ob
            if done:
                ob = self.env.reset()
            count += ep_len
            logger.record("training/ep_reward", ep_reward)
            logger.record("training/ep_len", ep_len)

        return {
            "avg_ep_reward": np.mean(ep_rewards),
            "min_ep_reward": np.min(ep_rewards),
            "max_ep_reward": np.max(ep_rewards)
        }

    def train(self, batch, step: int) -> dict:
        obs = torch.tensor(batch["obs"])
        next_obs = torch.tensor(batch["nxt_obs"])
        actions = torch.tensor(batch["actions"], dtype=torch.int64)
        rewards = torch.tensor(batch["rewards"])
        dones = torch.tensor(batch["dones"])

        with torch.no_grad():
            nxt_qa_vals = self.q_net_target(next_obs)
            nxt_q_vals, _ = nxt_qa_vals.max(dim=-1)
            target_vals = rewards + self.gamma * nxt_q_vals * (1 - dones)

        qa_vals = self.q_net(obs)
        q_vals = torch.gather(qa_vals, 1, actions.unsqueeze(1).long()).squeeze(1)

        qf_loss = self.q_loss_func(q_vals, target_vals)

        self.optimizer.zero_grad()
        qf_loss.backward()
        self.optimizer.step()

        if step % self.target_update_period == 0:
            with torch.no_grad():
                for p, p_targ in zip(self.q_net.parameters(), self.q_net_target.parameters()):
                    p_targ.data.copy_(p_targ.data * self.polyak + (1-self.polyak) * p.data)

        return {
            "loss": qf_loss.item(),
            "q_values": q_vals.mean().item(),
            "q_target_values": target_vals.mean().item()
        }
