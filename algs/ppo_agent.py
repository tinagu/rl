import torch
import torch.nn as nn
import gym
import os
from common import utils
from common.replay_buffer import OnPolicyReplayBuffer
from common.logger import logger
from .agent import Agent


class Actor(nn.Module):

    def __init__(self, is_discrete, input_sz, output_sz, hidden_layers_sz,
                 activation=torch.nn.Tanh, output_activation=torch.nn.Identity):
        super(Actor, self).__init__()

        self.is_discrete = is_discrete
        if is_discrete:
            self.logits_net = utils.build_net(input_sz, output_sz, hidden_layers_sz,
                                              activation, output_activation)
        else:
            self.mean_net = utils.build_net(input_sz, output_sz, hidden_layers_sz,
                                            activation, output_activation)
            self.log_std = torch.nn.Parameter(
                -0.5 * torch.ones(output_sz, dtype=torch.float32)
            )

    def forward(self, obs, action=None):
        log_action = None
        if self.is_discrete:
            logits = self.logits_net(obs)
            pi = torch.distributions.Categorical(logits=logits)
            if action is not None:
                log_action = pi.log_prob(action)
        else:
            mean = self.mean_net(obs)
            std = torch.exp(self.log_std)
            pi = torch.distributions.Normal(mean, std)
            if action is not None:
                log_action = pi.log_prob(action).sum(dim=-1)

        return pi, log_action


class Critic(nn.Module):

    def __init__(self, obs_dim, hidden_sz, activation=nn.Tanh):
        super(Critic, self).__init__()

        self.net = utils.build_net(obs_dim, 1, hidden_sz, activation)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class PPOAgent(Agent):

    def __init__(self, configs):
        super(PPOAgent, self).__init__(configs)

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]

        self.ppo_configs = self.configs["ppo"]
        self.gamma = self.ppo_configs["gamma"]
        self.tau = self.ppo_configs["tau"]
        self.clip_ratio = self.ppo_configs["clip_ratio"]
        self.actor_lr = self.ppo_configs["actor_learning_rate"]
        self.critic_lr = self.ppo_configs["critic_learning_rate"]
        self.actor_train_iters = self.ppo_configs["actor_train_iters"]
        self.critic_train_iters = self.ppo_configs["critic_train_iters"]
        self.target_kl = self.ppo_configs["target_kl"]

        self._replay_buffer = OnPolicyReplayBuffer(self.env.observation_space.shape,
                                                  self.env.action_space.shape,
                                                  self.batch_size,
                                                  self.gamma, self.tau)

        self.is_discrete = isinstance(self.env.action_space, gym.spaces.Discrete)

        self.actor = Actor(self.is_discrete, obs_dim, act_dim, self.ppo_configs["hidden_sizes"])
        self.critic = Critic(obs_dim, self.ppo_configs["hidden_sizes"])

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.register_networks(self.actor, self.critic)
        self.load_models()

    def get_explore_action(self, obs):
        obs = utils.to_tensor(obs)
        with torch.no_grad():
            action_distribution, _ = self.actor(obs)
            action = action_distribution.sample()
            log_action = action_distribution.log_prob(action)
            if not self.is_discrete:
                log_action = log_action.sum(axis=-1)
            v = self.critic(obs)
        return utils.from_tensor(action), utils.from_tensor(v), utils.from_tensor(log_action)

    def get_exploit_action(self, obs):
        action, _, _ = self.get_explore_action(obs)
        return action

    def collect_data(self, max_ep_len, num_samples, get_action_func):
        ob = self.env.reset()
        count = 0
        ep_rewards = []
        while count < num_samples:
            ep_len = 0
            done = False
            ep_ret = 0
            while not done and ep_len < min(max_ep_len, num_samples-count):
                action, value, logp = get_action_func(ob)
                nxt_ob, reward, done, _ = self.env.step(action)
                ep_ret += reward
                ep_rewards.append(reward)

                ep_len += 1

                self.replay_buffer.store(ob, action, value, logp, reward)
                ob = nxt_ob
            if done:
                ob = self.env.reset()
                value = 0
            else:
                value = self.critic(utils.to_tensor(ob))
            self.replay_buffer.end_episode(value)

            count += ep_len
            logger.record("training/ep_reward", ep_ret)
            logger.record("training/ep_len", ep_len)

    def load_models(self):
        if self.model_saving_path:
            fname = f"{self.model_saving_path}/actor"
            if os.path.exists(fname):
                self.actor.load_state_dict(torch.load(fname))
            fname = f"{self.model_saving_path}/critic"
            if os.path.exists(fname):
                self.critic.load_state_dict(torch.load(fname))

    def save_models(self):
        if self.model_saving_path:
            torch.save(self.actor.state_dict(), f"{self.model_saving_path}/actor")
            torch.save(self.critic.state_dict(), f"{self.model_saving_path}/critic")

    @property
    def replay_buffer(self):
        return self._replay_buffer

    def train(self, batch: dict, step: int):
        obs = batch["obs"]
        acts = batch["actions"]
        logp = batch["logp"]
        returns = batch["returns"]
        advantages = batch["adv"]

        old_actor_loss, old_actor_info = self.compute_actor_loss(obs, acts, advantages, logp)
        old_actor_loss = old_actor_loss.item()
        old_critic_loss = self.compute_critic_loss(obs, returns).item()

        for i in range(self.actor_train_iters):
            self.actor_optimizer.zero_grad()
            actor_loss, actor_info = self.compute_actor_loss(obs, acts, advantages, logp)
            kl = actor_info['kl']
            if kl > 1.5 * self.target_kl:
                print(f"Early stopping at step {i} due to reaching max kl")
                break

            actor_loss.backward()
            self.actor_optimizer.step()

        for i in range(self.critic_train_iters):
            self.critic_optimizer.zero_grad()
            critic_loss = self.compute_critic_loss(obs, returns)
            critic_loss.backward()
            self.critic_optimizer.step()

        update_info = dict(
            LossPi=old_actor_loss,
            LossV=old_critic_loss,
            KL=old_actor_info["kl"],
            Entropy=old_actor_info["ent"],
            ClipFrac=old_actor_info["cf"],
            DeltaLossPi=(actor_loss.item() - old_actor_loss),
            DeltaLossV=(critic_loss.item() - old_critic_loss)
        )

        return update_info

    def compute_actor_loss(self, obs, acts, advantages, logp_old):
        action_distribution, logp = self.actor(obs, acts)

        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * advantages
        actor_loss = -(torch.min(ratio * advantages, clip_adv)).mean()

        approx_kl = (logp_old - logp).mean().item()
        ent = action_distribution.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()

        actor_info = dict(
            kl=approx_kl,
            ent=ent,
            cf=clipfrac
        )

        return actor_loss, actor_info

    def compute_critic_loss(self, obs, returns):
        val = self.critic(obs)

        return ((val - returns)**2).mean()

