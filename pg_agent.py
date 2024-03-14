import itertools
from torch import nn

import numpy as np
import torch
from torch.nn import functional as F
import utils


class MLPPolicy(nn.Module):

    def __init__(self, ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate):
        super().__init__()

        self.discrete = discrete
        if discrete:
            self.logits_net = utils.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size
            ).to(utils.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = utils.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size
            )
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=utils.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = torch.optim.Adam(
            parameters,
            learning_rate
        )

    @torch.no_grad()
    def get_action(self, obs:np.ndarray)-> np.ndarray:
        if len(obs.shape) > 1:
            obs = obs
        else:
            obs = obs[None]

        state = utils.from_numpy(obs)

        action_distribution = self(state)
        action = action_distribution.sample()

        return utils.to_numpy(action)

    def forward(self, obs: torch.FloatTensor):
        if self.discrete:
            logits = self.logits_net(obs)
            return torch.distributions.Categorical(logits=logits)
        else:
            mean = self.mean_net(obs)
            return torch.distributions.Normal(
                mean,
                torch.exp(self.logstd)[None]
            )

    def update(self, obs: np.ndarray, actions: np.ndarray,
               advantages: np.ndarray):
        obs = utils.from_numpy(obs)
        advantages = utils.from_numpy(advantages)
        actions = utils.from_numpy(actions)

        action_distribution = self(obs)
        log_probs = action_distribution.log_prob(actions)
        loss = -(log_probs * advantages).sum()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            "Actor Loss": utils.to_numpy(loss)
        }


class ValueCritic(nn.Module):

    def __init__(self, ob_dim, n_layers, layer_size, learning_rate):
        super().__init__()

        self.network = utils.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=layer_size
        )

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            learning_rate
        )

    def forward(self, obs: torch.Tensor):
        return self.network(obs)

    def update(self, obs, q_values):
        obs = utils.from_numpy(obs)
        q_values = utils.from_numpy(q_values)

        predictions = self(obs)
        loss = F.mse_loss(predictions, q_values)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            "Baseline Loss": utils.to_numpy(loss)
        }


class PGAgent(nn.Module):

    def __init__(self,
                 ob_dim,
                 ac_dim,
                 discrete,
                 n_layers,
                 layer_size,
                 gamma,
                 learning_rate,
                 baseline_learning_rate,
                 baseline_gradient_steps,
                 gae_lambda
                 ):
        super().__init__()

        self.actor = MLPPolicy(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        self.critic = ValueCritic(
            ob_dim, n_layers, layer_size, baseline_learning_rate
        )
        self.baseline_gradient_steps = baseline_gradient_steps

        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def update(self,
               obs, actions, rewards, terminals):
        q_values = self._calculate_q_vals(rewards)

        obs = np.concatenate(obs)
        actions = np.concatenate(actions)
        rewards = np.concatenate(rewards)
        q_values = np.concatenate(q_values)
        terminals = np.concatenate(terminals)

        advantages = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        info: dict = self.actor.update(obs, actions, advantages)

        critic_info: dict = self.critic.update(obs, q_values)
        info.update(critic_info)

        return info

    def _calculate_q_vals(self, rewards):
        q_values = []
        for i in range(len(rewards)):
            q_value = []
            for t in range(len(rewards[i])):
                ret = 0
                gamma = 1
                for t_prime in range(t, len(rewards[i])):
                    ret += gamma * rewards[i][t_prime]
                    gamma *= self.gamma
                q_value.append(ret)
            q_values.append(q_value)

        return q_values

    def _estimate_advantage(self,
                            obs, rewards, q_values, terminals):
        critic_vals = self.critic(utils.from_numpy(obs))
        values = utils.to_numpy(critic_vals.squeeze())
        assert values.shape == q_values.shape

        batch_size = obs.shape[0]
        values = np.append(values, [0])
        advantages = np.zeros(batch_size+1)

        for i in range(batch_size-1, -1, -1):
            delta = rewards[i] + (1 - int(terminals[i])) * self.gamma * values[i] - values[i+1]
            advantages[i] = delta + self.gamma * (1 - int(terminals[i])) * advantages[i+1]

        advantages = advantages[:-1]
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages