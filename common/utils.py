import torch
import numpy as np
import torch.nn as nn
import random
import yaml


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_config(yml_cfg_path):
    with open(yml_cfg_path, "r") as i_f:
        configs = yaml.load(i_f, Loader=yaml.FullLoader)
    return configs


def to_tensor(data, dtype=torch.float):
    return torch.tensor(data, dtype=dtype, device=DEVICE)


def from_tensor(data):
    return data.detach().numpy()


def validate_agent(env, get_action_func,
                   logger,
                   max_ep_len,
                   num_eval_episodes=10,
                   render=False):
    ep_rewards = []
    ep_lens = []
    for _ in range(num_eval_episodes):
        ob = env.reset()
        ep_len = 0
        ep_rew = 0
        done = False
        while not done and ep_len < max_ep_len:
            if render:
                env.render()

            action = get_action_func(ob)

            nxt_ob, reward, done, info = env.step(action)
            ep_len += 1
            ep_rew += reward
            ob = nxt_ob

        ep_lens.append(ep_len)
        ep_rewards.append(ep_rew)

        if logger:
            logger.record("eval/ep_reward", ep_rew)
            logger.record("eval/ep_len", ep_len)


def build_net(input_size, output_size, hidden_layer_sizes,
              activation=torch.nn.Tanh, output_activation=torch.nn.Identity):
    layers = []
    for hidden_sz in hidden_layer_sizes:
        fc_layer = nn.Linear(input_size, hidden_sz)
        layers.append(fc_layer)
        layers.append(activation())
        input_size = hidden_sz
    if output_size:
        last_fc_layer = nn.Linear(input_size, output_size)
        layers.append(last_fc_layer)
        if output_activation:
            layers.append(output_activation())

    return nn.Sequential(*layers).to(DEVICE)