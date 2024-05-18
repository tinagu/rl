import gym
import os
import abc
from common.logger import logger
import common.utils as utils


class Agent(object, metaclass=abc.ABCMeta):

    def __init__(self, configs):
        super().__init__()

        self.configs = configs

        self.env = gym.make(configs["env_id"])
        self.test_env = gym.make(configs["env_id"])
        if configs.get("render_mode"):
            self.render = True
            self.render_env = gym.make(configs["env_id"], render_mode=configs["render_mode"])
        else:
            self.render = False
            self.render_env = None

        self.train_configs = self.configs["train"]
        self.batch_size = self.train_configs["batch_size"]
        self.num_epoch = self.train_configs["num_epoch"]
        self.max_ep_len = self.train_configs.get("max_ep_len") or self.env.spec.max_episode_steps
        self.min_explorations_before_train = self.train_configs["min_explorations_before_train"]
        self.num_explorations_per_epoch_train = self.train_configs["num_explorations_per_epoch"]
        self.num_train_loop_per_epoch = self.train_configs["num_train_loop_per_epoch"]
        self.num_train_per_train_loop = self.train_configs["num_train_per_train_loop"]
        self.num_validate_episode = self.train_configs["num_validate_episode"]

        self.model_saving_interval = self.configs.get("model_saving_interval")
        self.model_saving_path = self.configs.get("load_from_checkpoint")
        if self.model_saving_path:
            os.makedirs(self.model_saving_path, exist_ok=True)

        self.networks = []

    @abc.abstractmethod
    def save_models(self):
        pass

    @abc.abstractmethod
    def get_explore_action(self, ob):
        pass

    @abc.abstractmethod
    def get_exploit_action(self, ob):
        pass

    @abc.abstractproperty
    def replay_buffer(self):
        return None

    @abc.abstractmethod
    def collect_data(self, max_ep_len, num_samples, get_action_func):
        pass

    @abc.abstractmethod
    def train(self, batch, step: int) -> dict:
        pass

    def register_networks(self, *vargs):
        self.networks += vargs

    def set_training_mode(self, training=True):
        for net in self.networks:
            if training:
                net.train()
            else:
                net.eval()

    def learn(self):
        test_env = self.test_env if not self.render else self.render_env
        for epoch_idx in range(self.num_epoch):
            for tl_idx in range(self.num_train_loop_per_epoch):
                self.collect_data(self.max_ep_len, self.num_explorations_per_epoch_train,
                                  self.get_explore_action)

                self.set_training_mode()
                for training_loop_idx in range(self.num_train_per_train_loop):
                    batch = self.replay_buffer.sample_batch(batch_size=self.batch_size)
                    training_info = self.train(batch, training_loop_idx)
                    if training_info:
                        for k, v in training_info.items():
                            logger.record(f"training/{k}", v)

            self.set_training_mode(False)
            utils.validate_agent(test_env, self.get_exploit_action,
                                 logger,
                                 self.max_ep_len,
                                 self.num_validate_episode,
                                 self.render)
            logger.dump(epoch_idx=epoch_idx)
            if epoch_idx % self.model_saving_interval == 0:
                self.save_models()

    def test_agent(self, render_mode=None, num_test_episode=1000, max_ep_len=1000):
        if render_mode:
            env = gym.make(self.configs["env_id"], render_mode=render_mode)
        else:
            env = self.test_env

        utils.validate_agent(env, self.get_exploit_action,
                             None,
                             max_ep_len,
                             num_test_episode,
                             True)