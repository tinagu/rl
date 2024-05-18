import argparse
from algs.dqn_agent import DQNAgent
from algs.ppo_agent import PPOAgent
from common.logger import logger
from common.utils import load_config, set_random_seed


def run_dqn(configs):
    dqn_agent = DQNAgent(configs)

    dqn_agent.learn()
    if configs["test"]:
        dqn_agent.test_agent()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--test", default=False, action='store_true')
    args = parser.parse_args()

    configs = vars(args)
    configs.update(load_config(args.config))

    set_random_seed(configs["seed"])
    logger.setup(configs)

    run_dqn(configs)

