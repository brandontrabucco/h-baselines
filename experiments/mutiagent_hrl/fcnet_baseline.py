"""A runner script for fcnet models.

This run script used to test the performance of DDPG and DQN with fully
connected network models on various environments.
"""
import os
import datetime
import csv
import ray

from hbaselines.utils.logger import ensure_dir
from hbaselines.utils.train import create_parser, get_hyperparameters
from hbaselines.algs.ddpg import DDPG
from hbaselines.algs.dqn import DQN
from stable_baselines.deepq.policies import MlpPolicy as DQNPolicy
from hbaselines.policies.ddpg import FullyConnectedPolicy as DDPGPolicy

EXAMPLE_USAGE = 'python fcnet_baseline.py "HalfCheetah-v2" --gamma 0.995'
NUM_CPUS = 2
discrete = False


@ray.remote
def run_exp(env, hp, steps, dir_name, i):
    # initialize the algorithm
    if discrete:
        # if discrete, use DQN
        alg = DQN(policy=DQNPolicy, env=env, **hp)
    else:
        # if continuous, use DDPG
        alg = DDPG(policy=DDPGPolicy, env=env, **hp)

    # perform training
    _ = alg.learn(
        total_timesteps=steps,
        log_interval=10,
        file_path=os.path.join(dir_name, "results_{}.csv".format(i)))

    return None


def main():
    parser = create_parser(
        description='Test the performance of DDPG and DQN with fully connected'
                    ' network models on various environments.',
        example_usage=EXAMPLE_USAGE)
    args = parser.parse_args()

    # create a save directory folder (if it doesn't exist)
    dir_name = 'data/fcnet/{}'.format(datetime.datetime.now().time())
    ensure_dir(dir_name)

    # if the environment is in Flow or h-baselines, register it
    env = args.env_name

    # determine whether the env is discrete or continuous in the action space
    discrete = False

    # get the hyperparameters
    hp = get_hyperparameters(args, discrete)

    # add the hyperparameters to the folder
    with open(os.path.join(dir_name, 'hyperparameters.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=hp.keys())
        w.writeheader()
        w.writerow(hp)

    ray.init(num_cpus=NUM_CPUS)
    _ = ray.get([run_exp.remote(env, hp, args.steps, dir_name, i)
                 for i in range(args.n_training)])
    ray.shutdown()


if __name__ == '__main__':
    main()
    os._exit(1)