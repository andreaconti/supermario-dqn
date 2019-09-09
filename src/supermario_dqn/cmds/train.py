"""
Handles supermario training
"""

import torch
from supermario_dqn.environment import MarioEnvironment
import supermario_dqn.nn as nn
import supermario_dqn.preprocess as pr
import argparse


if torch.cuda.is_available():
    _device = torch.device('cuda')
else:
    _device = torch.device('cpu')
    print('[Warning] using CPU for training')


def main():
    """
    Starts training, handles many parameters for training and also in order to
    load a .pt state_dict
    """

    # parse arguments
    parser = argparse.ArgumentParser(description='Handle training')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='size of each batch used for training')
    parser.add_argument('--fit_interval', type=int, default=32,
                        help='fit every `fit_interval` examples available')
    parser.add_argument('--gamma', type=float, default=0.999,
                        help='discount rate used for Q-values learning')
    parser.add_argument('--eps_start', type=float, default=0.9,
                        help='start probability to choose a random action')
    parser.add_argument('--eps_end', type=float, default=0.05,
                        help='end probability to choose a random action')
    parser.add_argument('--target_update', type=int, default=10,
                        help='number of episodes between each target dqn update')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='number of episodes between each network checkpoint')
    parser.add_argument('--save_path', type=str, default='model.pt',
                        help='where save trained model')
    parser.add_argument('--memory_size', type=int, default=10000,
                        help='size of replay memory')
    parser.add_argument('--num_episodes', type=int, default=200,
                        help='number of games to be played before end')
    parser.add_argument('--verbose', type=int, default=1,
                        help='verbosity of output')
    parser.add_argument('--load', type=str, default=None,
                        help='load a saved state_dict')

    args = vars(parser.parse_args())

    print('training parameters:')
    for k, v in args.items():
        print('{:15} {}'.format(k, v))

    # create environment, DQN and start training
    env = MarioEnvironment(3, pr.preprocess)
    model = nn.create([3, 60, 110], env.n_actions, load_state_from=args.pop('load'))
    nn.train(model, env, device=_device, **args)
