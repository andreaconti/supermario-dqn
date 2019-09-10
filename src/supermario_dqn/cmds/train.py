"""
Handles supermario training
"""

import torch
from supermario_dqn.environment import MarioEnvironment
import supermario_dqn.nn as nn
import supermario_dqn.preprocess as pr
import argparse
from supermario_dqn.cmds.play import main as show_game


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
    parser.add_argument('--batch_size', type=int, default=256,
                        help='size of each batch used for training')
    parser.add_argument('--fit_interval', type=int, default=4,
                        help='fit every `fit_interval` examples available')
    parser.add_argument('--gamma', type=float, default=0.98,
                        help='discount rate used for Q-values learning')
    parser.add_argument('--eps_start', type=float, default=0.9,
                        help='start probability to choose a random action')
    parser.add_argument('--eps_end', type=float, default=0.25,
                        help='end probability to choose a random action')
    parser.add_argument('--target_update', type=int, default=15,
                        help='number of episodes between each target dqn update')
    parser.add_argument('--save_interval', type=int, default=30,
                        help='number of episodes between each network checkpoint')
    parser.add_argument('--save_path', type=str, default='model.pt',
                        help='where save trained model')
    parser.add_argument('--memory_size', type=int, default=100000,
                        help='size of replay memory')
    parser.add_argument('--num_episodes', type=int, default=4000,
                        help='number of games to be played before end')
    parser.add_argument('--verbose', type=int, default=1,
                        help='verbosity of output')
    parser.add_argument('--load', type=str, default=None,
                        help='load a saved state_dict')
    parser.add_argument('--log_file_dir', type=str, default=None,
                        help='file path where write logs')
    parser.add_argument('--finally_show', action='store_true',
                        help='finally show a play')

    args = vars(parser.parse_args())

    # log params
    show = args.pop('finally_show')
    print('training parameters:')
    for k, v in args.items():
        print('{:15} {}'.format(k, v))
    if args['log_file_dir'] is not None:
        with open('parameters.log', 'w') as params_log:
            params_log.write('{:15} {}\n'.format(k, v))

    # create environment, DQN and start training
    env = MarioEnvironment(4, lambda t: pr.preprocess(t, 30, 56))
    model = nn.create([4, 30, 56], env.n_actions, load_state_from=args.pop('load'), for_train=True)
    nn.train(model, env, device=_device, **args)

    # show
    if show:
        model.eval()
        show_game(model.to('cpu'))
