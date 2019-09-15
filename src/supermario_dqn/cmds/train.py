"""
Handles supermario training
"""

import torch
import torch.multiprocessing as mp
import os
from supermario_dqn.env import MarioEnvironment, SIMPLE_ACTIONS
import supermario_dqn.nn as nn
import supermario_dqn.preprocess as pr
import argparse
from supermario_dqn.cmds.play import main as show_game


if torch.cuda.is_available():
    _device = torch.device('cuda')
else:
    _device = torch.device('cpu')
    print('[Warning] using CPU for training')


def _create_and_train(proc_index, model, args):

    args_ = args

    # identify multiple workers
    if proc_index is not None:
        args_['log_postfix'] = str(proc_index)

    # disable saving for multiple workers
    if proc_index is not None and proc_index != 0:
        args_['save_interval'] = None

    env = MarioEnvironment(SIMPLE_ACTIONS, 4, lambda w, s, t: pr.preprocess(w, s, t, 30, 56),
                           random=args_.pop('random'),
                           render=args_.pop('render'),
                           world_stage=args_.pop('world_stage'))
    nn.train(model, env, device=_device, **args_)


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
    parser.add_argument('--random', action='store_true',
                        help='choose randomly different worlds and stages')
    parser.add_argument('--render', action='store_true',
                        help='rendering of frames, only for debug')
    parser.add_argument('--workers', type=int, default=1,
                        help='multiprocessing enable')
    parser.add_argument('--world_stage', type=int, nargs=2, default=None,
                        help='select specific world and stage')

    args = vars(parser.parse_args())

    if args['world_stage'] is not None:
        args['random'] = False

    # log params
    show = args.pop('finally_show')
    workers = args.pop('workers')
    print('training parameters:')
    for k, v in args.items():
        print('{:15} {}'.format(k, v))
    if args['log_file_dir'] is not None:
        with open(os.path.join(args['log_file_dir'], 'parameters.log'), 'w') as params_log:
            for k, v in args.items():
                params_log.write('{:15} {}\n'.format(k, v))
    else:
        with open('parameters.log', 'w') as params_log:
            for k, v in args.items():
                params_log.write('{:15} {}\n'.format(k, v))

    # create environment, DQN and start training
    model = nn.create([4, 30, 56], len(SIMPLE_ACTIONS), load_state_from=args.pop('load'), for_train=True)
    if workers == 1:
        _create_and_train(None, model, args)
    elif workers > 1:
        model.share_memory()

        args['num_episodes'] = args['num_episodes'] // workers
        args['memory_size'] = args['memory_size'] // workers

        mp.spawn(_create_and_train, args=(model, args), nprocs=workers, join=True)
    else:
        print('[Error] workers >= 1')

    print('end of training.')

    # show
    if show:
        model.eval()
        show_game(model.to('cpu'))
