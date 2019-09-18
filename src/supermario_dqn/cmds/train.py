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


def _find_ckpts(path):
    return [torch.load(f) for f in os.listdir(path) if f.endswith('.ckpt') and os.path.isfile(f)]


def _create_and_train(proc_index, device, model, args):

    # define environment
    env = MarioEnvironment(SIMPLE_ACTIONS, 4, lambda w, s, t: pr.preprocess(w, s, t, 30, 56),
                           random=args.pop('random'),
                           render=args.pop('render'),
                           world_stage=args.pop('world_stage'))

    # define callbacks
    save_path = args.pop('save_path')
    callbacks = [
        nn.train.callbacks.console_logger,
        nn.train.callbacks.log_episodes('episodes.csv')
    ]
    ckpt_interval = args.pop('checkpoint')
    if ckpt_interval is not None:
        callbacks.append(nn.train.callbacks.model_checkpoint('checkpoints', ckpt_interval))

    # set Id
    train_id = 0 if proc_index is None else proc_index

    # resume checkpoint
    steps_done = 0
    optimizer_state_dict = None
    resume = args.pop('resume')
    if resume is not None:
        if proc_index is None and os.path.isfile(resume):
            checkpoint = torch.load(resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            args['num_episodes'] = checkpoint['episodes_left']
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            steps_done = checkpoint['steps_done']
            print(f'loaded from ckpt: episodes left {checkpoint["episodes_left"]}')
        if proc_index is not None and os.path.isdir(resume):
            checkpoints = _find_ckpts(resume)
            if proc_index == 0:
                model.load_state_dict(checkpoints[proc_index]['model_state_dict'])
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            args['num_episodes'] = checkpoints[proc_index]['episodes_left']
            steps_done = checkpoints[proc_index]['steps_done']
            print(f'worker {proc_index} episodes left: {checkpoints[proc_index]["episodes_left"]}')

    # define memory
    memory = nn.train.RandomReplayMemory(args.pop('memory_size'))

    # train
    nn.train.train_dqn(model,
                       env,
                       memory=memory,
                       action_policy=nn.train.epsilon_greedy_choose(
                           args.pop('eps_start'),
                           args.pop('eps_end'),
                           args.pop('eps_decay'),
                           initial_step=steps_done),
                       device=device,
                       callbacks=callbacks,
                       train_id=train_id,
                       optimizer_state_dict=optimizer_state_dict,
                       **args)

    # save
    if os.path.isfile(save_path):
        os.remove(save_path)
    torch.save(model.state_dict(), save_path)


def main():
    """
    Starts training, handles many parameters for training and also in order to
    load a .pt state_dict
    """

    if torch.cuda.is_available():
        _device = torch.device('cuda')
    else:
        _device = torch.device('cpu')
        print('[Warning] using CPU for training')

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
    parser.add_argument('--eps_decay', type=float, default=200,
                        help='decay of eps probabilities')
    parser.add_argument('--target_update', type=int, default=15,
                        help='number of episodes between each target dqn update')
    parser.add_argument('--save_path', type=str, default='model.pt',
                        help='where save trained model')
    parser.add_argument('--memory_size', type=int, default=100000,
                        help='size of replay memory')
    parser.add_argument('--num_episodes', type=int, default=4000,
                        help='number of games to be played before end')
    parser.add_argument('--resume', type=str, default=None,
                        help='load from a checkpoint')
    parser.add_argument('--checkpoint', type=int, default=None,
                        help='number of episodes between each network checkpoint')
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
    workers = args.pop('workers')
    if args['resume'] is None:
        print('training parameters:')
        for k, v in args.items():
            print('{:15} {}'.format(k, v))
        with open('parameters.log', 'w') as params_log:
            for k, v in args.items():
                params_log.write('{:15} {}\n'.format(k, v))

    # resume checks
    if args['resume'] is not None and os.path.isfile(args['resume']) and not workers == 1:
        print(f'[Error] cannot resume a checkpoint with {workers} workers')
        return
    if args['resume'] is not None and os.path.isdir(args['resume']) and workers != len(_find_ckpts(args['resume'])):
        print(f'[Error] cannot resume checkpoints with {workers} workers')
        return

    # create environment, DQN and start training
    model = nn.create([4, 30, 56], len(SIMPLE_ACTIONS), for_train=True)

    if workers == 1:
        _create_and_train(None, _device, model, args)
    elif workers > 1:
        model.share_memory()

        args['num_episodes'] = args['num_episodes'] // workers
        args['memory_size'] = args['memory_size'] // workers

        mp.spawn(_create_and_train, args=(_device, model, args), nprocs=workers, join=True)
    else:
        print('[Error] workers >= 1')
