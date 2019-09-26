"""
Handles supermario training
"""

import torch
import os
from supermario_dqn.env import MarioEnvironment, SIMPLE_ACTIONS
import supermario_dqn.nn as nn
import supermario_dqn.preprocess as pr
import argparse

# Utils functions and variables

_actions = SIMPLE_ACTIONS
_actions_map = {
    'simple': SIMPLE_ACTIONS,
}


def _create_and_train(proc_index, device, model, target_net, args):

    # handle actions
    choosen_actions = _actions

    # resume checkpoint
    steps_done = 0
    start_episode = 0
    optimizer_state_dict = None
    resume = args.pop('resume')
    if resume is not None:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_episode = checkpoint['episode']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        steps_done = checkpoint['steps_done']
        choosen_actions = checkpoint['actions']

    # define environment
    env = MarioEnvironment(choosen_actions, 4, lambda w, s, t: pr.preprocess(w, s, t, 52, 56),
                           random=args.pop('random'),
                           render=args.pop('render'),
                           world_stage=args.pop('world_stage'))

    # define callbacks
    callbacks = [nn.train.callbacks.console_logger(start_episode=start_episode)]
    log = args.pop('log')
    if log:
        callbacks.append(nn.train.callbacks.log_episodes('episodes.csv', start_episode=start_episode))
    ckpt_interval = args.pop('checkpoint')
    if ckpt_interval is not None:
        callbacks.append(nn.train.callbacks.model_checkpoint('checkpoints', ckpt_interval, meta={
            'actions': choosen_actions
        }, start_episode=start_episode))
    test = args.pop('test')
    if test is not None and (proc_index == 0 or proc_index is None):
        callbacks.append(nn.train.callbacks.test_model(env, 'episodes_test.csv', test))

    # define memory
    memory = nn.train.RandomReplayMemory(args.pop('memory_size'))

    # train
    save_path = args.pop('save_path')
    nn.train.train_dqn(model,
                       target_net,
                       env,
                       memory=memory,
                       action_policy=nn.train.epsilon_greedy_choose(
                           args.pop('eps_start'),
                           args.pop('eps_end'),
                           args.pop('eps_decay'),
                           initial_step=steps_done),
                       device=device,
                       callbacks=callbacks,
                       optimizer_state_dict=optimizer_state_dict,
                       **args)

    # save
    if os.path.isfile(save_path):
        os.remove(save_path)
    torch.save(model.state_dict(), save_path)


# main

def main():
    """
    Starts training, handles many parameters for training and also in order to
    load a .pt state_dict
    """

    if torch.cuda.is_available():
        device = torch.device('cuda')  # noqa
    else:
        device = torch.device('cpu')
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
    parser.add_argument('--eps_end', type=float, default=0.15,
                        help='end probability to choose a random action')
    parser.add_argument('--eps_decay', type=float, default=200000,
                        help='decay of eps probabilities')
    parser.add_argument('--target_update', type=int, default=15,
                        help='number of episodes between each target dqn update')
    parser.add_argument('--save_path', type=str, default='model.pt',
                        help='where save trained model')
    parser.add_argument('--memory_size', type=int, default=100000,
                        help='size of replay memory')
    parser.add_argument('--num_episodes', type=int, default=5000,
                        help='number of games to be played before end')
    parser.add_argument('--resume', type=str, default=None,
                        help='load from a checkpoint')
    parser.add_argument('--checkpoint', type=int, default=None,
                        help='number of episodes between each network checkpoint')
    parser.add_argument('--random', action='store_true',
                        help='choose randomly different worlds and stages')
    parser.add_argument('--render', action='store_true',
                        help='rendering of frames, only for debug')
    parser.add_argument('--world_stage', type=int, nargs=2, default=None,
                        help='select specific world and stage')
    parser.add_argument('--actions', type=str, default='simple',
                        help='select actions used between ["simple"]')
    parser.add_argument('--test', type=int, default=None,
                        help='each `test` episodes network is used and tested over an episode')
    parser.add_argument('--log', action='store_true',
                        help='logs episodes results')
    parser.add_argument('--algorithm', default='double',
                        help='algorithm used for training, double DQN by default but is also possible to use simple "deep"')  # noqa

    args = vars(parser.parse_args())

    # handle world stage
    if args['world_stage'] is not None:
        args['random'] = False

    # handle choosen actions
    actions_type = args.pop('actions')
    if actions_type not in ['simple']:
        print("[Error] actions not found")
        return
    _actions = _actions_map[actions_type]

    # log params
    if args['resume'] is None:
        print_params = args.copy()
        print_params['actions'] = _actions
        print('training parameters:')
        for k, v in print_params.items():
            print('{:15} {}'.format(k, v))
        with open('parameters.log', 'w') as params_log:
            for k, v in print_params.items():
                params_log.write('{:15} {}\n'.format(k, v))

    # resume checks
    if args['resume'] is not None and not os.path.isfile(args['resume']):
        print(f'[Error] resume from a .ckpt file')
        return

    # create environment, DQN and start training
    policy_net = nn.create([4, 52, 56], len(_actions), for_train=True)
    target_net = nn.create([4, 52, 56], len(_actions), for_train=False)
    _create_and_train(None, device, policy_net, target_net, args)
