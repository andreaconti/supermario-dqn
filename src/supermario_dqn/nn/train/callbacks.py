"""
Callbacks coroutines used by training functions
"""

import os
import time
import torch


__ALL__ = ['console_logger', 'log_episodes', 'save_model', 'model_checkpoint']


def console_logger(mode, _, info):
    if mode == 'init':
        print('start training')

    if mode == 'run':
        print(f'episode {info["episode"]}/{info["episodes"]} ({info["steps"]}) | reward: {info["reward"]}')

    if mode == 'close':
        print('end training.')


def log_episodes(path_or_file, close=True):
    """
    Returns a callback that saves a log of episodes
    """

    def log_episodes_(mode, f, info):
        if mode == 'init':
            if type(path_or_file) is str:
                if os.path.isfile(path_or_file):
                    f = open(path_or_file, 'a')
                else:
                    f = open(path_or_file, 'a')
                    f.write('time,episode,reward,steps\n')
            else:
                f = path_or_file
            return f

        elif mode == 'run':
            f.write('{},{},{},{}\n'.format(int(time.time()), info['episode'], info['reward'], info['steps']))
            return f

        elif mode == 'close' and close:
            f.close()

    return log_episodes_


def save_model(path: str, interval: int, verbose: bool = False):
    """
    Returns a callback that saves model every :interval: episodes
    """

    def save_model_(mode, counter, info):
        if mode == 'init':
            return 0

        if mode == 'run':
            counter = counter + 1
            if counter % interval == 0:
                if os.path.isfile(path):
                    os.remove(path)
                torch.save(info['model'].state_dict(), path)
                if verbose:
                    print('model saved')
            return counter

    return save_model_


def model_checkpoint(path_dir: str, interval: int):
    """
    Returns a callback that checkpoints the model
    """

    def model_checkpoint_(mode, counter, info):
        if mode == 'init':
            if not os.path.isdir(path_dir):
                os.mkdir(path_dir)
            return 0

        if mode == 'run':
            counter = counter + 1
            if counter % interval == 0:
                train_id = info['train_id']
                curr_episode = info['episode']
                ckpt_dir = os.path.join(path_dir, 'ckpt_at_' + str(curr_episode))
                if not os.path.isdir(ckpt_dir):
                    os.mkdir(ckpt_dir)
                torch.save({
                    'model_state_dict': info['model'].state_dict(),
                    'optimizer_state_dict': info['optimizer'].state_dict(),
                    'episodes_left': info['episodes'] - info['episode'],
                    'steps_done': info['total_steps']
                }, os.path.join(ckpt_dir, f'ckpt_{train_id}.ckpt'))
            return counter

    return model_checkpoint_
