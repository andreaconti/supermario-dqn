"""
Callbacks coroutines used by training functions
"""

import os
import time
import torch


__ALL__ = ['console_logger', 'log_episodes', 'save_model']


def console_logger(mode, _, info):
    if mode == 'init':
        print('start training')

    if mode == 'run':
        print(f'episode {info["episode"]}/{info["episodes"]} ({info["steps"]}) | reward: {info["reward"]}')

    if mode == 'close':
        print('end training.')


def log_episodes(path):
    """
    Returns a callback that saves a log of episodes
    """

    def log_episodes_(mode, f, info):
        if mode == 'init':

            if os.path.isfile(path):
                f = open(path, 'a')
            else:
                f = open(path, 'w')
                f.write('time,episode,reward,steps,choosen_moves,random_moves\n')
            return f

        elif mode == 'run':

            f.write('{},{},{},{},{}\n'.format(int(time.time()), info['episode'], info['reward'], info['steps'], info['choosen_moves'], info['random_moves']))  # noqa
            return f

        elif mode == 'close':
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


def model_checkpoint(path: str, interval: int):
    """
    Returns a callback that checkpoints the model
    """
