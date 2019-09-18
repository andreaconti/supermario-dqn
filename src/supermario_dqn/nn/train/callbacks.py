"""
Callbacks coroutines used by training functions
"""

import os
import time
import torch
from supermario_dqn.nn.model import best_action


__ALL__ = ['console_logger', 'log_episodes', 'save_model', 'model_checkpoint']


def console_logger(start_episode=0):

    def console_logger_(mode, _, info):
        if mode == 'init':
            print('start training')

        if mode == 'run':
            print(f'episode {start_episode+info["episode"]}/{start_episode+info["episodes"]} ({info["steps"]}) | reward: {info["reward"]}')  # noqa

        if mode == 'close':
            print('end training.')

    return console_logger_


def log_episodes(path_or_file, close=True, start_episode=0, insert_head=True):
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
                    if insert_head:
                        f.write('time,episode,reward,steps\n')
                        f.flush()
            else:
                f = path_or_file
            return f

        elif mode == 'run':
            f.write('{},{},{},{}\n'.format(
                int(time.time()),
                start_episode + info['episode'],
                info['reward'],
                info['steps']
            ))
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


def model_checkpoint(path_dir: str, interval: int, meta: dict = None, start_episode=0):
    """
    Returns a callback that checkpoints the model
    """

    if not type(meta) is dict:
        raise ValueError('meta is None or a dict')

    def model_checkpoint_(mode, counter, info):
        if mode == 'init':
            if not os.path.isdir(path_dir):
                try:
                    os.mkdir(path_dir)
                except FileExistsError:
                    pass
            return 0

        if mode == 'run':
            counter = counter + 1
            if counter % interval == 0:
                train_id = info['train_id']
                curr_episode = info['episode']
                ckpt_dir = os.path.join(path_dir, 'ckpt_at_' + str(start_episode + curr_episode))
                if not os.path.isdir(ckpt_dir):
                    os.mkdir(ckpt_dir)

                tosave = {
                    'model_state_dict': info['model'].state_dict(),
                    'optimizer_state_dict': info['optimizer'].state_dict(),
                    'episode': start_episode + info['episode'],
                    'steps_done': info['total_steps']
                }
                if meta is not None:
                    for k, v in meta.items():
                        tosave[k] = v
                torch.save(tosave, os.path.join(ckpt_dir, f'ckpt_{train_id}.ckpt'))
            return counter

    return model_checkpoint_


def test_model(env, save_path, interval, start_episode=0, insert_head=True):
    """
    Play a test game and save results
    """

    def convert_tensor(vector):
        if type(vector) is torch.Tensor:
            return vector
        else:
            return torch.tensor(vector)

    def test_model_(mode, state, info):
        if mode == 'init':
            if os.path.isfile(save_path):
                f = open(save_path, 'a')
            else:
                f = open(save_path, 'a')
                if insert_head:
                    f.write('episode,reward\n')
                    f.flush()
            counter = 0
            return f, counter

        if mode == 'run':
            f, counter = state
            counter += 1
            if counter % interval == 0:
                model = info['model']
                reward = 0
                with torch.no_grad():
                    obs = env.reset()
                    done = False
                    while not done:
                        obs, r, done, _ = env.step(
                                best_action(model, convert_tensor(obs).to(info['device']).unsqueeze(0))
                        )
                        reward += r

                f.write('{},{}\n'.format(info['episode'] + start_episode, reward))
                f.flush()
            return f, counter

        if mode == 'close':
            f, _ = state
            f.close()

    return test_model_
