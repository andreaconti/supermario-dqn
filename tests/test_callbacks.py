"""
Tests for callbacks module
"""

from supermario_dqn.nn.train.callbacks import log_episodes
import os


def test_log_episodes():

    # test on create
    logger = log_episodes('/tmp/test_log_episodes.csv')
    handle = logger('init', None, None)
    logger('run', handle, {
        'train_id': 1,
        'model': None,
        'optimizer': None,
        'episode': 1,
        'episodes': 100,
        'reward': 56,
        'steps': 2300,
        'choosen_moves': 2000,
        'random_moves': 300,
    })
    logger('close', handle, None)

    assert(os.path.isfile('/tmp/test_log_episodes.csv'))
    assert(len(open('/tmp/test_log_episodes.csv').readlines()) == 2)

    # test on append
    logger = log_episodes('/tmp/test_log_episodes.csv')
    handle = logger('init', None, None)
    logger('run', handle, {
        'train_id': 1,
        'model': None,
        'optimizer': None,
        'episode': 1,
        'episodes': 100,
        'reward': 56,
        'steps': 2300,
        'choosen_moves': 2000,
        'random_moves': 300,
    })
    logger('close', handle, None)

    assert(len(open('/tmp/test_log_episodes.csv').readlines()) == 3)

    os.remove('/tmp/test_log_episodes.csv')
