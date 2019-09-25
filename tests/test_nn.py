"""
Testing NN routines
"""

import supermario_dqn.nn as nn
import torch
import os


def test_create_save_load():

    # tries to build
    dqn = nn.create([3, 60, 110], 11)

    # save
    torch.save(dqn.state_dict(), '/tmp/dict.pt')
    del dqn

    # try to load from disk
    nn.create([3, 60, 110], 11, load_state_from='/tmp/dict.pt')
    os.remove('/tmp/dict.pt')


def test_best_action():
    dqn = nn.create([3, 60, 110], 11)
    dqn.requires_grad_(False)
    choosen = nn.best_action(dqn, torch.randn(3, 60, 110))

    assert(choosen in range(11))
