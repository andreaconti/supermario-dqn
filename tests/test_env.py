"""
Test environment utilities
"""

import supermario_dqn.environment as mario_env
from supermario_dqn.preprocess import preprocess
import torch


def test_reset():
    env = mario_env.MarioEnvironment(3, preprocess)
    state = env.reset()

    assert(type(state) == torch.Tensor)
    assert(state.shape == torch.Size([3, 60, 110]))


def test_step():
    env = mario_env.MarioEnvironment(3, preprocess)
    env.reset()
    state, _, _, _ = env.step(0)

    assert(type(state) == torch.Tensor)
    assert(state.shape == torch.Size([3, 60, 110]))
