"""
Test environment utilities
"""

import supermario_dqn.env as mario_env
from supermario_dqn.preprocess import preprocess
import torch


def test_reset():
    env = mario_env.MarioEnvironment(mario_env.SIMPLE_ACTIONS, 3, lambda w, s, t: preprocess(w, s, t, 35, 60))
    state = env.reset()

    assert(type(state) == torch.Tensor)
    assert(state.shape == torch.Size([3, 35, 60]))


def test_step():
    env = mario_env.MarioEnvironment(mario_env.SIMPLE_ACTIONS, 3, lambda w, s, t: preprocess(w, s, t, 35, 60))
    env.reset()
    state, _, _, _ = env.step(0)

    assert(type(state) == torch.Tensor)
    assert(state.shape == torch.Size([3, 35, 60]))


def test_world_stage():
    env = mario_env.MarioEnvironment(mario_env.SIMPLE_ACTIONS,
                                     3, lambda w, s, t: preprocess(w, s, t, 35, 60), world_stage=(2, 3))
    env.reset()
    state, _, _, _ = env.step(0)


def test_random():
    env = mario_env.MarioEnvironment(mario_env.SIMPLE_ACTIONS,
                                     3, lambda w, s, t: preprocess(w, s, t, 35, 60), random=True)
    env.reset()

    curr_world = env._world
    curr_stage = env._stage

    state, _, _, _ = env.step(0)

    env.reset()
    assert(curr_world != env._world)
    assert(curr_stage != env._stage)
