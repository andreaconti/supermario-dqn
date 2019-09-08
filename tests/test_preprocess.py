"""
Test for preprocessing module
"""

import torch
import gym_super_mario_bros as gym
from supermario_dqn.preprocess import preprocess


def test_preprocess():
    env = gym.make('SuperMarioBros-v0')
    result = preprocess(env.reset())

    assert(type(result) is torch.Tensor)
    assert(result.size(0) == 60)
    assert(result.size(1) == 110)
