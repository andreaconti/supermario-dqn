"""
Test for preprocessing module
"""

import torch
import pytest
import gym_super_mario_bros as gym
from supermario_dqn.preprocess import preprocess


@pytest.mark.skip(reason='too long to test each time')
def test_preprocess():

    for world in range(1, 9):
        for stage in range(1, 5):
            env = gym.make(f'SuperMarioBros-{world}-{stage}-v0')
            result = preprocess(world, stage, env.reset(), 35, 56)

            assert(type(result) is torch.Tensor)
            assert(result.size(0) == 35)
            assert(result.size(1) == 56)
