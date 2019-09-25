"""
Tests for exploration policy
"""

import supermario_dqn.nn.train.exploration as ex
import torch


def test_eps_start_1_choose_random():
    """
    Tests that if eps_start = 1 and eps_end = 0 at step=0
    calls a random action
    """

    policy = ex.epsilon_greedy_choose(1, 0, 2000, initial_step=0)
    choosed = policy(1,  # to return always 0 with random
                     lambda _: torch.tensor([[0, 1]]),  # to return always 1
                     None)
    assert(choosed.item() == 0)


def test_eps_end_0_choose_model():
    """
    Tests that if eps_start = 1 end eps_end = 0 at step=infinite
    calls model action
    """

    policy = ex.epsilon_greedy_choose(1, 0, 1, initial_step=2)
    choosed = policy(1,  # to return always 0 with random
                     lambda _: torch.tensor([[0, 1]]),  # to return always 1
                     None)
    assert(choosed.item() == 1)
