"""
Provides exploration policy functions
"""

import random
import torch
import math

__ALL__ = ['epsilon_greedy_choose']


def epsilon_greedy_choose(eps_start, eps_end, eps_decay, initial_step=0):
    """
    The action is selected randomly with probability 'eps' otherwise is choosen using the neural network.
    eps decreases using an exponential law until eps_end.

    Args:
        eps_start: initial value of eps.
        eps_end: end value of eps.
        eps_decay: number of steps between eps_start and eps_end
        initial_step: the initial step to be used, useful to restore a training.

    Returns:
        A Callables with as inputs the number of actions, the model and the state on which
        choose action. This callable returns a integer, the choosen action.
    """

    total_step = initial_step

    def select_action_(n_actions, model, state):
        nonlocal total_step
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-5. * total_step / eps_decay)
        total_step += 1
        if sample < eps_threshold:
            return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)
        else:
            with torch.no_grad():
                qvalues = model(state)
                return qvalues.max(1)[1].view(1, 1)

    return select_action_
