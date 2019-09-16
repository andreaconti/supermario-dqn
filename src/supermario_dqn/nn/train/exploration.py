"""
This module defines many exploration policies
"""

import random
import torch
import math

__ALL__ = ['epsilon_greedy_choose']


def epsilon_greedy_choose(eps_start, eps_end, eps_decay, initial_step=0):

    total_step = initial_step

    def select_action_(n_actions, model, state):
        nonlocal total_step
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * total_step / eps_decay)
        total_step += 1
        if sample > eps_threshold:
            with torch.no_grad():
                qvalues = model(state)
                return qvalues.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)

    return select_action_
