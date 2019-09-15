"""
Module containing many implementation of Replay Memory
"""

from collections import namedtuple, deque
import random


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class RandomReplayMemory(object):

    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition):
        """Saves a transition."""
        self.memory.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
