"""
Module containing many implementation of Replay Memory
"""

from collections import namedtuple, deque
import random

__ALL__ = ['RandomReplayMemory']

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class RandomReplayMemory:
    """
    Replay Memory with random sampling.
    """

    def __init__(self, capacity: int):
        """
        Args:
            capacity: max number of examples in the buffer, once
                exceeded the size older examples are dropped.
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, transition):
        """Saves a transition."""
        self.memory.append(transition)

    def sample(self, batch_size: int):
        """
        Returns:
            a list of Transitions of the given batch_size
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
