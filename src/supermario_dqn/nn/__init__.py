"""
package managing neural network
"""

from .model import create, best_action
from .train import train
from .memory import RandomReplayMemory, Transition

__ALL__ = ['create', 'best_action', 'train', 'RandomReplayMemory', 'Transition']
