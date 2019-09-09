"""
Utilities for environment handling
"""

from collections import deque
from typing import Callable

import gym_super_mario_bros as gym
import torch
from nes_py.wrappers import JoypadSpace

__all__ = ['State', 'MarioEnvironment']

State = torch.Tensor


class MarioEnvironment():
    """
    Provides environment for SuperMario Bros
    """

    def __init__(self, n_frames: int, preprocess: Callable = None):

        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

        self._preprocess = preprocess
        self.n_frames = n_frames
        self.actions = SIMPLE_MOVEMENT
        self._env = JoypadSpace(gym.make('SuperMarioBros-v0'), self.actions)
        self.n_actions = self._env.action_space.n

    def reset(self, original=False):
        frame = self._env.reset()
        if self._preprocess is not None:
            frame_ = self._preprocess(frame)
        else:
            frame_ = frame

        self.frames = deque([frame_]*self.n_frames, self.n_frames)

        if not original:
            return torch.stack(tuple(self.frames))
        else:
            return torch.stack(tuple(self.frames)), frame

    def step(self, action: int, original=False):
        frame, reward, done, info = self._env.step(action)
        if self._preprocess is not None:
            frame_ = self._preprocess(frame)
        else:
            frame_ = frame
        self.frames.appendleft(frame_)

        if not original:
            return torch.stack(tuple(self.frames)), reward, done, info
        else:
            return [torch.stack(tuple(self.frames)), frame], reward, done, info
