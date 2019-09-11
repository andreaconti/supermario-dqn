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

    def __init__(self, n_frames: int, preprocess: Callable, random=False, world_stage: (int, int) = None):

        # compute world
        world = 'SuperMarioBros-v0'
        self._stage = None
        if world_stage is not None:
            assert(world_stage[0] in range(1, 9))
            assert(world_stage[1] in range(1, 5))
            self._stage = world_stage[1]
            world = 'SuperMarioBros-{}-{}-v0'.format(world_stage[0], world_stage[1])

        if random:
            world = 'SuperMarioBrosRandomStages-v0'
            self._stage = None

        self._preprocess = preprocess
        self.n_frames = n_frames
        self.actions = [['right'], ['right', 'A', 'B'], ['right', 'A'], ['left']]
        self._env = JoypadSpace(gym.make(world), self.actions + [['NOOP']])
        self.n_actions = len(self.actions)

    def reset(self, original=False):
        frame = self._env.reset()
        frame_ = self._preprocess(frame)

        self.frames = deque([frame_]*self.n_frames, self.n_frames)
        self._last_x_pos = 0
        self._last_y_pos = 0

        if not original:
            return torch.stack(tuple(self.frames))
        else:
            return torch.stack(tuple(self.frames)), frame

    def step(self, action: int, original: bool = False, apply_times: int = 3):

        noop_action: int = self.n_actions
        last_x_pos: int = 0
        last_y_pos: int = 0
        reward = 0
        original_frames = []

        # apply
        for i in range(apply_times):

            frame, reward_, done, info = self._env.step(action)
            last_x_pos = info['x_pos']
            last_y_pos = info['y_pos']
            reward += reward_

            if original:
                original_frames.append(frame)

            if self._stage is not None and self._stage < info['stage']:
                done = True

            if done:
                break

        # apply noop action
        if not done and 'A' in self.actions[action] and self._last_y_pos > last_y_pos:
            frame, reward_, done, info = self._env.step(noop_action)
            original_frames.append(frame)
            reward += reward_
        self._last_y_pos = last_y_pos

        # preprocess reward
        if 'right' in self.actions[action] and 'A' not in self.actions[action] and last_x_pos == self._last_x_pos:
            reward = -4
        self._last_x_pos = last_x_pos

        # preprocess image
        if self._preprocess is not None:
            frame_ = self._preprocess(frame)
        else:
            frame_ = frame
        self.frames.appendleft(frame_)

        # return
        if not original:
            return torch.stack(tuple(self.frames)), reward, done, info
        else:
            return [torch.stack(tuple(self.frames)), list(original_frames)], reward, done, info
