"""
Utilities for environment handling
"""

from collections import deque
from typing import Callable

import gym_super_mario_bros as gym
import torch
from nes_py.wrappers import JoypadSpace
import random as r

__all__ = ['State', 'MarioEnvironment']

State = torch.Tensor


# Definition of actions spaces
SIMPLE_ACTIONS = [['right'], ['right', 'A', 'B'], ['right', 'A'], ['left']]


class MarioEnvironment():
    """
    Provides environment for SuperMario Bros
    """

    def __init__(self, actions, n_frames: int, preprocess: Callable, random=False, world_stage: (int, int) = None,
                 render: bool = False):

        self._render = render

        # compute world
        world_name = 'SuperMarioBros-v0'
        self._world = 1
        self._stage = 1
        self._selected = None
        if world_stage is not None and not random:
            assert(world_stage[0] in range(1, 9))
            assert(world_stage[1] in range(1, 5))
            self._world, self._stage = world_stage
            self._selected = world_stage
            world_name = 'SuperMarioBros-{}-{}-v0'.format(self._world, self._stage)

        self._random = random
        if random:
            self._world = r.sample(range(1, 9), 1)[0]
            self._stage = r.sample(range(1, 5), 1)[0]
            world_name = f'SuperMarioBros-{self._world}-{self._stage}-v0'

        self._preprocess = preprocess
        self.n_frames = n_frames
        self._actions = actions

        self._env_actions = actions
        if ['NOOP'] not in self._env_actions:
            self._env_actions = self._env_actions + [['NOOP']]
        self._env = JoypadSpace(gym.make(world_name), self._env_actions)

    @property
    def actions(self):
        return self._actions

    @property
    def curr_world(self):
        return self._world

    @property
    def curr_stage(self):
        return self._stage

    def reset(self, original=False):

        if not self._random:
            frame = self._env.reset()
            if self._selected is not None:
                self._world, self._stage = self._selected
            else:
                self._world = 1
                self._stage = 1
        else:
            self._world = r.sample(range(1, 9), 1)[0]
            self._stage = r.sample(range(1, 5), 1)[0]
            self._env = JoypadSpace(gym.make('SuperMarioBros-{}-{}-v0'.format(self._world, self._stage)), self._env_actions)  # noqa
            frame = self._env.reset()

        if self._render:
            self._env.render()

        frame_ = self._preprocess(self._world, self._stage, frame)

        self.frames = deque([frame_]*self.n_frames, self.n_frames)
        self._last_x_pos = 0
        self._last_y_pos = 0

        if not original:
            return torch.stack(tuple(self.frames))
        else:
            return torch.stack(tuple(self.frames)), frame

    def step(self, action: int, original: bool = False, apply_times: int = 3):

        noop_action: int = self._env_actions.index(['NOOP'])
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
            self._world = info['world']
            self._stage = info['stage']

            if self._render:
                self._env.render()

            if original:
                original_frames.append(frame)

            if done:
                break

        # apply noop action
        if not done and 'A' in self.actions[action] and self._last_y_pos > last_y_pos:
            frame, reward_, done, info = self._env.step(noop_action)
            original_frames.append(frame)
            reward += reward_
            self._world = info['world']
            self._stage = info['stage']

            if self._render:
                self._env.render()

        self._last_y_pos = last_y_pos

        # preprocess reward
        if 'right' in self.actions[action] and 'A' not in self.actions[action] and last_x_pos == self._last_x_pos:
            reward = -4
        self._last_x_pos = last_x_pos

        # preprocess image
        if self._preprocess is not None:
            frame_ = self._preprocess(self._world, self._stage, frame)
        else:
            frame_ = frame
        self.frames.appendleft(frame_)

        # check done
        if self._selected is not None and (self._selected[0] != self._world or self._selected[1] != self._stage):
            done = True

        # return
        if not original:
            return torch.stack(tuple(self.frames)), reward, done, info
        else:
            return [torch.stack(tuple(self.frames)), list(original_frames)], reward, done, info
