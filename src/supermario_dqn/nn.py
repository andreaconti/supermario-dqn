"""
Module containing utilities for training
"""

import math
import random
import typing
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn, optim

from supermario_dqn.environment import MarioEnvironment


__all__ = ['create', 'train']


# UTILS stuff

_Transition = namedtuple('_Transition',
                         ('state', 'action', 'next_state', 'reward'))


class _ReplayMemory(object):

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# COMPUTE model

class DQN(nn.Module):
    """
    class of the used Q-value Neural Network
    """

    def __init__(self, channels: int, height: int, width: int, outputs: int):
        super(DQN, self).__init__()

        # parameters
        self._outputs = outputs
        self._channels = channels
        self._height = height
        self._width = width

        # CNN
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Dense
        convw = DQN._conv2d_size_out(DQN._conv2d_size_out(DQN._conv2d_size_out(width)))
        convh = DQN._conv2d_size_out(DQN._conv2d_size_out(DQN._conv2d_size_out(height)))
        linear_input_size = convw * convh * 32
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs)

    def _conv2d_size_out(size: int, kernel_size: int = 5, stride: int = 2):
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.head(x)
        return x


# FUNCTIONS

def create(size: typing.List[int], outputs: int,
           load_state_from: str = None, for_train=False) -> DQN:
    """
    create model
    """
    if len(size) != 3 or size[0] < 0 or size[1] < 0 or size[2] < 0:
        raise ValueError(f'size must be positive: [channels, height, width]')

    dqn = DQN(size[0], size[1], size[2], outputs)

    if load_state_from is not None:
        dqn.load_state_dict(torch.load(load_state_from))

    if not for_train:
        dqn.eval()

    return dqn


def train(policy_net: DQN, env: MarioEnvironment, batch_size=128, fit_interval=32,
          gamma=0.999, eps_start=0.9, eps_end=0.05, eps_decay=200, target_update=10,
          save_path='model.pt', save_interval=10, memory_size=10000, num_episodes=50,
          device='cpu', verbose=1):
    """
    Handles training of network
    """

    n_actions = env.n_actions
    assert(n_actions == policy_net._outputs)

    # switch to CPU or GPU
    policy_net.to(device)

    # compute target net and instances
    target_net = DQN(policy_net._channels, policy_net._height, policy_net._width, policy_net._outputs)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.to(device)
    target_net.eval()
    memory = _ReplayMemory(memory_size)
    optimizer = optim.RMSprop(policy_net.parameters())

    # select random action
    steps_done = 0

    def select_action(state):
        nonlocal steps_done
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

    # perform a single optimization step
    def optimize_model(verbose=1):

        if len(memory) < batch_size:
            return

        transitions = memory.sample(batch_size)
        batch = _Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch.to(device))
        state_action_values = state_action_values.gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states.to(device)).max(1)[0].detach()
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        if verbose > 1:
            print(f"Fitting, loss: {loss.mean()}")

        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-15, 15)  # to avoid exploding gradient problem
        optimizer.step()

    # training loop
    for i_episode in range(num_episodes):
        episode_reward = 0
        curr_state = env.reset()
        done = False
        while not done:
            action = select_action(curr_state.unsqueeze(0).to(device))
            next_state, reward, done, _ = env.step(action.item())
            episode_reward += reward
            reward = torch.tensor([reward], device=device, dtype=torch.float32)

            if not done:
                memory.push(_Transition(curr_state, action, next_state.to(device), reward))
                curr_state = next_state
            else:
                memory.push(_Transition(curr_state, action, None, reward))

            # Perform one step of the optimization (on the target network)
            if steps_done % fit_interval == 0:
                optimize_model()

        if verbose > 0:
            print(f'end episode ({i_episode}/{num_episodes}): {episode_reward} reward')

        # Update the target network, copying all weights and biases in DQN
        if i_episode % target_update == 0:
            if verbose > 0:
                print('updating target network')
            target_net.load_state_dict(policy_net.state_dict())

        # Save on file
        if i_episode % save_interval == 0:
            if verbose > 0:
                print(f'saving model ({steps_done} steps done)')
            torch.save(policy_net.state_dict(), save_path)


def best_action(model: DQN, state: torch.Tensor) -> int:
    """
    provides best action for given state
    """
    return model(state).max(1)[1].view(1, 1).item()
