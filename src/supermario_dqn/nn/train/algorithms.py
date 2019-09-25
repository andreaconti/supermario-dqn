"""
This module provides function to train DQN networks.
"""

from supermario_dqn.env import MarioEnvironment
from supermario_dqn.nn.train.memory import RandomReplayMemory, Transition
from supermario_dqn.nn.train.exploration import epsilon_greedy_choose

import torch
from torch import optim
import torch.nn.functional as F


__ALL__ = ['train_dqn']


def train_dqn(policy_net: torch.nn.Module, target_net: torch.nn.Module, env: MarioEnvironment, memory=RandomReplayMemory(200000),  # noqa
              action_policy=epsilon_greedy_choose(0.9, 0.1, 50000), batch_size=128, fit_interval=32,
              gamma=0.98, target_update=30, optimizer_f=optim.Adam, optimizer_state_dict=None, num_episodes=50,
              device='cpu', callbacks=[]):
    """
    Training of a DQN network.

    Args:
        policy_net: a pytorch Model able to be feeded with observations in output by `env` and whose
            output is a 2D torch.Tensor with a line for each observation provided a row for each action
            available in the env.
        target_net: instance of target_net, must be an instance of the same class of policy_net
        env: OpenAI gym environment, specifically must provide `step`, `reset` functions and also
            `env.action_space.n` which must be equal to the number of rows in output by the network.
        memory: a custom ReplayMemory.
        action_policy: used action_policy.
        batch_size: batch_size used by memory.
        fit_interval: number of steps between each training phase.
        gamma: discount ratio.
        target_update: number of episodes between each update of the target network.
        optimizer_f: optimizer used during training as provided by pytorch.
        optimizer_state_dict: useful in order to restore the optimizer.
        num_episodes: num of episodes of training.
        device: device used, cpu by default.
        callbacks: a list of callbacks as provided in callbacks module.
    """

    n_actions = len(env.actions)
    assert(n_actions == policy_net._outputs)

    # active callbacks
    callbacks_ = {}
    for callback in callbacks:
        args = callback('init', None, None)
        callbacks_[callback] = args

    # switch to CPU or GPU
    policy_net.to(device)

    # compute target net and instances
    target_net.load_state_dict(policy_net.state_dict())
    target_net.to(device)
    target_net.eval()

    optimizer = optimizer_f(policy_net.parameters())
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    # for logs
    curr_episode = 0  # current episode
    steps_done = 0  # step in a single episode
    total_steps = 0  # total steps performed

    # perform a single optimization step
    def optimize_model():

        if len(memory) < batch_size:
            return

        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # training loop
    policy_net.train(False)
    for i_episode in range(num_episodes):
        curr_episode = i_episode + 1
        steps_done = 0
        episode_reward = 0
        curr_state = env.reset()
        done = False

        while not done:
            steps_done += 1
            total_steps += 1
            action = action_policy(n_actions, policy_net, curr_state.unsqueeze(0).to(device)).to(device)
            next_state, reward, done, _ = env.step(action.item())
            episode_reward += reward
            reward = torch.tensor([reward], device=device, dtype=torch.float32)

            if not done:
                memory.push(Transition(curr_state, action, next_state, reward))
                curr_state = next_state
            else:
                memory.push(Transition(curr_state, action, None, reward))

            # Perform one step of the optimization (on the target network)
            if steps_done % fit_interval == 0:
                optimize_model()

        # copy to target
        if i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()

        # call callbacks
        for callback, args in callbacks_.items():
            new_state = callback('run', args, {
                'model': policy_net,
                'optimizer': optimizer,
                'episode': curr_episode,
                'episodes': num_episodes,
                'reward': episode_reward,
                'steps': steps_done,
                'total_steps': total_steps,
                'device': device,
            })
            callbacks_[callback] = new_state

    for callback, args in callbacks_.items():
        callback('close', args, None)
