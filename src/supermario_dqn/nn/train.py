"""
training algorithms
"""

from supermario_dqn.env import MarioEnvironment
from supermario_dqn.nn.memory import RandomReplayMemory, Transition
from supermario_dqn.nn.model import DQN

import os
import datetime
import random
import math

import torch
from torch import optim
import torch.nn.functional as F


def train(policy_net: DQN, env: MarioEnvironment, memory=RandomReplayMemory(200000),
          batch_size=128, fit_interval=32, gamma=0.98, eps_start=0.9,
          eps_end=0.05, eps_decay=200, target_update=15, save_path='model.pt',
          save_interval=10, memory_size=200000, num_episodes=50,
          device='cpu', log_file_dir=None, verbose=1, log_postfix=''):
    """
    Handles training of network
    """

    n_actions = len(env.actions)
    assert(n_actions == policy_net._outputs)

    # switch to CPU or GPU
    policy_net.to(device)

    # compute target net and instances
    target_net = DQN(policy_net._channels, policy_net._height, policy_net._width, policy_net._outputs)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.to(device)
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters())

    if verbose > 0:
        if log_file_dir is not None:
            episode_log_file_path = os.path.join(log_file_dir, 'episodes.csv')
        else:
            episode_log_file_path = os.path.join('episodes.csv')

        if not os.path.isfile(episode_log_file_path):
            episode_log_file = open(episode_log_file_path, 'w')
            episode_log_file.write('episode,reward,steps,choosen_moves,random_moves\n')
        else:
            episode_log_file = open(episode_log_file_path, 'a')

    if log_file_dir is not None:
        fitting_log_file = open(os.path.join(log_file_dir, 'fitting_log' + log_postfix + '.csv'), 'w')
        fitting_log_file.write('episode,step,mean_error\n')
        qvalues_log_file = open(os.path.join(log_file_dir, 'qvalues_log', log_postfix + '.csv'), 'w')
        qvalues_log_file.write('episode,step,' + ','.join(["{}{}".format(a, b) for a, b in zip(['action'] * n_actions, range(n_actions))]) + ',choosen,reward') # noqa

    # for logs
    curr_episode = 0  # current episode
    steps_done = 0  # step in a single episode
    choosen_moves = 0  # number of choosen moves in a single episode
    random_moves = 0  # number of random moves in a single episode

    # select random action
    total_step = 0

    def select_action(state):
        nonlocal total_step, choosen_moves, random_moves
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * total_step / eps_decay)
        total_step += 1
        if sample > eps_threshold:
            choosen_moves += 1
            with torch.no_grad():
                qvalues = policy_net(state)
                return qvalues.max(1)[1].view(1, 1), qvalues
        else:
            random_moves += 1
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), None

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

        if log_file_dir is not None:
            fitting_log_file.write('{},{},{}\n'.format(curr_episode, total_step, loss.mean()))

    # training loop
    try:
        for i_episode in range(num_episodes):
            curr_episode = i_episode + 1
            steps_done = 0
            choosen_moves = 0
            random_moves = 0
            episode_reward = 0
            curr_state = env.reset()
            done = False
            while not done:
                steps_done += 1
                action, qvalues = select_action(curr_state.unsqueeze(0).to(device))
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

                # log on qvalues_file_log.csv
                if log_file_dir is not None and qvalues is not None:
                    qvalues_log_file.write('{},{},'.format(curr_episode, total_step)
                                           + ','.join(list(map(lambda t: str(t.item()), qvalues[0])))
                                           + ',{},{}\n'.format(action.item(), reward.item()))

            # log on episode.csv
            if verbose > 0:
                episode_log_file.write('{},{},{},{},{}\n'
                                       .format(i_episode+1, episode_reward, steps_done, choosen_moves, random_moves))

            # print
            if verbose > 0:
                print(f'[{datetime.datetime.now().strftime("%d:%m:%Y %H:%M")}] end episode ({i_episode+1}/{num_episodes}, world: {env.curr_world} stage: {env.curr_stage}, {steps_done} steps): {episode_reward} reward')  # noqa

            # Update the target network, copying all weights and biases in DQN
            if i_episode % target_update == 0:
                if verbose > 0:
                    print(f'[{datetime.datetime.now().strftime("%d:%m:%Y %H:%M")}] updating target network')
                target_net.load_state_dict(policy_net.state_dict())

            # Save on file
            if save_interval is not None and i_episode % save_interval == 0:
                if verbose > 0:
                    print(f'[{datetime.datetime.now().strftime("%d:%m:%Y %H:%M")}] saving model ({total_step} total steps done)')  # noqa
                torch.save(policy_net.state_dict(), save_path)

    finally:

        # close logs
        if verbose > 0:
            episode_log_file.close()

        if log_file_dir is not None:
            qvalues_log_file.close()
            fitting_log_file.close()

        # save network
        torch.save(policy_net.state_dict(), save_path)
