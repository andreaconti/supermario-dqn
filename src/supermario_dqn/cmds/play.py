"""
Play a game loading a model.pt file
"""

from supermario_dqn.environment import MarioEnvironment
from supermario_dqn.preprocess import preprocess
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import supermario_dqn.nn as nn
import matplotlib.pyplot as plt
import argparse


def main(model=None):

    env = MarioEnvironment(4, lambda t: preprocess(t, 30, 56))

    if model is None:
        parser = argparse.ArgumentParser('play a game')
        parser.add_argument('model', type=str, help='neural network model')
        args = vars(parser.parse_args())

        model = nn.create([4, 30, 56], env.n_actions, load_state_from=args['model'])
        model.requires_grad_(False)

    # play loop
    done = False
    step = 0
    reward = 0
    [pr_state, or_state] = env.reset(original=True)
    plt.figure(1)
    while not done:
        plt.clf()
        step += 1

        action = nn.best_action(model, pr_state.unsqueeze(0))
        [pr_state, or_state], r, done, _ = env.step(action, original=True)
        reward += r

        plt.imshow(or_state)
        plt.title(f'action {SIMPLE_MOVEMENT[action]}, reward {reward}')
        plt.pause(0.05)

    print(f'final reward: {reward}')
    print(f'number of steps: {step}')
