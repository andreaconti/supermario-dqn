"""
Play a game loading a model.pt file
"""

from supermario_dqn.environment import MarioEnvironment
from supermario_dqn.preprocess import preprocess
import supermario_dqn.nn as nn
import matplotlib.pyplot as plt
import argparse


def main(model=None, world_stage=None):

    if model is None:
        parser = argparse.ArgumentParser('play a game')
        parser.add_argument('model', type=str, help='neural network model')
        parser.add_argument('--world_stage', type=int, nargs=2, default=None,
                            help='select a specific world and stage, world in [1..8], stage in [1..4]')
        args = vars(parser.parse_args())

        env = MarioEnvironment(4, lambda w, s, t: preprocess(w, s, t, 30, 56), world_stage=args['world_stage'])
        model = nn.create([4, 30, 56], env.n_actions, load_state_from=args['model'])
        model.requires_grad_(False)
    else:
        env = MarioEnvironment(4, lambda w, s, t: preprocess(w, s, t, 30, 56), world_stage=world_stage)

    # play loop
    done = False
    step = 0
    reward = 0
    [pr_state, _] = env.reset(original=True)
    plt.figure(1)
    while not done:
        plt.clf()
        step += 1

        action = nn.best_action(model, pr_state.unsqueeze(0))
        [pr_state, or_states], r, done, _ = env.step(action, original=True)
        reward += r

        for state in or_states:
            plt.imshow(state)
            plt.xlabel(f'reward {reward}')
            plt.pause(0.05)

    print(f'final reward: {reward}')
    print(f'number of steps: {step}')
