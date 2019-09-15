"""
Creates models for training and usage
"""

import torch
from torch import nn
import torch.nn.functional as F
import typing


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
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=6, stride=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Dense
        convw = DQN._conv2d_size_out(DQN._conv2d_size_out(DQN._conv2d_size_out(width, kernel_size=6, stride=3),
                                     kernel_size=4, stride=2), kernel_size=3, stride=1)
        convh = DQN._conv2d_size_out(DQN._conv2d_size_out(DQN._conv2d_size_out(height, kernel_size=6, stride=3),
                                     kernel_size=4, stride=2), kernel_size=3, stride=1)
        linear_input_size = convw * convh * 64
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs)

    def _conv2d_size_out(size: int, kernel_size: int = 5, stride: int = 2):
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.fc1(x.view(x.size(0), -1)))
        x = F.softsign(self.head(x)) * 15.
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
        if torch.cuda.is_available():
            dqn.load_state_dict(torch.load(load_state_from))
        else:
            dqn.load_state_dict(torch.load(load_state_from, map_location=torch.device('cpu')))

    if not for_train:
        dqn.eval()

    return dqn


def best_action(model: DQN, state: torch.Tensor) -> int:
    """
    provides best action for given state
    """
    return model(state).max(1)[1].view(1, 1).item()
