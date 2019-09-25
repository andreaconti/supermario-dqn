"""
Creates models for training and usage
"""

import torch
from torch import nn
import torch.nn.functional as F
import typing


class DQN(nn.Module):
    """
    DQN Pytorch neural network.
    """

    def __init__(self, channels: int, height: int, width: int, outputs: int):
        """
        Initializes DQN network

        Args:
            channels: number of channels of the input image
            height: height of the input image
            width: width of the input image
            outputs: number of actions in output from the DQN model
        """

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
    creates a DQN neural network.

    Is possible to load from a file the state_dict or create a new
    network.

    Args:
        size: a sequence of 3 integers representing the input shape of the network,
           for instance [4, 30, 56] if the input is a tensor of shape 4x30x56
        outputs: integer, the number of outputs of the network
        load_state_from: path of a file containing a pytorch `state_dict`
        for_train: false by default, setup network for training or not

    Returns:
        A `DQN` class instance
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
    provides best action for given state.

    Args:
        model: a DQN class instance, or a Callable that returns
            a `torch.Tensor` of shape [1, n_actions].
        state: tensor used as input for the model with an added
            axis at index 0

    Returns:
        index of the action with the best score
    """
    return model(state.unsqueeze(0)).max(1)[1].view(1, 1).item()
