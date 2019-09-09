"""
Utilities for frames preprocessing
"""

from torchvision import transforms

__all__ = ['preprocess']


def preprocess(tensor, resize_h, resize_w):
    """
    Preprocess a frame deleting background and highlighting
    Mario
    """
    mario_dress = 240  # , 56, 0]
    mario_skin = 252  # , 16, 68]
    mario_other = 172  # , 140, 0]
    background = 104

    state_ = tensor[70:208, :, 0]  # crop and red signal
    state_[state_ == mario_dress] = 255
    state_[state_ == mario_skin] = 255
    state_[state_ == mario_other] = 255
    state_[state_ == background] = 0

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([resize_h, resize_w]),
        transforms.ToTensor()
    ])(state_)[0]
