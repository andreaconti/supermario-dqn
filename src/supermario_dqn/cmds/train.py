"""
Handles supermario training
"""

import torch

if torch.cuda.is_available():
    _device = torch.device('cuda')
else:
    _device = torch.device('cpu')
    print('[Warning] using CPU for training')

parameters = {
    'batch_size': 128,
    'gamma': 0.999,
    'eps_start': 0.9,
    'eps_end': 0.05,
    'target_update': 10,
    'save_path': 'model.pt',
    'save_interval': 10,
    'memory_size': 10000,
    'num_episodes': 50,
    'device': _device,
    'verbose': 1,
}

def main():
    print('not yet implemented')
