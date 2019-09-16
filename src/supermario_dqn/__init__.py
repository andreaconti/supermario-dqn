# -*- coding: utf-8 -*-
from pkg_resources import DistributionNotFound, get_distribution

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = 'supermario-dqn'
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = '1.0'
finally:
    del get_distribution, DistributionNotFound
