"""Utility modules for wilson."""

from wilson.util.common import EFTutil as EFTutil

smeftutil = EFTutil(
    'SMEFT',
    'Warsaw',
    dim4_keys_shape = {
       'g': 1,
       'gp': 1,
       'gs': 1,
       'Lambda': 1,
       'm2': 1,
       'Gu': (3, 3),
       'Gd': (3, 3),
       'Ge': (3, 3),
    },
    dim4_symm_keys =  {
        0: ['g', 'gp', 'gs', 'Lambda', 'm2'],
        1: ['Gu', 'Gd', 'Ge'],
    },
)
wetutil = EFTutil(
    'WET',
    'JMS',
    dim4_keys_shape = {
        'e': 1,
        'gs': 1,
        'Md': (3, 3),
        'Mu': (2, 2),
        'Me': (3, 3),
        'Mnu': (3, 3),
    },
    dim4_symm_keys = {
        0: ['e', 'gs'],
        1: ['Md', 'Mu', 'Me', 'Mnu'],
    },
)
