"""Utility modules for wilson."""

from wilson.util.common import EFTutil as _EFTutil
from . import smeft_warsaw

smeftutil = _EFTutil(
    'SMEFT',
    'Warsaw',
    smeft_warsaw.dim4_keys_shape,
    smeft_warsaw.C_symm_keys,
)
