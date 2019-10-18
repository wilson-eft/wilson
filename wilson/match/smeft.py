"""Matcher from the SMEFT 'Warsaw up' basis to the WET JMS basis.

Based on arXiv:1908.05295."""


import numpy as np
from math import sqrt, pi
import wcxf
import wilson
from wilson.run.smeft.smpar import p as default_parameters
from wilson.util import smeftutil, wetutil
from wilson.match import smeft_tree, smeft_loop


def match_all(d_SMEFT, scale, parameters=None):
    """Match the SMEFT Warsaw basis onto the WET JMS basis.
    
    The optional `parameters` dictionary allows to overwrite the default
    numerical input parameters (such as CKM elements and quark masses).
    Moreover, there is a key `'loop_order'` which, if set to 1, allows
    to switch on the one-loop matching contributions (which are)
    omitted by default.
    """
    p = default_parameters.copy()
    if parameters is not None:
        # if parameters are passed in, overwrite the default values
        p.update(parameters)
    C = wilson.util.smeftutil.wcxf2arrays_symmetrized(d_SMEFT)
    C_WET_tree = smeft_tree.match_all_array(C, p)
    if parameters and parameters.get('loop_order') == 1:
        # One loop matching only added if 'loop_order' is 1!
        C_WET_loop = smeft_loop.match_all_array(C, p, scale=scale)
        C_WET = {k: np.array(C_WET_tree[k] + C_WET_loop[k], complex) for k in C_WET_tree}
    else:
        C_WET = C_WET_tree
    C_WET = wilson.translate.wet.rotate_down(C_WET, p)
    C_WET = wetutil.unscale_dict_wet(C_WET)
    d_WET = wilson.util.smeftutil.arrays2wcxf(C_WET)
    basis = wcxf.Basis['WET', 'JMS']
    keys = set(d_WET.keys()) & set(basis.all_wcs)
    d_WET = {k: d_WET[k] for k in keys}
    return d_WET
