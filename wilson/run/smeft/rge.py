"""Solving the SMEFT RGEs."""


from . import beta
from copy import deepcopy
from math import pi, log
from scipy.integrate import solve_ivp
from wilson.run.smeft.beta import C_array2dict
from wilson.translate.smeft import arrays2wcxf
from wilson.util.smeftutil import C_symm_keys
import numpy as np


# computing the scale vector required for rescale_dict below
# initialize with factor 1
_d_4 = np.zeros((3,3,3,3))
_d_6 = np.zeros((3,3,3,3))
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                # class 4: symmetric under interachange of currents
                _d_4[i, j, k, l] = len(set([(i, j, k, l), (k, l, i, j)]))
                # class 4: symmetric under interachange of currents + Fierz
                _d_6[i, j, k, l] = len(set([(i, j, k, l), (k, l, i, j), (k, j, i, l), (i, l, k, j)]))


_scale_dict = C_array2dict(np.ones(9999))
for k in C_symm_keys[4]:
    _scale_dict[k] = _d_4
for k in C_symm_keys[6]:
    _scale_dict[k] = _d_6


def scale_dict(C):
    """To account for the fact that arXiv:1312.2014 uses a flavour
    non-redundant basis in contrast to WCxf, symmetry factors of two have to
    be introduced in several places for operators that are symmetric
    under the interchange of two currents."""
    return {k: _scale_dict[k] * v for k, v in C.items()}


def unscale_dict(C):
    """Undo the scaling applied in `scale_dict`."""
    return {k: 1 / _scale_dict[k] * v for k, v in C.items()}


def smeft_evolve_leadinglog(C_in, scale_in, scale_out, newphys=True):
    """Solve the SMEFT RGEs in the leading log approximation.

    Input C_in and output C_out are dictionaries of arrays."""
    C_out = scale_dict(deepcopy(C_in))
    b = beta.beta(C_out, newphys=newphys)
    for k, C in C_out.items():
        C_out[k] = C + b[k] / (16 * pi**2) * log(scale_out / scale_in)
    return unscale_dict(C_out)


def _smeft_evolve(C_in, scale_in, scale_out, newphys=True, **kwargs):
    """Axuliary function used in `smeft_evolve` and `smeft_evolve_continuous`"""
    def fun(t0, y):
        return beta.beta_array(C=beta.C_array2dict(y.view(complex)),
                               newphys=newphys).view(float) / (16 * pi**2)
    y0 = beta.C_dict2array(scale_dict(C_in)).view(float)
    sol = solve_ivp(fun=fun,
                    t_span=(log(scale_in), log(scale_out)),
                    y0=y0, **kwargs)
    return sol


def smeft_evolve(C_in, scale_in, scale_out, newphys=True, **kwargs):
    """Solve the SMEFT RGEs by numeric integration.

    Input C_in and output C_out are dictionaries of arrays."""
    sol = _smeft_evolve(C_in, scale_in, scale_out, newphys=newphys, **kwargs)
    return unscale_dict(beta.C_array2dict(sol.y[:, -1].view(complex)))


def smeft_evolve_continuous(C_in, scale_in, scale_out, newphys=True, **kwargs):
    """Solve the SMEFT RGEs by numeric integration, returning a function that
    allows to compute an interpolated solution at arbitrary intermediate
    scales."""
    sol = _smeft_evolve(C_in, scale_in, scale_out, newphys=newphys,
                        dense_output=True, **kwargs)
    @np.vectorize
    def _rge_solution(scale):
        t = log(scale)
        y = sol.sol(t).view(complex)
        yd = unscale_dict(C_array2dict(y))
        yw = arrays2wcxf(yd)
        return yw
    def rge_solution(scale):
        # this is to return a scalar if the input is scalar
        return _rge_solution(scale)[()]
    return rge_solution
