from . import beta
from copy import deepcopy
from math import pi, log
from scipy.integrate import solve_ivp
from wilson.run.smeft.beta import C_array2dict
from wilson.translate.smeft import arrays2wcxf
import numpy as np


def smeft_evolve_leadinglog(C_in, scale_in, scale_out, newphys=True):
    """Solve the SMEFT RGEs in the leading log approximation.

    Input C_in and output C_out are dictionaries of arrays."""
    C_out = deepcopy(C_in)
    b = beta.beta(C_in, newphys=newphys)
    for k, C in C_out.items():
        C_out[k] = C + b[k] / (16 * pi**2) * log(scale_out / scale_in)
    return C_out


def _smeft_evolve(C_in, scale_in, scale_out, newphys=True, **kwargs):
    """Axuliary function used in `smeft_evolve` and `smeft_evolve_continuous`"""
    def fun(t0, y):
        return beta.beta_array(C=beta.C_array2dict(y.view(complex)),
                               newphys=newphys).view(float) / (16 * pi**2)
    y0 = beta.C_dict2array(C_in).view(float)
    sol = solve_ivp(fun=fun,
                    t_span=(log(scale_in), log(scale_out)),
                    y0=y0, **kwargs)
    return sol


def smeft_evolve(C_in, scale_in, scale_out, newphys=True, **kwargs):
    """Solve the SMEFT RGEs by numeric integration.

    Input C_in and output C_out are dictionaries of arrays."""
    sol = _smeft_evolve(C_in, scale_in, scale_out, newphys=newphys, **kwargs)
    return beta.C_array2dict(sol.y[:, -1].view(complex))


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
        yd = C_array2dict(y)
        yw = arrays2wcxf(yd)
        return yw
    def rge_solution(scale):
        # this is to return a scalar if the input is scalar
        return _rge_solution(scale)[()]
    return rge_solution
