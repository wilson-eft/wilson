"""Solving the LEFT RGEs."""
from wilson.run.wet import beta
from copy import deepcopy
from math import pi, log
from scipy.integrate import solve_ivp
import numpy as np
#
#
def left_evolve_leadinglog(C_in, scale_in, scale_out, Nu,Nd):
    """Solve the LEFT RGEs in the leading log approximation.
    Input C_in and output C_out are dictionaries of arrays."""
    C_out = deepcopy(C_in)
    b = beta.beta(C_out, Nu, Nd)
    for k, C in C_in.items(): # look for loopholes C_in vs C_out
        C_out[k] = C + b[k] / (16 * pi**2) * log(scale_out / scale_in)
    return C_out




