"""Utility functions and dictionaries useful for the manipulation of SMEFT Wilson coefficients.
"""

import numpy as np
from collections import OrderedDict
from functools import reduce
import operator


# names of SM parameters
SM_keys = ['g', 'gp', 'gs', 'Lambda', 'm2', 'Gu', 'Gd', 'Ge',]


# names of WCs with 0, 2, or 4 fermions (i.e. scalars, 3x3 matrices,
# and 3x3x3x3 tensors)
WC_keys_0f = ["G", "Gtilde", "W", "Wtilde", "phi", "phiBox", "phiD", "phiG",
              "phiB", "phiW", "phiWB", "phiGtilde", "phiBtilde", "phiWtilde",
              "phiWtildeB"]
WC_keys_2f = ["uphi", "dphi", "ephi", "eW", "eB", "uG", "uW", "uB", "dG", "dW",
              "dB", "phil1", "phil3", "phie", "phiq1", "phiq3", "phiu", "phid",
              "phiud", "llphiphi"]
WC_keys_4f = ["ll", "qq1", "qq3", "lq1", "lq3", "ee", "uu", "dd", "eu", "ed",
              "ud1", "ud8", "le", "lu", "ld",  "qe", "qu1", "qd1", "qu8",
              "qd8", "ledq", "quqd1", "quqd8", "lequ1", "lequ3", "duql",
              "qque", "qqql", "duue"]


C_keys = SM_keys + WC_keys_0f + WC_keys_2f + WC_keys_4f
WC_keys = WC_keys_0f + WC_keys_2f + WC_keys_4f

C_keys_shape = {
   'g': 1,
   'gp': 1,
   'gs': 1,
   'Lambda': 1,
   'm2': 1,
   'Gu': (3, 3),
   'Gd': (3, 3),
   'Ge': (3, 3),
   'G': 1,
   'Gtilde': 1,
   'W': 1,
   'Wtilde': 1,
   'phi': 1,
   'phiBox': 1,
   'phiD': 1,
   'phiG': 1,
   'phiB': 1,
   'phiW': 1,
   'phiWB': 1,
   'phiGtilde': 1,
   'phiBtilde': 1,
   'phiWtilde': 1,
   'phiWtildeB': 1,
   'uphi': (3, 3),
   'dphi': (3, 3),
   'ephi': (3, 3),
   'eW': (3, 3),
   'eB': (3, 3),
   'uG': (3, 3),
   'uW': (3, 3),
   'uB': (3, 3),
   'dG': (3, 3),
   'dW': (3, 3),
   'dB': (3, 3),
   'phil1': (3, 3),
   'phil3': (3, 3),
   'phie': (3, 3),
   'phiq1': (3, 3),
   'phiq3': (3, 3),
   'phiu': (3, 3),
   'phid': (3, 3),
   'phiud': (3, 3),
   'llphiphi': (3, 3),
   'll': (3, 3, 3, 3),
   'qq1': (3, 3, 3, 3),
   'qq3': (3, 3, 3, 3),
   'lq1': (3, 3, 3, 3),
   'lq3': (3, 3, 3, 3),
   'ee': (3, 3, 3, 3),
   'uu': (3, 3, 3, 3),
   'dd': (3, 3, 3, 3),
   'eu': (3, 3, 3, 3),
   'ed': (3, 3, 3, 3),
   'ud1': (3, 3, 3, 3),
   'ud8': (3, 3, 3, 3),
   'le': (3, 3, 3, 3),
   'lu': (3, 3, 3, 3),
   'ld': (3, 3, 3, 3),
   'qe': (3, 3, 3, 3),
   'qu1': (3, 3, 3, 3),
   'qd1': (3, 3, 3, 3),
   'qu8': (3, 3, 3, 3),
   'qd8': (3, 3, 3, 3),
   'ledq': (3, 3, 3, 3),
   'quqd1': (3, 3, 3, 3),
   'quqd8': (3, 3, 3, 3),
   'lequ1': (3, 3, 3, 3),
   'lequ3': (3, 3, 3, 3),
   'duql': (3, 3, 3, 3),
   'qque': (3, 3, 3, 3),
   'qqql': (3, 3, 3, 3),
   'duue': (3, 3, 3, 3),
}

# names of Wilson coefficients with the same fermionic symmetry properties
C_symm_keys = {}
# 0 0F scalar object
C_symm_keys[0] = WC_keys_0f + ['g', 'gp', 'gs', 'Lambda', 'm2',]
# 1 2F general 3x3 matrix
C_symm_keys[1] = ["uphi", "dphi", "ephi", "eW", "eB", "uG", "uW", "uB", "dG", "dW", "dB", "phiud"] + ['Gu', 'Gd', 'Ge']
# 2 2F Hermitian matrix
C_symm_keys[2] = ["phil1", "phil3", "phie", "phiq1", "phiq3", "phiu", "phid",]
# 3 4F general 3x3x3x3 object
C_symm_keys[3] = ["ledq", "quqd1", "quqd8", "lequ1", "lequ3", "duql", "duue"]
# 4 4F two identical ffbar currents
C_symm_keys[4] = ["ll", "qq1", "qq3", "uu", "dd",]
# 5 4F two independent ffbar currents
C_symm_keys[5] = ["lq1", "lq3", "eu", "ed", "ud1", "ud8", "le", "lu", "ld", "qe", "qu1", "qd1", "qu8", "qd8",]
# 6 4F two identical ffbar currents - special case Cee
C_symm_keys[6] = ["ee",]
# 7 4F Baryon-number-violating - special case Cqque
C_symm_keys[7] = ["qque",]
# 8 4F Baryon-number-violating - special case Cqqql
C_symm_keys[8] = ["qqql",]
# 9 2F symmetric matrix
C_symm_keys[9] = ["llphiphi"]


def symmetrize_2(b):
    a = np.array(b, copy=True, dtype=complex)
    a[1, 0] = a[0, 1].conj()
    a[2, 0] = a[0, 2].conj()
    a[2, 1] = a[1, 2].conj()
    a.imag[0, 0] = 0
    a.imag[1, 1] = 0
    a.imag[2, 2] = 0
    return a

def symmetrize_4(b):
    a = np.array(b, copy=True, dtype=complex)
    a.real[0, 0, 1, 0] = a.real[0, 0, 0, 1]
    a.real[0, 0, 2, 0] = a.real[0, 0, 0, 2]
    a.real[0, 0, 2, 1] = a.real[0, 0, 1, 2]
    a.real[0, 1, 0, 0] = a.real[0, 0, 0, 1]
    a.real[0, 2, 0, 0] = a.real[0, 0, 0, 2]
    a.real[0, 2, 0, 1] = a.real[0, 1, 0, 2]
    a.real[0, 2, 1, 0] = a.real[0, 1, 2, 0]
    a.real[1, 0, 0, 0] = a.real[0, 0, 0, 1]
    a.real[1, 0, 0, 1] = a.real[0, 1, 1, 0]
    a.real[1, 0, 0, 2] = a.real[0, 1, 2, 0]
    a.real[1, 0, 1, 0] = a.real[0, 1, 0, 1]
    a.real[1, 0, 1, 1] = a.real[0, 1, 1, 1]
    a.real[1, 0, 1, 2] = a.real[0, 1, 2, 1]
    a.real[1, 0, 2, 0] = a.real[0, 1, 0, 2]
    a.real[1, 0, 2, 1] = a.real[0, 1, 1, 2]
    a.real[1, 0, 2, 2] = a.real[0, 1, 2, 2]
    a.real[1, 1, 0, 0] = a.real[0, 0, 1, 1]
    a.real[1, 1, 0, 1] = a.real[0, 1, 1, 1]
    a.real[1, 1, 0, 2] = a.real[0, 2, 1, 1]
    a.real[1, 1, 1, 0] = a.real[0, 1, 1, 1]
    a.real[1, 1, 2, 0] = a.real[0, 2, 1, 1]
    a.real[1, 1, 2, 1] = a.real[1, 1, 1, 2]
    a.real[1, 2, 0, 0] = a.real[0, 0, 1, 2]
    a.real[1, 2, 0, 1] = a.real[0, 1, 1, 2]
    a.real[1, 2, 0, 2] = a.real[0, 2, 1, 2]
    a.real[1, 2, 1, 0] = a.real[0, 1, 2, 1]
    a.real[1, 2, 1, 1] = a.real[1, 1, 1, 2]
    a.real[1, 2, 2, 0] = a.real[0, 2, 2, 1]
    a.real[2, 0, 0, 0] = a.real[0, 0, 0, 2]
    a.real[2, 0, 0, 1] = a.real[0, 1, 2, 0]
    a.real[2, 0, 0, 2] = a.real[0, 2, 2, 0]
    a.real[2, 0, 1, 0] = a.real[0, 1, 0, 2]
    a.real[2, 0, 1, 1] = a.real[0, 2, 1, 1]
    a.real[2, 0, 1, 2] = a.real[0, 2, 2, 1]
    a.real[2, 0, 2, 0] = a.real[0, 2, 0, 2]
    a.real[2, 0, 2, 1] = a.real[0, 2, 1, 2]
    a.real[2, 0, 2, 2] = a.real[0, 2, 2, 2]
    a.real[2, 1, 0, 0] = a.real[0, 0, 1, 2]
    a.real[2, 1, 0, 1] = a.real[0, 1, 2, 1]
    a.real[2, 1, 0, 2] = a.real[0, 2, 2, 1]
    a.real[2, 1, 1, 0] = a.real[0, 1, 1, 2]
    a.real[2, 1, 1, 1] = a.real[1, 1, 1, 2]
    a.real[2, 1, 1, 2] = a.real[1, 2, 2, 1]
    a.real[2, 1, 2, 0] = a.real[0, 2, 1, 2]
    a.real[2, 1, 2, 1] = a.real[1, 2, 1, 2]
    a.real[2, 1, 2, 2] = a.real[1, 2, 2, 2]
    a.real[2, 2, 0, 0] = a.real[0, 0, 2, 2]
    a.real[2, 2, 0, 1] = a.real[0, 1, 2, 2]
    a.real[2, 2, 0, 2] = a.real[0, 2, 2, 2]
    a.real[2, 2, 1, 0] = a.real[0, 1, 2, 2]
    a.real[2, 2, 1, 1] = a.real[1, 1, 2, 2]
    a.real[2, 2, 1, 2] = a.real[1, 2, 2, 2]
    a.real[2, 2, 2, 0] = a.real[0, 2, 2, 2]
    a.real[2, 2, 2, 1] = a.real[1, 2, 2, 2]
    a.imag[0, 0, 0, 0] = 0
    a.imag[0, 0, 1, 0] = -a.imag[0, 0, 0, 1]
    a.imag[0, 0, 1, 1] = 0
    a.imag[0, 0, 2, 0] = -a.imag[0, 0, 0, 2]
    a.imag[0, 0, 2, 1] = -a.imag[0, 0, 1, 2]
    a.imag[0, 0, 2, 2] = 0
    a.imag[0, 1, 0, 0] = a.imag[0, 0, 0, 1]
    a.imag[0, 1, 1, 0] = 0
    a.imag[0, 2, 0, 0] = a.imag[0, 0, 0, 2]
    a.imag[0, 2, 0, 1] = a.imag[0, 1, 0, 2]
    a.imag[0, 2, 1, 0] = -a.imag[0, 1, 2, 0]
    a.imag[0, 2, 2, 0] = 0
    a.imag[1, 0, 0, 0] = -a.imag[0, 0, 0, 1]
    a.imag[1, 0, 0, 1] = 0
    a.imag[1, 0, 0, 2] = -a.imag[0, 1, 2, 0]
    a.imag[1, 0, 1, 0] = -a.imag[0, 1, 0, 1]
    a.imag[1, 0, 1, 1] = -a.imag[0, 1, 1, 1]
    a.imag[1, 0, 1, 2] = -a.imag[0, 1, 2, 1]
    a.imag[1, 0, 2, 0] = -a.imag[0, 1, 0, 2]
    a.imag[1, 0, 2, 1] = -a.imag[0, 1, 1, 2]
    a.imag[1, 0, 2, 2] = -a.imag[0, 1, 2, 2]
    a.imag[1, 1, 0, 0] = 0
    a.imag[1, 1, 0, 1] = a.imag[0, 1, 1, 1]
    a.imag[1, 1, 0, 2] = a.imag[0, 2, 1, 1]
    a.imag[1, 1, 1, 0] = -a.imag[0, 1, 1, 1]
    a.imag[1, 1, 1, 1] = 0
    a.imag[1, 1, 2, 0] = -a.imag[0, 2, 1, 1]
    a.imag[1, 1, 2, 1] = -a.imag[1, 1, 1, 2]
    a.imag[1, 1, 2, 2] = 0
    a.imag[1, 2, 0, 0] = a.imag[0, 0, 1, 2]
    a.imag[1, 2, 0, 1] = a.imag[0, 1, 1, 2]
    a.imag[1, 2, 0, 2] = a.imag[0, 2, 1, 2]
    a.imag[1, 2, 1, 0] = -a.imag[0, 1, 2, 1]
    a.imag[1, 2, 1, 1] = a.imag[1, 1, 1, 2]
    a.imag[1, 2, 2, 0] = -a.imag[0, 2, 2, 1]
    a.imag[1, 2, 2, 1] = 0
    a.imag[2, 0, 0, 0] = -a.imag[0, 0, 0, 2]
    a.imag[2, 0, 0, 1] = a.imag[0, 1, 2, 0]
    a.imag[2, 0, 0, 2] = 0
    a.imag[2, 0, 1, 0] = -a.imag[0, 1, 0, 2]
    a.imag[2, 0, 1, 1] = -a.imag[0, 2, 1, 1]
    a.imag[2, 0, 1, 2] = -a.imag[0, 2, 2, 1]
    a.imag[2, 0, 2, 0] = -a.imag[0, 2, 0, 2]
    a.imag[2, 0, 2, 1] = -a.imag[0, 2, 1, 2]
    a.imag[2, 0, 2, 2] = -a.imag[0, 2, 2, 2]
    a.imag[2, 1, 0, 0] = -a.imag[0, 0, 1, 2]
    a.imag[2, 1, 0, 1] = a.imag[0, 1, 2, 1]
    a.imag[2, 1, 0, 2] = a.imag[0, 2, 2, 1]
    a.imag[2, 1, 1, 0] = -a.imag[0, 1, 1, 2]
    a.imag[2, 1, 1, 1] = -a.imag[1, 1, 1, 2]
    a.imag[2, 1, 1, 2] = 0
    a.imag[2, 1, 2, 0] = -a.imag[0, 2, 1, 2]
    a.imag[2, 1, 2, 1] = -a.imag[1, 2, 1, 2]
    a.imag[2, 1, 2, 2] = -a.imag[1, 2, 2, 2]
    a.imag[2, 2, 0, 0] = 0
    a.imag[2, 2, 0, 1] = a.imag[0, 1, 2, 2]
    a.imag[2, 2, 0, 2] = a.imag[0, 2, 2, 2]
    a.imag[2, 2, 1, 0] = -a.imag[0, 1, 2, 2]
    a.imag[2, 2, 1, 1] = 0
    a.imag[2, 2, 1, 2] = a.imag[1, 2, 2, 2]
    a.imag[2, 2, 2, 0] = -a.imag[0, 2, 2, 2]
    a.imag[2, 2, 2, 1] = -a.imag[1, 2, 2, 2]
    a.imag[2, 2, 2, 2] = 0
    return a


def symmetrize_5(b):
    a = np.array(b, copy=True, dtype=complex)
    a.real[0, 0, 1, 0] = a.real[0, 0, 0, 1]
    a.real[0, 0, 2, 0] = a.real[0, 0, 0, 2]
    a.real[0, 0, 2, 1] = a.real[0, 0, 1, 2]
    a.real[1, 0, 0, 0] = a.real[0, 1, 0, 0]
    a.real[1, 0, 0, 1] = a.real[0, 1, 1, 0]
    a.real[1, 0, 0, 2] = a.real[0, 1, 2, 0]
    a.real[1, 0, 1, 0] = a.real[0, 1, 0, 1]
    a.real[1, 0, 1, 1] = a.real[0, 1, 1, 1]
    a.real[1, 0, 1, 2] = a.real[0, 1, 2, 1]
    a.real[1, 0, 2, 0] = a.real[0, 1, 0, 2]
    a.real[1, 0, 2, 1] = a.real[0, 1, 1, 2]
    a.real[1, 0, 2, 2] = a.real[0, 1, 2, 2]
    a.real[1, 1, 1, 0] = a.real[1, 1, 0, 1]
    a.real[1, 1, 2, 0] = a.real[1, 1, 0, 2]
    a.real[1, 1, 2, 1] = a.real[1, 1, 1, 2]
    a.real[2, 0, 0, 0] = a.real[0, 2, 0, 0]
    a.real[2, 0, 0, 1] = a.real[0, 2, 1, 0]
    a.real[2, 0, 0, 2] = a.real[0, 2, 2, 0]
    a.real[2, 0, 1, 0] = a.real[0, 2, 0, 1]
    a.real[2, 0, 1, 1] = a.real[0, 2, 1, 1]
    a.real[2, 0, 1, 2] = a.real[0, 2, 2, 1]
    a.real[2, 0, 2, 0] = a.real[0, 2, 0, 2]
    a.real[2, 0, 2, 1] = a.real[0, 2, 1, 2]
    a.real[2, 0, 2, 2] = a.real[0, 2, 2, 2]
    a.real[2, 1, 0, 0] = a.real[1, 2, 0, 0]
    a.real[2, 1, 0, 1] = a.real[1, 2, 1, 0]
    a.real[2, 1, 0, 2] = a.real[1, 2, 2, 0]
    a.real[2, 1, 1, 0] = a.real[1, 2, 0, 1]
    a.real[2, 1, 1, 1] = a.real[1, 2, 1, 1]
    a.real[2, 1, 1, 2] = a.real[1, 2, 2, 1]
    a.real[2, 1, 2, 0] = a.real[1, 2, 0, 2]
    a.real[2, 1, 2, 1] = a.real[1, 2, 1, 2]
    a.real[2, 1, 2, 2] = a.real[1, 2, 2, 2]
    a.real[2, 2, 1, 0] = a.real[2, 2, 0, 1]
    a.real[2, 2, 2, 0] = a.real[2, 2, 0, 2]
    a.real[2, 2, 2, 1] = a.real[2, 2, 1, 2]
    a.imag[0, 0, 0, 0] = 0
    a.imag[0, 0, 1, 0] = -a.imag[0, 0, 0, 1]
    a.imag[0, 0, 1, 1] = 0
    a.imag[0, 0, 2, 0] = -a.imag[0, 0, 0, 2]
    a.imag[0, 0, 2, 1] = -a.imag[0, 0, 1, 2]
    a.imag[0, 0, 2, 2] = 0
    a.imag[1, 0, 0, 0] = -a.imag[0, 1, 0, 0]
    a.imag[1, 0, 0, 1] = -a.imag[0, 1, 1, 0]
    a.imag[1, 0, 0, 2] = -a.imag[0, 1, 2, 0]
    a.imag[1, 0, 1, 0] = -a.imag[0, 1, 0, 1]
    a.imag[1, 0, 1, 1] = -a.imag[0, 1, 1, 1]
    a.imag[1, 0, 1, 2] = -a.imag[0, 1, 2, 1]
    a.imag[1, 0, 2, 0] = -a.imag[0, 1, 0, 2]
    a.imag[1, 0, 2, 1] = -a.imag[0, 1, 1, 2]
    a.imag[1, 0, 2, 2] = -a.imag[0, 1, 2, 2]
    a.imag[1, 1, 0, 0] = 0
    a.imag[1, 1, 1, 0] = -a.imag[1, 1, 0, 1]
    a.imag[1, 1, 1, 1] = 0
    a.imag[1, 1, 2, 0] = -a.imag[1, 1, 0, 2]
    a.imag[1, 1, 2, 1] = -a.imag[1, 1, 1, 2]
    a.imag[1, 1, 2, 2] = 0
    a.imag[2, 0, 0, 0] = -a.imag[0, 2, 0, 0]
    a.imag[2, 0, 0, 1] = -a.imag[0, 2, 1, 0]
    a.imag[2, 0, 0, 2] = -a.imag[0, 2, 2, 0]
    a.imag[2, 0, 1, 0] = -a.imag[0, 2, 0, 1]
    a.imag[2, 0, 1, 1] = -a.imag[0, 2, 1, 1]
    a.imag[2, 0, 1, 2] = -a.imag[0, 2, 2, 1]
    a.imag[2, 0, 2, 0] = -a.imag[0, 2, 0, 2]
    a.imag[2, 0, 2, 1] = -a.imag[0, 2, 1, 2]
    a.imag[2, 0, 2, 2] = -a.imag[0, 2, 2, 2]
    a.imag[2, 1, 0, 0] = -a.imag[1, 2, 0, 0]
    a.imag[2, 1, 0, 1] = -a.imag[1, 2, 1, 0]
    a.imag[2, 1, 0, 2] = -a.imag[1, 2, 2, 0]
    a.imag[2, 1, 1, 0] = -a.imag[1, 2, 0, 1]
    a.imag[2, 1, 1, 1] = -a.imag[1, 2, 1, 1]
    a.imag[2, 1, 1, 2] = -a.imag[1, 2, 2, 1]
    a.imag[2, 1, 2, 0] = -a.imag[1, 2, 0, 2]
    a.imag[2, 1, 2, 1] = -a.imag[1, 2, 1, 2]
    a.imag[2, 1, 2, 2] = -a.imag[1, 2, 2, 2]
    a.imag[2, 2, 0, 0] = 0
    a.imag[2, 2, 1, 0] = -a.imag[2, 2, 0, 1]
    a.imag[2, 2, 1, 1] = 0
    a.imag[2, 2, 2, 0] = -a.imag[2, 2, 0, 2]
    a.imag[2, 2, 2, 1] = -a.imag[2, 2, 1, 2]
    a.imag[2, 2, 2, 2] = 0
    return a


def symmetrize_6(b):
    a = np.array(b, copy=True, dtype=complex)
    a.real[0, 0, 1, 0] = a.real[0, 0, 0, 1]
    a.real[0, 0, 2, 0] = a.real[0, 0, 0, 2]
    a.real[0, 0, 2, 1] = a.real[0, 0, 1, 2]
    a.real[0, 1, 0, 0] = a.real[0, 0, 0, 1]
    a.real[0, 1, 1, 0] = a.real[0, 0, 1, 1]
    a.real[0, 1, 2, 0] = a.real[0, 0, 1, 2]
    a.real[0, 2, 0, 0] = a.real[0, 0, 0, 2]
    a.real[0, 2, 0, 1] = a.real[0, 1, 0, 2]
    a.real[0, 2, 1, 0] = a.real[0, 0, 1, 2]
    a.real[0, 2, 1, 1] = a.real[0, 1, 1, 2]
    a.real[0, 2, 2, 0] = a.real[0, 0, 2, 2]
    a.real[0, 2, 2, 1] = a.real[0, 1, 2, 2]
    a.real[1, 0, 0, 0] = a.real[0, 0, 0, 1]
    a.real[1, 0, 0, 1] = a.real[0, 0, 1, 1]
    a.real[1, 0, 0, 2] = a.real[0, 0, 1, 2]
    a.real[1, 0, 1, 0] = a.real[0, 1, 0, 1]
    a.real[1, 0, 1, 1] = a.real[0, 1, 1, 1]
    a.real[1, 0, 1, 2] = a.real[0, 1, 2, 1]
    a.real[1, 0, 2, 0] = a.real[0, 1, 0, 2]
    a.real[1, 0, 2, 1] = a.real[0, 1, 1, 2]
    a.real[1, 0, 2, 2] = a.real[0, 1, 2, 2]
    a.real[1, 1, 0, 0] = a.real[0, 0, 1, 1]
    a.real[1, 1, 0, 1] = a.real[0, 1, 1, 1]
    a.real[1, 1, 0, 2] = a.real[0, 1, 1, 2]
    a.real[1, 1, 1, 0] = a.real[0, 1, 1, 1]
    a.real[1, 1, 2, 0] = a.real[0, 1, 1, 2]
    a.real[1, 1, 2, 1] = a.real[1, 1, 1, 2]
    a.real[1, 2, 0, 0] = a.real[0, 0, 1, 2]
    a.real[1, 2, 0, 1] = a.real[0, 1, 1, 2]
    a.real[1, 2, 0, 2] = a.real[0, 2, 1, 2]
    a.real[1, 2, 1, 0] = a.real[0, 1, 2, 1]
    a.real[1, 2, 1, 1] = a.real[1, 1, 1, 2]
    a.real[1, 2, 2, 0] = a.real[0, 1, 2, 2]
    a.real[1, 2, 2, 1] = a.real[1, 1, 2, 2]
    a.real[2, 0, 0, 0] = a.real[0, 0, 0, 2]
    a.real[2, 0, 0, 1] = a.real[0, 0, 1, 2]
    a.real[2, 0, 0, 2] = a.real[0, 0, 2, 2]
    a.real[2, 0, 1, 0] = a.real[0, 1, 0, 2]
    a.real[2, 0, 1, 1] = a.real[0, 1, 1, 2]
    a.real[2, 0, 1, 2] = a.real[0, 1, 2, 2]
    a.real[2, 0, 2, 0] = a.real[0, 2, 0, 2]
    a.real[2, 0, 2, 1] = a.real[0, 2, 1, 2]
    a.real[2, 0, 2, 2] = a.real[0, 2, 2, 2]
    a.real[2, 1, 0, 0] = a.real[0, 0, 1, 2]
    a.real[2, 1, 0, 1] = a.real[0, 1, 2, 1]
    a.real[2, 1, 0, 2] = a.real[0, 1, 2, 2]
    a.real[2, 1, 1, 0] = a.real[0, 1, 1, 2]
    a.real[2, 1, 1, 1] = a.real[1, 1, 1, 2]
    a.real[2, 1, 1, 2] = a.real[1, 1, 2, 2]
    a.real[2, 1, 2, 0] = a.real[0, 2, 1, 2]
    a.real[2, 1, 2, 1] = a.real[1, 2, 1, 2]
    a.real[2, 1, 2, 2] = a.real[1, 2, 2, 2]
    a.real[2, 2, 0, 0] = a.real[0, 0, 2, 2]
    a.real[2, 2, 0, 1] = a.real[0, 1, 2, 2]
    a.real[2, 2, 0, 2] = a.real[0, 2, 2, 2]
    a.real[2, 2, 1, 0] = a.real[0, 1, 2, 2]
    a.real[2, 2, 1, 1] = a.real[1, 1, 2, 2]
    a.real[2, 2, 1, 2] = a.real[1, 2, 2, 2]
    a.real[2, 2, 2, 0] = a.real[0, 2, 2, 2]
    a.real[2, 2, 2, 1] = a.real[1, 2, 2, 2]
    a.imag[0, 0, 0, 0] = 0
    a.imag[0, 0, 1, 0] = -a.imag[0, 0, 0, 1]
    a.imag[0, 0, 1, 1] = 0
    a.imag[0, 0, 2, 0] = -a.imag[0, 0, 0, 2]
    a.imag[0, 0, 2, 1] = -a.imag[0, 0, 1, 2]
    a.imag[0, 0, 2, 2] = 0
    a.imag[0, 1, 0, 0] = a.imag[0, 0, 0, 1]
    a.imag[0, 1, 1, 0] = 0
    a.imag[0, 1, 2, 0] = -a.imag[0, 0, 1, 2]
    a.imag[0, 2, 0, 0] = a.imag[0, 0, 0, 2]
    a.imag[0, 2, 0, 1] = a.imag[0, 1, 0, 2]
    a.imag[0, 2, 1, 0] = a.imag[0, 0, 1, 2]
    a.imag[0, 2, 1, 1] = a.imag[0, 1, 1, 2]
    a.imag[0, 2, 2, 0] = 0
    a.imag[0, 2, 2, 1] = a.imag[0, 1, 2, 2]
    a.imag[1, 0, 0, 0] = -a.imag[0, 0, 0, 1]
    a.imag[1, 0, 0, 1] = 0
    a.imag[1, 0, 0, 2] = a.imag[0, 0, 1, 2]
    a.imag[1, 0, 1, 0] = -a.imag[0, 1, 0, 1]
    a.imag[1, 0, 1, 1] = -a.imag[0, 1, 1, 1]
    a.imag[1, 0, 1, 2] = -a.imag[0, 1, 2, 1]
    a.imag[1, 0, 2, 0] = -a.imag[0, 1, 0, 2]
    a.imag[1, 0, 2, 1] = -a.imag[0, 1, 1, 2]
    a.imag[1, 0, 2, 2] = -a.imag[0, 1, 2, 2]
    a.imag[1, 1, 0, 0] = 0
    a.imag[1, 1, 0, 1] = a.imag[0, 1, 1, 1]
    a.imag[1, 1, 0, 2] = a.imag[0, 1, 1, 2]
    a.imag[1, 1, 1, 0] = -a.imag[0, 1, 1, 1]
    a.imag[1, 1, 1, 1] = 0
    a.imag[1, 1, 2, 0] = -a.imag[0, 1, 1, 2]
    a.imag[1, 1, 2, 1] = -a.imag[1, 1, 1, 2]
    a.imag[1, 1, 2, 2] = 0
    a.imag[1, 2, 0, 0] = a.imag[0, 0, 1, 2]
    a.imag[1, 2, 0, 1] = a.imag[0, 1, 1, 2]
    a.imag[1, 2, 0, 2] = a.imag[0, 2, 1, 2]
    a.imag[1, 2, 1, 0] = -a.imag[0, 1, 2, 1]
    a.imag[1, 2, 1, 1] = a.imag[1, 1, 1, 2]
    a.imag[1, 2, 2, 0] = -a.imag[0, 1, 2, 2]
    a.imag[1, 2, 2, 1] = 0
    a.imag[2, 0, 0, 0] = -a.imag[0, 0, 0, 2]
    a.imag[2, 0, 0, 1] = -a.imag[0, 0, 1, 2]
    a.imag[2, 0, 0, 2] = 0
    a.imag[2, 0, 1, 0] = -a.imag[0, 1, 0, 2]
    a.imag[2, 0, 1, 1] = -a.imag[0, 1, 1, 2]
    a.imag[2, 0, 1, 2] = -a.imag[0, 1, 2, 2]
    a.imag[2, 0, 2, 0] = -a.imag[0, 2, 0, 2]
    a.imag[2, 0, 2, 1] = -a.imag[0, 2, 1, 2]
    a.imag[2, 0, 2, 2] = -a.imag[0, 2, 2, 2]
    a.imag[2, 1, 0, 0] = -a.imag[0, 0, 1, 2]
    a.imag[2, 1, 0, 1] = a.imag[0, 1, 2, 1]
    a.imag[2, 1, 0, 2] = a.imag[0, 1, 2, 2]
    a.imag[2, 1, 1, 0] = -a.imag[0, 1, 1, 2]
    a.imag[2, 1, 1, 1] = -a.imag[1, 1, 1, 2]
    a.imag[2, 1, 1, 2] = 0
    a.imag[2, 1, 2, 0] = -a.imag[0, 2, 1, 2]
    a.imag[2, 1, 2, 1] = -a.imag[1, 2, 1, 2]
    a.imag[2, 1, 2, 2] = -a.imag[1, 2, 2, 2]
    a.imag[2, 2, 0, 0] = 0
    a.imag[2, 2, 0, 1] = a.imag[0, 1, 2, 2]
    a.imag[2, 2, 0, 2] = a.imag[0, 2, 2, 2]
    a.imag[2, 2, 1, 0] = -a.imag[0, 1, 2, 2]
    a.imag[2, 2, 1, 1] = 0
    a.imag[2, 2, 1, 2] = a.imag[1, 2, 2, 2]
    a.imag[2, 2, 2, 0] = -a.imag[0, 2, 2, 2]
    a.imag[2, 2, 2, 1] = -a.imag[1, 2, 2, 2]
    a.imag[2, 2, 2, 2] = 0
    return a


def symmetrize_7(b):
    a = np.array(b, copy=True, dtype=complex)
    a[1, 0, 0, 0] = a[0, 1, 0, 0]
    a[1, 0, 0, 1] = a[0, 1, 0, 1]
    a[1, 0, 0, 2] = a[0, 1, 0, 2]
    a[1, 0, 1, 0] = a[0, 1, 1, 0]
    a[1, 0, 1, 1] = a[0, 1, 1, 1]
    a[1, 0, 1, 2] = a[0, 1, 1, 2]
    a[1, 0, 2, 0] = a[0, 1, 2, 0]
    a[1, 0, 2, 1] = a[0, 1, 2, 1]
    a[1, 0, 2, 2] = a[0, 1, 2, 2]
    a[2, 0, 0, 0] = a[0, 2, 0, 0]
    a[2, 0, 0, 1] = a[0, 2, 0, 1]
    a[2, 0, 0, 2] = a[0, 2, 0, 2]
    a[2, 0, 1, 0] = a[0, 2, 1, 0]
    a[2, 0, 1, 1] = a[0, 2, 1, 1]
    a[2, 0, 1, 2] = a[0, 2, 1, 2]
    a[2, 0, 2, 0] = a[0, 2, 2, 0]
    a[2, 0, 2, 1] = a[0, 2, 2, 1]
    a[2, 0, 2, 2] = a[0, 2, 2, 2]
    a[2, 1, 0, 0] = a[1, 2, 0, 0]
    a[2, 1, 0, 1] = a[1, 2, 0, 1]
    a[2, 1, 0, 2] = a[1, 2, 0, 2]
    a[2, 1, 1, 0] = a[1, 2, 1, 0]
    a[2, 1, 1, 1] = a[1, 2, 1, 1]
    a[2, 1, 1, 2] = a[1, 2, 1, 2]
    a[2, 1, 2, 0] = a[1, 2, 2, 0]
    a[2, 1, 2, 1] = a[1, 2, 2, 1]
    a[2, 1, 2, 2] = a[1, 2, 2, 2]
    return a


def symmetrize_8(b):
    """Symmetrize class-8 coefficients.

    Note that this function does not correctly take into account the
    translation between a basis where Wilson coefficients are symmetrized
    like the operators and the non-redundant WCxf basis!
    """
    a = np.array(b, copy=True, dtype=complex)
    a[1, 0, 0, 0] = a[0, 0, 1, 0]
    a[1, 0, 0, 1] = a[0, 0, 1, 1]
    a[1, 0, 0, 2] = a[0, 0, 1, 2]
    a[1, 1, 0, 0] = a[0, 1, 1, 0]
    a[1, 1, 0, 1] = a[0, 1, 1, 1]
    a[1, 1, 0, 2] = a[0, 1, 1, 2]
    a[2, 0, 0, 0] = a[0, 0, 2, 0]
    a[2, 0, 0, 1] = a[0, 0, 2, 1]
    a[2, 0, 0, 2] = a[0, 0, 2, 2]
    a[2, 0, 1, 0] = a[1, 2, 0, 0]+a[1, 0, 2, 0]-a[0, 2, 1, 0]
    a[2, 0, 1, 1] = a[1, 2, 0, 1]+a[1, 0, 2, 1]-a[0, 2, 1, 1]
    a[2, 0, 1, 2] = a[1, 2, 0, 2]+a[1, 0, 2, 2]-a[0, 2, 1, 2]
    a[2, 1, 0, 0] = a[0, 2, 1, 0]+a[0, 1, 2, 0]-a[1, 2, 0, 0]
    a[2, 1, 0, 1] = a[0, 2, 1, 1]+a[0, 1, 2, 1]-a[1, 2, 0, 1]
    a[2, 1, 0, 2] = a[0, 2, 1, 2]+a[0, 1, 2, 2]-a[1, 2, 0, 2]
    a[2, 1, 1, 0] = a[1, 1, 2, 0]
    a[2, 1, 1, 1] = a[1, 1, 2, 1]
    a[2, 1, 1, 2] = a[1, 1, 2, 2]
    a[2, 2, 0, 0] = a[0, 2, 2, 0]
    a[2, 2, 0, 1] = a[0, 2, 2, 1]
    a[2, 2, 0, 2] = a[0, 2, 2, 2]
    a[2, 2, 1, 0] = a[1, 2, 2, 0]
    a[2, 2, 1, 1] = a[1, 2, 2, 1]
    a[2, 2, 1, 2] = a[1, 2, 2, 2]
    return a


def scale_8(b):
    """Translations necessary for class-8 coefficients
    to go from a basis with only non-redundant WCxf
    operators to a basis where the Wilson coefficients are symmetrized like
    the operators."""
    a = np.array(b, copy=True, dtype=complex)
    for i in range(3):
        a[0, 0, 1, i] = 1/2 * b[0, 0, 1, i]
        a[0, 0, 2, i] = 1/2 * b[0, 0, 2, i]
        a[0, 1, 1, i] = 1/2 * b[0, 1, 1, i]
        a[0, 1, 2, i] = 2/3 * b[0, 1, 2, i] - 1/6 * b[0, 2, 1, i] - 1/6 * b[1, 0, 2, i] + 1/6 * b[1, 2, 0, i]
        a[0, 2, 1, i] = - (1/6) * b[0, 1, 2, i] + 2/3 * b[0, 2, 1, i] + 1/6 * b[1, 0, 2, i] + 1/3 * b[1, 2, 0, i]
        a[0, 2, 2, i] = 1/2 * b[0, 2, 2, i]
        a[1, 0, 2, i] = - (1/6) * b[0, 1, 2, i] + 1/6 * b[0, 2, 1, i] + 2/3 * b[1, 0, 2, i] - 1/6 * b[1, 2, 0, i]
        a[1, 1, 2, i] = 1/2 * b[1, 1, 2, i]
        a[1, 2, 0, i] = 1/6 * b[0, 1, 2, i] + 1/3 * b[0, 2, 1, i] - 1/6 * b[1, 0, 2, i] + 2/3 * b[1, 2, 0, i]
        a[1, 2, 2, i] = 1/2 * b[1, 2, 2, i]
    return a


def unscale_8(b):
    """Translations necessary for class-8 coefficients
    to go from a basis where the Wilson coefficients are symmetrized like
    the operators to a basis with only non-redundant WCxf operators."""
    a = np.array(b, copy=True, dtype=complex)
    for i in range(3):
        a[0, 0, 1, i] = 2 * b[0, 0, 1, i]
        a[0, 0, 2, i] = 2 * b[0, 0, 2, i]
        a[0, 1, 1, i] = 2 * b[0, 1, 1, i]
        a[0, 1, 2, i] = 2 * b[0, 1, 2, i] + b[0, 2, 1, i] - b[1, 2, 0, i]
        a[0, 2, 1, i] = b[0, 1, 2, i] + 3 * b[0, 2, 1, i] - b[1, 0, 2, i] - 2 * b[1, 2, 0, i]
        a[0, 2, 2, i] = 2 * b[0, 2, 2, i]
        a[1, 0, 2, i] = - b[0, 2, 1, i] + 2 * b[1, 0, 2, i] + b[1, 2, 0, i]
        a[1, 1, 2, i] = 2 * b[1, 1, 2, i]
        a[1, 2, 0, i] = - b[0, 1, 2, i] - 2 * b[0, 2, 1, i] + b[1, 0, 2, i] + 3 * b[1, 2, 0, i]
        a[1, 2, 2, i] = 2 * b[1, 2, 2, i]
    return a


def symmetrize_9(b):
    a = np.array(b, copy=True, dtype=complex)
    a[1, 0] = a[0, 1]
    a[2, 0] = a[0, 2]
    a[2, 1] = a[1, 2]
    return a


def symmetrize(C):
    """Symmetrize the Wilson coefficient arrays.

    Note that this function does not take into account the symmetry factors
    that occur when transitioning from a basis with only non-redundant operators
    (like in WCxf) to a basis where the Wilson coefficients are symmetrized
    like the operators. See `symmetrize_nonred` for this case."""
    C_symm = {}
    for i, v in C.items():
        if i in C_symm_keys[0]:
            C_symm[i] = v.real
        elif i in C_symm_keys[1] + C_symm_keys[3]:
            C_symm[i] = v # nothing to do
        elif i in C_symm_keys[2]:
            C_symm[i] = symmetrize_2(C[i])
        elif i in C_symm_keys[4]:
            C_symm[i] = symmetrize_4(C[i])
        elif i in C_symm_keys[5]:
            C_symm[i] = symmetrize_5(C[i])
        elif i in C_symm_keys[6]:
            C_symm[i] = symmetrize_6(C[i])
        elif i in C_symm_keys[7]:
            C_symm[i] = symmetrize_7(C[i])
        elif i in C_symm_keys[8]:
            C_symm[i] = symmetrize_8(C[i])
        elif i in C_symm_keys[9]:
            C_symm[i] = symmetrize_9(C[i])
    return C_symm


def arrays2wcxf(C):
    """Convert a dictionary with Wilson coefficient names as keys and
    numbers or numpy arrays as values to a dictionary with a Wilson coefficient
    name followed by underscore and numeric indices as keys and numbers as
    values. This is needed for the output in WCxf format."""
    d = {}
    for k, v in C.items():
        if np.shape(v) == () or np.shape(v) == (1,):
            d[k] = v
        else:
            ind = np.indices(v.shape).reshape(v.ndim, v.size).T
            for i in ind:
                name = k + '_' + ''.join([str(int(j) + 1) for j in i])
                d[name] = v[tuple(i)]
    return d


def wcxf2arrays(d):
    """Convert a dictionary with a Wilson coefficient
    name followed by underscore and numeric indices as keys and numbers as
    values to a dictionary with Wilson coefficient names as keys and
    numbers or numpy arrays as values. This is needed for the parsing
    of input in WCxf format."""
    C = {}
    for k, v in d.items():
        name = k.split('_')[0]
        s = C_keys_shape[name]
        if s == 1:
            C[k] = v
        else:
            ind = k.split('_')[-1]
            if name not in C:
                C[name] = np.zeros(s, dtype=complex)
            C[name][tuple([int(i) - 1 for i in ind])] = v
    return C


def add_missing(C):
    """Add arrays with zeros for missing Wilson coefficient keys"""
    C_out = C.copy()
    for k in (set(WC_keys) - set(C.keys())):
        s = C_keys_shape[k]
        if s == 1:
            C_out[k] = 0
        else:
            C_out[k] = np.zeros(C_keys_shape[k])
    return C_out


def flavor_rotation(C_in, Uq, Uu, Ud, Ul, Ue):
    """Gauge-invariant $U(3)^5$ flavor rotation of all Wilson coefficients."""
    C = {}
    # nothing to do for purely bosonic operators
    for k in WC_keys_0f:
        if k in C_in:
            C[k] = C_in[k]
    # see 1704.03888 table 4 (but staying SU(2) invariant here)
    # LR
    for k in ['ephi', 'eW', 'eB']:
        if k in C_in:
            C[k] = Ul.conj().T @ C_in[k] @ Ue
    for k in ['uphi', 'uW', 'uB', 'uG']:
        if k in C_in:
            C[k] = Uq.conj().T @ C_in[k] @ Uu
    for k in ['dphi', 'dW', 'dB', 'dG']:
        if k in C_in:
            C[k] = Uq.conj().T @ C_in[k] @ Ud
    # LL
    for k in ['phil1', 'phil3']:
        if k in C_in:
            C[k] = Ul.conj().T @ C_in[k] @ Ul
    for k in ['phiq1', 'phiq3']:
        if k in C_in:
            C[k] = Uq.conj().T @ C_in[k] @ Uq
    if 'llphiphi' in C_in:
        C['llphiphi'] = Ul.T @ C_in['llphiphi'] @ Ul
    # RR
    if 'phie' in C_in:
        C['phie'] = Ue.conj().T @ C_in['phie'] @ Ue
    if 'phiu' in C_in:
        C['phiu'] = Uu.conj().T @ C_in['phiu'] @ Uu
    if 'phid' in C_in:
        C['phid'] = Ud.conj().T @ C_in['phid'] @ Ud
    if 'phiud' in C_in:
        C['phiud'] = Uu.conj().T @ C_in['phiud'] @ Ud
    # 4-fermion
    if 'll' in C_in:
        C['ll'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ul, Ul, Ul.conj(), Ul.conj(), C_in['ll'])
    if 'ee' in C_in:
        C['ee'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ue, Ue, Ue.conj(), Ue.conj(), C_in['ee'])
    if 'le' in C_in:
        C['le'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ul, Ue, Ul.conj(), Ue.conj(), C_in['le'])
    if 'qq1' in C_in:
        C['qq1'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Uq, Uq.conj(), Uq.conj(), C_in['qq1'])
    if 'qq3' in C_in:
        C['qq3'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Uq, Uq.conj(), Uq.conj(), C_in['qq3'])
    if 'dd' in C_in:
        C['dd'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ud, Ud, Ud.conj(), Ud.conj(), C_in['dd'])
    if 'uu' in C_in:
        C['uu'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uu, Uu, Uu.conj(), Uu.conj(), C_in['uu'])
    if 'ud8' in C_in:
        C['ud8'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uu, Ud, Uu.conj(), Ud.conj(), C_in['ud8'])
    if 'ud1' in C_in:
        C['ud1'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uu, Ud, Uu.conj(), Ud.conj(), C_in['ud1'])
    if 'qu1' in C_in:
        C['qu1'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Uu, Uq.conj(), Uu.conj(), C_in['qu1'])
    if 'qu8' in C_in:
        C['qu8'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Uu, Uq.conj(), Uu.conj(), C_in['qu8'])
    if 'qd1' in C_in:
        C['qd1'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Ud, Uq.conj(), Ud.conj(), C_in['qd1'])
    if 'qd8' in C_in:
        C['qd8'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Ud, Uq.conj(), Ud.conj(), C_in['qd8'])
    if 'quqd1' in C_in:
        C['quqd1'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uu, Ud, Uq.conj(), Uq.conj(), C_in['quqd1'])
    if 'quqd8' in C_in:
        C['quqd8'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uu, Ud, Uq.conj(), Uq.conj(), C_in['quqd8'])
    if 'lq1' in C_in:
        C['lq1'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ul, Uq, Ul.conj(), Uq.conj(), C_in['lq1'])
    if 'lq3' in C_in:
        C['lq3'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ul, Uq, Ul.conj(), Uq.conj(), C_in['lq3'])
    if 'ld' in C_in:
        C['ld'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ul, Ud, Ul.conj(), Ud.conj(), C_in['ld'])
    if 'lu' in C_in:
        C['lu'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ul, Uu, Ul.conj(), Uu.conj(), C_in['lu'])
    if 'qe' in C_in:
        C['qe'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Ue, Uq.conj(), Ue.conj(), C_in['qe'])
    if 'ed' in C_in:
        C['ed'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ue, Ud, Ue.conj(), Ud.conj(), C_in['ed'])
    if 'eu' in C_in:
        C['eu'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ue, Uu, Ue.conj(), Uu.conj(), C_in['eu'])
    if 'ledq' in C_in:
        C['ledq'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ue, Uq, Ul.conj(), Ud.conj(), C_in['ledq'])
    if 'lequ1' in C_in:
        C['lequ1'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ue, Uu, Ul.conj(), Uq.conj(), C_in['lequ1'])
    if 'lequ3' in C_in:
        C['lequ3'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ue, Uu, Ul.conj(), Uq.conj(), C_in['lequ3'])
    if 'duql' in C_in:
        C['duql'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uu, Ul, Ud, Uq, C_in['duql'])
    if 'qque' in C_in:
        C['qque'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Ue, Uq, Uu, C_in['qque'])
    if 'qqql' in C_in:
        C['qqql'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Ul, Uq, Uq, C_in['qqql'])
    if 'duue' in C_in:
        C['duue'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uu, Ue, Ud, Uu, C_in['duue'])
    return C


def C_array2dict(C):
    """Convert a 1D array containing C values to a dictionary."""
    d = OrderedDict()
    i=0
    for k in C_keys:
        s = C_keys_shape[k]
        if s == 1:
            j = i+1
            d[k] = C[i]
        else:
            j = i \
      + reduce(operator.mul, s, 1)
            d[k] = C[i:j].reshape(s)
        i = j
    return d


def C_dict2array(C):
    """Convert an OrderedDict containing C values to a 1D array."""
    return np.hstack([np.asarray(C[k]).ravel() for k in C_keys])


# computing the scale vector required for scale_dict below
# initialize with factor 1
_d_4 = np.zeros((3,3,3,3))
_d_6 = np.zeros((3,3,3,3))
_d_7 = np.zeros((3,3,3,3))
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                # class 4: symmetric under interachange of currents
                _d_4[i, j, k, l] = len({(i, j, k, l), (k, l, i, j)})
                # class 6: symmetric under interachange of currents + Fierz
                _d_6[i, j, k, l] = len({(i, j, k, l), (k, l, i, j), (k, j, i, l), (i, l, k, j)})
                # class 7: symmetric under interachange of first two indices
                _d_7[i, j, k, l] = len({(i, j, k, l), (j, i, k, l)})


_scale_dict = C_array2dict(np.ones(9999))
for k in C_symm_keys[4]:
    _scale_dict[k] = _d_4
for k in C_symm_keys[6]:
    _scale_dict[k] = _d_6
for k in C_symm_keys[7]:
    _scale_dict[k] = _d_7


def symmetrize_nonred(C):
    """Symmetrize the Wilson coefficient arrays.

    This function takes into account the symmetry factors
    that occur when transitioning from a basis with only non-redundant operators
    (like in WCxf) to a basis where the Wilson coefficients are symmetrized
    like the operators."""
    C_symm = {}
    for i, v in C.items():
        if i in C_symm_keys[0]:
            C_symm[i] = v.real
        elif i in C_symm_keys[1] + C_symm_keys[3]:
            C_symm[i] = v # nothing to do
        elif i in C_symm_keys[2]:
            C_symm[i] = symmetrize_2(C[i])
        elif i in C_symm_keys[4]:
            C_symm[i] = symmetrize_4(C[i])
            C_symm[i] = C_symm[i] / _d_4
        elif i in C_symm_keys[5]:
            C_symm[i] = symmetrize_5(C[i])
        elif i in C_symm_keys[6]:
            C_symm[i] = symmetrize_6(C[i])
            C_symm[i] = C_symm[i] / _d_6
        elif i in C_symm_keys[7]:
            C_symm[i] = symmetrize_7(C[i])
            C_symm[i] = C_symm[i] / _d_7
        elif i in C_symm_keys[8]:
            C_symm[i] = scale_8(C[i])
            C_symm[i] = symmetrize_8(C_symm[i])
        elif i in C_symm_keys[9]:
            C_symm[i] = symmetrize_9(C[i])
    return C_symm


def unscale_dict(C):
    """Undo the scaling applied in `scale_dict`."""
    C_out = {k: _scale_dict[k] * v for k, v in C.items()}
    for k in C_symm_keys[8]:
        C_out['qqql'] = unscale_8(C_out['qqql'])
    return C_out


def wcxf2arrays_symmetrized(d):
    """Convert a dictionary with a Wilson coefficient
    name followed by underscore and numeric indices as keys and numbers as
    values to a dictionary with Wilson coefficient names as keys and
    numbers or numpy arrays as values.


    In contrast to `wcxf2arrays`, here the numpy arrays fulfill the same
    symmetry relations as the operators (i.e. they contain redundant entries)
    and they do not contain undefined indices.

    Zero arrays are added for missing coefficients."""
    C = wcxf2arrays(d)
    C = symmetrize_nonred(C)
    C = add_missing(C)
    return C


def arrays2wcxf_nonred(C):
    """Convert a dictionary with Wilson coefficient names as keys and
    numbers or numpy arrays as values to a dictionary with a Wilson coefficient
    name followed by underscore and numeric indices as keys and numbers as
    values.

    In contrast to `arrays2wcxf`, here the Wilson coefficient arrays are assumed
    to fulfill the same symmetry relations as the operators, i.e. contain
    redundant entries, while the WCxf output refers to the non-redundant basis."""
    C_out = unscale_dict(C)
    d = arrays2wcxf(C_out)
    return d
