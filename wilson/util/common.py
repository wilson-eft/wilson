import numpy as np
from collections import OrderedDict
from functools import reduce
import operator




def C_array2dict(C, C_keys, C_keys_shape):
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


def C_dict2array(C, C_keys):
    """Convert an OrderedDict containing C values to a 1D array."""
    return np.hstack([np.asarray(C[k]).ravel() for k in C_keys])


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


def wcxf2arrays(d, C_keys_shape):
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


def add_missing(C, WC_keys, C_keys_shape):
    """Add arrays with zeros for missing Wilson coefficient keys"""
    C_out = C.copy()
    for k in (set(WC_keys) - set(C.keys())):
        s = C_keys_shape[k]
        if s == 1:
            C_out[k] = 0
        else:
            C_out[k] = np.zeros(C_keys_shape[k])
    return C_out


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


def symmetrize_41(b):
    a = np.array(b, copy=True, dtype=complex)
    a[0,1,0,0] = a[0,0,0,1]
    a[0,2,0,0] = a[0,0,0,2]
    a[0,2,0,1] = a[0,1,0,2]
    a[1,0,0,0] = a[0,0,1,0]
    a[1,0,0,1] = a[0,1,1,0]
    a[1,0,0,2] = a[0,2,1,0]
    a[1,1,0,0] = a[0,0,1,1]
    a[1,1,0,1] = a[0,1,1,1]
    a[1,1,0,2] = a[0,2,1,1]
    a[1,1,1,0] = a[1,0,1,1]
    a[1,2,0,0] = a[0,0,1,2]
    a[1,2,0,1] = a[0,1,1,2]
    a[1,2,0,2] = a[0,2,1,2]
    a[1,2,1,0] = a[1,0,1,2]
    a[1,2,1,1] = a[1,1,1,2]
    a[2,0,0,0] = a[0,0,2,0]
    a[2,0,0,1] = a[0,1,2,0]
    a[2,0,0,2] = a[0,2,2,0]
    a[2,0,1,0] = a[1,0,2,0]
    a[2,0,1,1] = a[1,1,2,0]
    a[2,0,1,2] = a[1,2,2,0]
    a[2,1,0,0] = a[0,0,2,1]
    a[2,1,0,1] = a[0,1,2,1]
    a[2,1,0,2] = a[0,2,2,1]
    a[2,1,1,0] = a[1,0,2,1]
    a[2,1,1,1] = a[1,1,2,1]
    a[2,1,1,2] = a[1,2,2,1]
    a[2,1,2,0] = a[2,0,2,1]
    a[2,2,0,0] = a[0,0,2,2]
    a[2,2,0,1] = a[0,1,2,2]
    a[2,2,0,2] = a[0,2,2,2]
    a[2,2,1,0] = a[1,0,2,2]
    a[2,2,1,1] = a[1,1,2,2]
    a[2,2,1,2] = a[1,2,2,2]
    a[2,2,2,0] = a[2,0,2,2]
    a[2,2,2,1] = a[2,1,2,2]
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


def symmetrize_71(b):
    a = np.array(b, copy=True, dtype=complex)
    a[1, 0, 0, 0] = -a[0, 1, 0, 0]
    a[1, 0, 0, 1] = -a[0, 1, 0, 1]
    a[1, 0, 0, 2] = -a[0, 1, 0, 2]
    a[1, 0, 1, 0] = -a[0, 1, 1, 0]
    a[1, 0, 1, 1] = -a[0, 1, 1, 1]
    a[1, 0, 1, 2] = -a[0, 1, 1, 2]
    a[1, 0, 2, 0] = -a[0, 1, 2, 0]
    a[1, 0, 2, 1] = -a[0, 1, 2, 1]
    a[1, 0, 2, 2] = -a[0, 1, 2, 2]
    a[2, 0, 0, 0] = -a[0, 2, 0, 0]
    a[2, 0, 0, 1] = -a[0, 2, 0, 1]
    a[2, 0, 0, 2] = -a[0, 2, 0, 2]
    a[2, 0, 1, 0] = -a[0, 2, 1, 0]
    a[2, 0, 1, 1] = -a[0, 2, 1, 1]
    a[2, 0, 1, 2] = -a[0, 2, 1, 2]
    a[2, 0, 2, 0] = -a[0, 2, 2, 0]
    a[2, 0, 2, 1] = -a[0, 2, 2, 1]
    a[2, 0, 2, 2] = -a[0, 2, 2, 2]
    a[2, 1, 0, 0] = -a[1, 2, 0, 0]
    a[2, 1, 0, 1] = -a[1, 2, 0, 1]
    a[2, 1, 0, 2] = -a[1, 2, 0, 2]
    a[2, 1, 1, 0] = -a[1, 2, 1, 0]
    a[2, 1, 1, 1] = -a[1, 2, 1, 1]
    a[2, 1, 1, 2] = -a[1, 2, 1, 2]
    a[2, 1, 2, 0] = -a[1, 2, 2, 0]
    a[2, 1, 2, 1] = -a[1, 2, 2, 1]
    a[2, 1, 2, 2] = -a[1, 2, 2, 2]
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


def unscale_dict(C, C_symm_keys, scale_dict):
    """Undo the scaling applied in `scale_dict`."""
    C_out = {k: scale_dict[k] * v for k, v in C.items()}
    for k in C_symm_keys[8]:
        C_out[k] = unscale_8(C_out[k])
    return C_out


def symmetrize(C, C_symm_keys):
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


# computing the scale vector required for symmetrize_nonred
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


def symmetrize_nonred(C, C_symm_keys):
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


def wcxf2arrays_symmetrized(d, WC_keys, C_keys_shape, C_symm_keys):
    """Convert a dictionary with a Wilson coefficient
    name followed by underscore and numeric indices as keys and numbers as
    values to a dictionary with Wilson coefficient names as keys and
    numbers or numpy arrays as values.


    In contrast to `wcxf2arrays`, here the numpy arrays fulfill the same
    symmetry relations as the operators (i.e. they contain redundant entries)
    and they do not contain undefined indices.

    Zero arrays are added for missing coefficients."""
    C = wcxf2arrays(d, C_keys_shape)
    C = symmetrize_nonred(C, C_symm_keys)
    C = add_missing(C, WC_keys, C_keys_shape)
    return C


def arrays2wcxf_nonred(C, C_symm_keys, scale_dict):
    """Convert a dictionary with Wilson coefficient names as keys and
    numbers or numpy arrays as values to a dictionary with a Wilson coefficient
    name followed by underscore and numeric indices as keys and numbers as
    values.

    In contrast to `arrays2wcxf`, here the Wilson coefficient arrays are assumed
    to fulfill the same symmetry relations as the operators, i.e. contain
    redundant entries, while the WCxf output refers to the non-redundant basis."""
    C_out = unscale_dict(C, C_symm_keys, scale_dict)
    d = arrays2wcxf(C_out)
    return d
