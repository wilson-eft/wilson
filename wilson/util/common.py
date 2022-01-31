import numpy as np


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
