"""Function to perform the RG evolution for Wilson coefficients of a given
sector."""


from wetrunner.definitions import sectors
from collections import OrderedDict
import numpy as np
from functools import lru_cache
from wetrunner import adm
from math import log


# new ADM functions
@lru_cache(maxsize=32)
def admeig(classname, f, m_u, m_d, m_s, m_c, m_b, m_e, m_mu, m_tau):
    """Compute the eigenvalues and eigenvectors for a QCD anomalous dimension
    matrix that is defined in `adm.adm_s_X` where X is the name of the sector.

    Supports memoization. Output analogous to `np.linalg.eig`."""
    args = f, m_u, m_d, m_s, m_c, m_b, m_e, m_mu, m_tau
    A = getattr(adm, 'adm_s_' + classname)(*args)
    w, v = np.linalg.eig(A.T)
    return w, v


@lru_cache(maxsize=32)
def getUs(classname, eta_s, f, alpha_s, alpha_e, m_u, m_d, m_s, m_c, m_b, m_e, m_mu, m_tau):
    """Get the QCD evolution matrix."""
    w, v = admeig(classname, f, m_u, m_d, m_s, m_c, m_b, m_e, m_mu, m_tau)
    b0s = 11 - 2 * f / 3
    a = w / (2 * b0s)
    return v @ np.diag(eta_s**a) @ np.linalg.inv(v)


@lru_cache(maxsize=32)
def getUe(classname, eta_s, f, alpha_s, alpha_e, m_u, m_d, m_s, m_c, m_b, m_e, m_mu, m_tau):
    """Get the QCD evolution matrix."""
    args = f, m_u, m_d, m_s, m_c, m_b, m_e, m_mu, m_tau
    A = getattr(adm, 'adm_e_' + classname)(*args)
    w, v = admeig(classname, *args)
    b0s = 11 - 2 * f / 3
    a = w / (2 * b0s)
    K = np.linalg.inv(v) @ A.T @ v
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            if a[i] - a[j] != 1:
                K[i, j] *= (eta_s**(a[j] + 1) - eta_s**a[i]) / (a[i] - a[j] - 1)
            else:
                K[i, j] *= eta_s**a[i] * log(1 / eta_s)
    return -alpha_e / (2 * b0s * alpha_s) * v @ K @ np.linalg.inv(v)


def run_sector(sector, C_in, eta_s, f, p):
    Cdictout = OrderedDict()
    for classname, C_lists in sectors[sector].items():
        for i, keylist in enumerate(C_lists):
            C_input = np.array([C_in.get(key, 0) for key in keylist])
            if np.count_nonzero(C_input) == 0 or classname == 'Vnu':
                # nothing to do for SM-like WCs or qqnunu operators
                C_result = C_input
            else:
                Us = getUs(classname, eta_s, f, **p)
                try:
                    Ue = getUe(classname, eta_s, f, **p)
                except AttributeError:
                    # Ue not implemented!
                    Ue = 0 * Us
                C_result = (Us + Ue) @ C_input
            for j in range(len(C_result)):
                Cdictout[keylist[j]] = C_result[j]
    return Cdictout
