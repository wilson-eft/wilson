"""Function to perform the RG evolution for Wilson coefficients of a given
sector."""


from wilson.run.wet.definitions import sectors, coeffs
from collections import OrderedDict
import numpy as np
from functools import lru_cache
from wilson.run.wet import adm
from math import log, sqrt, pi
from wilson import wcxf

@lru_cache(maxsize=32)
def get_permissible_wcs(classname, f):
    r"""For some classes (in particular $\Delta F=1$), only a subset of Wilson
    coefficients exist in WET-3 and WET-4. Therefore, depending on the number
    of flavours `f`, the dimensionality of the ADM has to be reduced."""
    # these are the problematic sectors
    classes = ['cu', 'db', 'sb', 'sd', 'mue', 'mutau', 'taue', 'dF0']
    if classname not in classes or f == 5:
        # for 5-flavour WET, nothing to do.
        # Neither for other classes (I, II, ...) because they exist either all
        # or not at all in WET-3 and WET-4 (they have specific flavours).
        return 'all'
    if f not in [3, 4]:
        raise ValueError("f must be 3, 4, or 5.")
    if classname == 'dF0':
        sector = 'dF=0'
    else:
        sector = classname
    perm_keys = wcxf.Basis[f'WET-{f}', 'JMS'].sectors[sector].keys()
    all_keys = coeffs[sector]
    return [i for i, c in enumerate(all_keys) if c in perm_keys]


# new ADM functions
@lru_cache(maxsize=32)
def admeig(classname, f, m_u, m_d, m_s, m_c, m_b, m_e, m_mu, m_tau):
    """Compute the eigenvalues and eigenvectors for a QCD anomalous dimension
    matrix that is defined in `adm.adm_s_X` where X is the name of the sector.

    Supports memoization. Output analogous to `np.linalg.eig`."""
    args = f, m_u, m_d, m_s, m_c, m_b, m_e, m_mu, m_tau
    A = getattr(adm, 'adm_s_' + classname)(*args)
    perm_keys = get_permissible_wcs(classname, f)
    if perm_keys != 'all':
        # remove disallowed rows & columns if necessary
        A = A[perm_keys][:, perm_keys]
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
    perm_keys = get_permissible_wcs(classname, f)
    if perm_keys != 'all':
        # remove disallowed rows & columns if necessary
        A = A[perm_keys][:, perm_keys]
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


qG = ['uG', 'dG']
qgamma = ['ugamma', 'dgamma']
lgamma = ['egamma', 'nugamma']
G3 = ['G', 'Gtilde']


def get_m(dipole_key):
    ind = dipole_key.split('_')[-1]
    gen = int(max(ind))
    if dipole_key[:2] == 'nu':
        return 0
    elif dipole_key[0] == 'e':
        return ['m_e', 'm_mu', 'm_tau'][gen - 1]
    elif dipole_key[0] == 'u':
        return ['m_u', 'm_c', 'm_t'][gen - 1]
    elif dipole_key[0] == 'd':
        return ['m_d', 'm_s', 'm_b'][gen - 1]


def scale_C(key, p):
    g = sqrt(4 * pi * p['alpha_s'])
    e = sqrt(4 * pi * p['alpha_e'])
    name = key.split('_')[0]
    if  name in qG:
        m = p[get_m(key)]
        return g / m
    elif  name in qgamma + lgamma:
        m = p[get_m(key)]
        return g**2 / e / m
    elif key in G3:
        return g
    else:
        return 1


def run_sector(sector, C_in, eta_s, f, p_in, p_out, qed_order=1, qcd_order=1):
    r"""Solve the WET RGE for a specific sector.

    Parameters:

    - sector: sector of interest
    - C_in: dictionary of Wilson coefficients
    - eta_s: ratio of $\alpha_s$ at input and output scale
    - f: number of active quark flavours
    - p_in: running parameters at the input scale
    - p_out: running parameters at the output scale
    """
    Cdictout = OrderedDict()
    classname = sectors[sector]
    keylist = coeffs[sector]
    if sector == 'dF=0':
        perm_keys = get_permissible_wcs('dF0', f)
    else:
        perm_keys = get_permissible_wcs(sector, f)
    if perm_keys != 'all':
        # remove disallowed keys if necessary
        keylist = np.asarray(keylist)[perm_keys]
    C_input = np.array([C_in.get(key, 0) for key in keylist])
    if np.count_nonzero(C_input) == 0 or classname == 'inv':
        # nothing to do for SM-like WCs or RG invariant operators
        C_result = C_input
    else:
        C_scaled = np.asarray([C_input[i] * scale_C(key, p_in) for i, key in enumerate(keylist)])
        if qcd_order == 0:
            Us = np.eye(len(C_scaled))
        elif qcd_order == 1:
            Us = getUs(classname, eta_s, f, **p_in)
        if qed_order == 0:
            Ue = np.zeros(C_scaled.shape)
        elif qed_order == 1:
            if qcd_order == 0:
                Ue = getUe(classname, 1, f, **p_in)
            else:
                Ue = getUe(classname, eta_s, f, **p_in)
        C_out = (Us + Ue) @ C_scaled
        C_result = [C_out[i] / scale_C(key, p_out) for i, key in enumerate(keylist)]
    for j in range(len(C_result)):
        Cdictout[keylist[j]] = C_result[j]
    return Cdictout
