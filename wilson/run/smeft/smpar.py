"""Extraction of Standard Model parameters taking into account dimension-six contributions."""


from math import sqrt, pi
import numpy as np
import ckmutil
import ckmutil.ckm
import scipy.optimize
from cmath import phase
import warnings


# Default values for SM parameters: MSbar parameters at M_Z (except GF)
p = {}
p['GF'] = 1.1663787e-5
p['alpha_e'] = 1 / 127.9
p['alpha_s'] = 0.1185
p['Vus'] = 0.2243
p['Vub'] = 3.62e-3
p['Vcb'] = 4.221e-2
p['gamma'] = 1.27
p['m_e'] = 0.000511
p['m_mu'] = 0.1057
p['m_tau'] = 1.777
p['m_u'] = 0.00127  # mu(2 GeV)=0.0022
p['m_c'] = 0.635  # mc(mc)=1.28
p['m_d'] = 0.00270  # md(2 GeV)=0.0047
p['m_s'] = 0.0551  # ms(2 GeV)=0.095
p['m_b'] = 2.85  # mb(mb)=4.18
# MSbar running masses at MZ computed with "mr"
# https://github.com/apik/mr, https://arxiv.org/abs/1601.08143
p['m_t'] = 169.0
p['m_W'] = 80.20
p['m_Z'] = 91.46
p['m_h'] = 130.6


def m2Lambda_to_vMh2(m2, Lambda, C):
    """Function to numerically determine the  physical Higgs VEV and mass
    given the parameters of the Higgs potential.

    The relations used by this function have been derived from the Lagrangian
    in eq. (3.1) of arXiv:1704.03888. This paper presents results expanded up to
    linear order in the Wilson coefficients while the all-order expressions are
    implemented here.
    """
    Cphi = C['phi'].real
    Ckin = C['phiBox'].real - C['phiD'].real / 4
    if abs(Cphi) < 1e-16:
        v2 = 2 * m2 / Lambda
    else:
        sqrt_arg = Lambda**2 - 12 * Cphi * m2
        if not sqrt_arg >= 0:
            raise ValueError("'Lambda**2 - 12 * Cphi * m2' must be positive.")
        v2 = ( Lambda - sqrt(sqrt_arg) )/( 3 * Cphi )
    if not v2 > 0:
        raise ValueError('No solution with positive v2.')
    Mh2 = v2 / ( 1 - 2 * Ckin * v2 ) * ( Lambda - 3 * Cphi * v2 )
    if not Mh2 > 0:
        raise ValueError('No solution with positive Mh2.')
    return {'v': sqrt(v2), 'Mh2': Mh2}

def vMh2_to_m2Lambda(v, Mh2, C):
    """Function to numerically determine the parameters of the Higgs potential
    given the physical Higgs VEV and mass.

    The relations used by this function have been derived from the Lagrangian
    in eq. (3.1) of arXiv:1704.03888. This paper presents results expanded up to
    linear order in the Wilson coefficients while the all-order expressions are
    implemented here."""
    if not v > 0 or not Mh2 > 0:
        raise ValueError('`v` and `Mh2` are expected to be positive.')
    v2 = v**2
    Cphi = C['phi'].real
    Ckin = C['phiBox'].real - C['phiD'].real / 4
    Lambda = Mh2 / v2 - 2 * Ckin * Mh2 + 3 * Cphi * v2
    m2 = Lambda * v2 / 2 - 3/4 * Cphi * v2**2
    return {'m2': m2, 'Lambda': Lambda}

def get_gpbar(ebar, gbar, v, C):
    r"""Function to numerically determine the hypercharge gauge coupling
    in terms of $\bar e$, $\bar g$, v, and the Wilson coefficients."""
    if C['phiWB'] == 0:  # this is the trivial case
        gpbar = ebar * gbar / sqrt(gbar**2 - ebar**2)
    else:  # if epsilon != 0, need to iterate
        def f0(x):  # we want the root of this function
            gpb = x
            gb = gbar
            eps = C['phiWB'].real * (v**2)
            ebar_calc = (gb * gpb / sqrt(gb**2 + gpb**2) *
                        (1 - eps * gb * gpb / (gb**2 + gpb**2)))
            return (ebar_calc - ebar).real
        try:
            gpbar = scipy.optimize.brentq(f0, 0, 3)
        except (scipy.optimize.nonlin.NoConvergence, ValueError) as e:
            raise ValueError("No solution for gp found. This problem can be caused by very large values for one or several Wilson coefficients.")
    return gpbar * (1 - C['phiB'].real * (v**2))


def smeftpar(scale, C, basis):
    """Get the running parameters in SMEFT."""
    # start with a zero dict and update it with the input values
    MW = p['m_W']
    # MZ = p['m_Z']
    GF = p['GF']
    Mh = p['m_h']
    vb = sqrt(1 / sqrt(2) / GF)
    v = vb  # TODO
    _d = vMh2_to_m2Lambda(v=v, Mh2=Mh**2, C=C)
    m2 = _d['m2'].real
    Lambda = _d['Lambda'].real
    gsbar = sqrt(4 * pi * p['alpha_s'])
    gs = (1 - C['phiG'].real * (v**2)) * gsbar
    gbar = 2 * MW / v
    g = gbar * (1 - C['phiW'].real * (v**2))
    ebar = sqrt(4 * pi * p['alpha_e'])
    gp = get_gpbar(ebar, gbar, v, C)
    c = {}
    c['m2'] = m2
    c['Lambda'] = Lambda
    c['g'] = g
    c['gp'] = gp
    c['gs'] = gs
    K = ckmutil.ckm.ckm_tree(p['Vus'], p['Vub'], p['Vcb'], p['gamma'])
    if basis == 'Warsaw':
        Mu = K.conj().T @ np.diag([p['m_u'], p['m_c'], p['m_t']])
        Md = np.diag([p['m_d'], p['m_s'], p['m_b']])
    elif basis == 'Warsaw up':
        Mu = np.diag([p['m_u'], p['m_c'], p['m_t']])
        Md = K @ np.diag([p['m_d'], p['m_s'], p['m_b']])
    else:
        raise ValueError(f"Basis '{basis}' not supported")
    Me = np.diag([p['m_e'], p['m_mu'], p['m_tau']])
    c['Gd'] = Md / (v / sqrt(2)) + C['dphi'] * (v**2) / 2
    c['Gu'] = Mu / (v / sqrt(2)) + C['uphi'] * (v**2) / 2
    c['Ge'] = Me / (v / sqrt(2)) + C['ephi'] * (v**2) / 2
    return c


def smpar(C):
    """Get the running effective SM parameters."""
    m2 = C['m2'].real
    Lambda = C['Lambda'].real
    vMh2 = m2Lambda_to_vMh2(m2, Lambda, C)
    v = vMh2['v']
    Mh2 = vMh2['Mh2']
    GF = 1 / (sqrt(2) * v**2)  # TODO
    eps = C['phiWB'].real * (v**2)
    gb = (C['g'] / (1 - C['phiW'].real * (v**2))).real
    gpb = (C['gp'] / (1 - C['phiB'].real * (v**2))).real
    gsb = (C['gs'] / (1 - C['phiG'].real * (v**2))).real
    MW = gb * v / 2
    ZG0 = 1 + C['phiD'].real * (v**2) / 4
    MZ = (sqrt(gb**2 + gpb**2) / 2 * v
          * (1 + eps * gb * gpb / (gb**2 + gpb**2)) * ZG0)
    Mnup = -(v**2) * C['llphiphi']
    Mep = v / sqrt(2) * (C['Ge'] - C['ephi'] * (v**2) / 2)
    Mup = v / sqrt(2) * (C['Gu'] - C['uphi'] * (v**2) / 2)
    Mdp = v / sqrt(2) * (C['Gd'] - C['dphi'] * (v**2) / 2)
    UeL, Me, UeR = ckmutil.diag.msvd(Mep)
    UuL, Mu, UuR = ckmutil.diag.msvd(Mup)
    UdL, Md, UdR = ckmutil.diag.msvd(Mdp)
    UnuL, Mnu = ckmutil.diag.mtakfac(Mnup)
    eb = (gb * gpb / sqrt(gb**2 + gpb**2) *
          (1 - eps * gb * gpb / (gb**2 + gpb**2)))
    K = UuL.conj().T @ UdL
    # U = UeL.conj().T @ UnuL
    sm = {}
    sm['GF'] = GF
    sm['alpha_e'] = eb**2 / (4 * pi)
    sm['alpha_s'] = gsb**2 / (4 * pi)
    sm['Vub'] = abs(K[0, 2])
    sm['Vcb'] = abs(K[1, 2])
    sm['Vus'] = abs(K[0, 1])
    sm['gamma'] = phase(-K[0, 0] * K[0, 2].conj()
                        / (K[1, 0] * K[1, 2].conj()))
    # sm['U'] = Uu
    sm['m_W'] = MW
    sm['m_Z'] = MZ
    sm['m_h'] = sqrt(abs(Mh2))
    sm['m_u'] = Mu[0]
    sm['m_c'] = Mu[1]
    sm['m_t'] = Mu[2]
    sm['m_d'] = Md[0]
    sm['m_s'] = Md[1]
    sm['m_b'] = Md[2]
    sm['m_e'] = Me[0]
    sm['m_mu'] = Me[1]
    sm['m_tau'] = Me[2]
    return  {k: v.real for k, v in sm.items()}
