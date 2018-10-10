"""Matcher from the SMEFT 'Warsaw up' basis to the WET JMS basis.

Based on arXiv:1709.04486."""


import numpy as np
from math import sqrt, pi
import wcxf
import wilson
from wilson.parameters import p as default_parameters
from wilson.util import smeftutil, wetutil

# Based on arXiv:1709.04486

# CONSTANTS

Nc = 3

# AUXILIARY FUNCTIONS

# Eq. (6.4)
def vT(C, p):
    GF = p['GF']
    GFx = GF - sqrt(2)/4 * ( -C['ll'][1,0,0,1] - C['ll'][0,1,1,0] + 2*C['phil3'][1,1] + 2*C['phil3'][0,0] )
    return sqrt(1/sqrt(2)/abs(GFx))

# Eq. (2.22)
def eps(C, p):
    return C["phiWB"] * vT(C, p)**2

# Eq. (2.24)
def g2b(C, p):
    p["m_W"] = p["m_W"]
    return 2*p["m_W"]/vT(C, p)

def g1b(C, p):
    alpha_e = p['alpha_e']
    eb = sqrt(4*pi*alpha_e)
    return eb*g2b(C, p)/sqrt(g2b(C, p)**2-eb**2) + eb**2*g2b(C, p)/(g2b(C, p)**2-eb**2) * eps(C, p)

# Eq. (2.23)
def sb(C, p):
    return g1b(C, p)/sqrt(g1b(C, p)**2+g2b(C, p)**2) * (1 + eps(C, p)/2. * g2b(C, p)/g1b(C, p) * ((g2b(C, p)**2-g1b(C, p)**2)/(g1b(C, p)**2+g2b(C, p)**2)))
def cb(C, p):
    return g2b(C, p)/sqrt(g1b(C, p)**2+g2b(C, p)**2) * (1 - eps(C, p)/2. * g1b(C, p)/g2b(C, p) * ((g2b(C, p)**2-g1b(C, p)**2)/(g1b(C, p)**2+g2b(C, p)**2)))

# Eq. (2.26)
def eb(C, p):
    return g2b(C, p)*sb(C, p) - 1/2.*cb(C, p)*g2b(C, p)* vT(C, p)**2*C["phiWB"]
def gzb(C, p):
    return eb(C, p)/(sb(C, p)*cb(C, p)) * (1 + (g1b(C, p)**2+g2b(C, p)**2)/(2*g1b(C, p)*g2b(C, p))*vT(C, p)**2*C["phiWB"])

# Eq. (2.30)
def wl(C, p):
    return np.eye(3)+vT(C, p)**2*C["phil3"]
def wq(C, p):
    return np.eye(3)+vT(C, p)**2*C["phiq3"]
def wr(C, p):
    return 1/2.*vT(C, p)**2*C["phiud"]
def znu(C, p):
    return np.eye(3)*1/2.-1/2.*vT(C, p)**2*C["phil1"]+1/2.*vT(C, p)**2*C["phil3"]
def zel(C, p):
    return np.eye(3)*(-1/2.+sb(C, p)**2)-1/2.*vT(C, p)**2*C["phil1"]-1/2.*vT(C, p)**2*C["phil3"]
def zer(C, p):
    return np.eye(3)*sb(C, p)**2-1/2.*vT(C, p)**2*C["phie"]
def zul(C, p):
    return np.eye(3)*(1/2.-2./3*sb(C, p)**2)-1/2.*vT(C, p)**2*C["phiq1"]+1/2.*vT(C, p)**2*C["phiq3"]
def zur(C, p):
    return np.eye(3)*(-2./3)*sb(C, p)**2-1/2.*vT(C, p)**2*C["phiu"]
def zdl(C, p):
    return np.eye(3)*(-1/2.+1/3.*sb(C, p)**2)-1/2.*vT(C, p)**2*C["phiq1"]-1/2.*vT(C, p)**2*C["phiq3"]
def zdr(C, p):
    return np.eye(3)*(1/3.)*sb(C, p)**2-1/2.*vT(C, p)**2*C["phid"]

# MATCHING CONDITIONS

# initialize empty dict that will become a dict of functions
C = {}

# Table 9
# C["nu"] = lambda C, p: 1/2.*C["llphiphi"]* vT(C, p)**2

# Table 10
# C["nugamma"] = lambda C, p: np.zeros((3,3))

# Table 11
C["egamma"] = lambda C, p: 1/sqrt(2) * (-C["eW"] * sb(C, p) + C["eB"] * cb(C, p)) * vT(C, p)

C["ugamma"] = lambda C, p: 1/sqrt(2) * (C["uW"] * sb(C, p) + C["uB"] * cb(C, p)) * vT(C, p)
C["dgamma"] = lambda C, p: 1/sqrt(2) * (-C["dW"] * sb(C, p) + C["dB"] * cb(C, p)) * vT(C, p)
C["uG"] = lambda C, p: 1/sqrt(2) * C["uG"] * vT(C, p)
C["dG"] = lambda C, p: 1/sqrt(2) * C["dG"] * vT(C, p)

#Table 12
C["G"] = lambda C, p: C["G"]
C["Gtilde"] = lambda C, p: C["Gtilde"]

# Table 13
C["VnunuLL"] = lambda C, p: C["ll"]-gzb(C, p)**2/(4*p["m_Z"]**2)*np.einsum('pr,st',znu(C, p),znu(C, p))-gzb(C, p)**2/(4*p["m_Z"]**2)*np.einsum('pt,sr',znu(C, p),znu(C, p))
C["VeeLL"] = lambda C, p: C["ll"]-gzb(C, p)**2/(4*p["m_Z"]**2)*np.einsum('pr,st',zel(C, p),zel(C, p))-gzb(C, p)**2/(4*p["m_Z"]**2)*np.einsum('pt,sr',zel(C, p),zel(C, p))
C["VnueLL"] = lambda C, p: C["ll"]+np.einsum('stpr',C["ll"])-g2b(C, p)**2/(2*p["m_W"]**2)*np.einsum('pt,rs',wl(C, p),wl(C, p).conjugate())-gzb(C, p)**2/(p["m_Z"]**2)*np.einsum('pr,st',znu(C, p),zel(C, p))

C["VnuuLL"] = lambda C, p: C["lq1"]+C["lq3"]-gzb(C, p)**2/(p["m_Z"]**2)*np.einsum('pr,st',znu(C, p),zul(C, p))
C["VnudLL"] = lambda C, p: C["lq1"]-C["lq3"]-gzb(C, p)**2/(p["m_Z"]**2)*np.einsum('pr,st',znu(C, p),zdl(C, p))
C["VeuLL"] = lambda C, p: C["lq1"]-C["lq3"]-gzb(C, p)**2/(p["m_Z"]**2)*np.einsum('pr,st',zel(C, p),zul(C, p))
C["VedLL"] = lambda C, p: C["lq1"]+C["lq3"]-gzb(C, p)**2/(p["m_Z"]**2)*np.einsum('pr,st',zel(C, p),zdl(C, p))
C["VnueduLL"] = lambda C, p: 2*C["lq3"]-g2b(C, p)**2/(2*p["m_W"]**2)*np.einsum('pr,ts',wl(C, p),wq(C, p).conjugate())
# + h.c.

C["VuuLL"] = lambda C, p: C["qq1"]+C["qq3"]-gzb(C, p)**2/(2*p["m_Z"]**2)*np.einsum('pr,st',zul(C, p),zul(C, p))
C["VddLL"] = lambda C, p: C["qq1"]+C["qq3"]-gzb(C, p)**2/(2*p["m_Z"]**2)*np.einsum('pr,st',zdl(C, p),zdl(C, p))
C["V1udLL"] = lambda C, p: C["qq1"]+np.einsum('stpr',C["qq1"])-C["qq3"]-np.einsum('stpr',C["qq3"])+2/Nc*np.einsum('ptsr',C["qq3"])+2/Nc*np.einsum('srpt',C["qq3"])-g2b(C, p)**2/(2*p["m_W"]**2)*np.einsum('pt,rs',wq(C, p),wq(C, p).conjugate())/Nc-gzb(C, p)**2/(p["m_Z"]**2)*np.einsum('pr,st',zul(C, p),zdl(C, p))
C["V8udLL"] = lambda C, p: 4*np.einsum('ptsr',C["qq3"])+4*np.einsum('srpt',C["qq3"])-g2b(C, p)**2/(p["m_W"]**2)*np.einsum('pt,rs',wq(C, p),wq(C, p).conjugate())


# Table 14
C["VeeRR"] = lambda C, p: C["ee"]-gzb(C, p)**2/(4*p["m_Z"]**2)*np.einsum('pr,st',zer(C, p),zer(C, p))-gzb(C, p)**2/(4*p["m_Z"]**2)*np.einsum('pt,sr',zer(C, p),zer(C, p))

C["VeuRR"] = lambda C, p: C["eu"]-gzb(C, p)**2/(p["m_Z"]**2)*np.einsum('pr,st',zer(C, p),zur(C, p))
C["VedRR"] = lambda C, p: C["ed"]-gzb(C, p)**2/(p["m_Z"]**2)*np.einsum('pr,st',zer(C, p),zdr(C, p))

C["VuuRR"] = lambda C, p: C["uu"]-gzb(C, p)**2/(2*p["m_Z"]**2)*np.einsum('pr,st',zur(C, p),zur(C, p))
C["VddRR"] = lambda C, p: C["dd"]-gzb(C, p)**2/(2*p["m_Z"]**2)*np.einsum('pr,st',zdr(C, p),zdr(C, p))
C["V1udRR"] = lambda C, p: C["ud1"]-g2b(C, p)**2/(2*p["m_W"]**2)*np.einsum('pt,rs',wr(C, p),wr(C, p).conjugate())/Nc-gzb(C, p)**2/(p["m_Z"]**2)*np.einsum('pr,st',zur(C, p),zdr(C, p))
C["V8udRR"] = lambda C, p: C["ud8"]-g2b(C, p)**2/(p["m_W"]**2)*np.einsum('pt,rs',wr(C, p),wr(C, p).conjugate())


# Table 15
C["VnueLR"] = lambda C, p: C["le"]-gzb(C, p)**2/(p["m_Z"]**2)*np.einsum('pr,st',znu(C, p),zer(C, p))
C["VeeLR"] = lambda C, p: C["le"]-gzb(C, p)**2/(p["m_Z"]**2)*np.einsum('pr,st',zel(C, p),zer(C, p))

C["VnuuLR"] = lambda C, p: C["lu"]-gzb(C, p)**2/(p["m_Z"]**2)*np.einsum('pr,st',znu(C, p),zur(C, p))
C["VnudLR"] = lambda C, p: C["ld"]-gzb(C, p)**2/(p["m_Z"]**2)*np.einsum('pr,st',znu(C, p),zdr(C, p))
C["VeuLR"] = lambda C, p: C["lu"]-gzb(C, p)**2/(p["m_Z"]**2)*np.einsum('pr,st',zel(C, p),zur(C, p))
C["VedLR"] = lambda C, p: C["ld"]-gzb(C, p)**2/(p["m_Z"]**2)*np.einsum('pr,st',zel(C, p),zdr(C, p))
C["VueLR"] = lambda C, p: C["qe"]-gzb(C, p)**2/(p["m_Z"]**2)*np.einsum('pr,st',zul(C, p),zer(C, p))
C["VdeLR"] = lambda C, p: C["qe"]-gzb(C, p)**2/(p["m_Z"]**2)*np.einsum('pr,st',zdl(C, p),zer(C, p))
C["VnueduLR"] = lambda C, p: -g2b(C, p)**2/(2*p["m_W"]**2)*np.einsum('pr,ts',wl(C, p),wr(C, p).conjugate())
#+ h.c.

C["V1uuLR"] = lambda C, p: C["qu1"]-gzb(C, p)**2/(p["m_Z"]**2)*np.einsum('pr,st',zul(C, p),zur(C, p))
C["V8uuLR"] = lambda C, p: C["qu8"]
C["V1udLR"] = lambda C, p: C["qd1"]-gzb(C, p)**2/(p["m_Z"]**2)*np.einsum('pr,st',zul(C, p),zdr(C, p))
C["V8udLR"] = lambda C, p: C["qd8"]
C["V1duLR"] = lambda C, p: C["qu1"]-gzb(C, p)**2/(p["m_Z"]**2)*np.einsum('pr,st',zdl(C, p),zur(C, p))
C["V8duLR"] = lambda C, p: C["qu8"]
C["V1ddLR"] = lambda C, p: C["qd1"]-gzb(C, p)**2/(p["m_Z"]**2)*np.einsum('pr,st',zdl(C, p),zdr(C, p))
C["V8ddLR"] = lambda C, p: C["qd8"]
C["V1udduLR"] = lambda C, p: -g2b(C, p)**2/(2*p["m_W"]**2)*np.einsum('pr,ts',wq(C, p),wr(C, p).conjugate())
C["V8udduLR"] = lambda C, p: np.zeros((3,3,3,3))


# Table 16
C["SeuRL"] = lambda C, p: np.zeros((3,3,3,3))
C["SedRL"] = lambda C, p: C["ledq"]
C["SnueduRL"] = lambda C, p: C["ledq"]


# Table 17
C["SeeRR"] = lambda C, p: np.zeros((3,3,3,3))

C["SeuRR"] = lambda C, p: -C["lequ1"]
C["TeuRR"] = lambda C, p: -C["lequ3"]
C["SedRR"] = lambda C, p: np.zeros((3,3,3,3))
C["TedRR"] = lambda C, p: np.zeros((3,3,3,3))
C["SnueduRR"] = lambda C, p: C["lequ1"]
C["TnueduRR"] = lambda C, p: C["lequ3"]

C["S1uuRR"] = lambda C, p: np.zeros((3,3,3,3))
C["S8uuRR"] = lambda C, p: np.zeros((3,3,3,3))
C["S1udRR"] = lambda C, p: C["quqd1"]
C["S8udRR"] = lambda C, p: C["quqd8"]
C["S1ddRR"] = lambda C, p: np.zeros((3,3,3,3))
C["S8ddRR"] = lambda C, p: np.zeros((3,3,3,3))
C["S1udduRR"] = lambda C, p: -np.einsum('stpr',C["quqd1"])
C["S8udduRR"] = lambda C, p: -np.einsum('stpr',C["quqd8"])

# # Table 18
# C["SnunuLL"] = lambda C, p: np.zeros((3,3,3,3))
#
# # Table 19
# C["SnueLL"] = lambda C, p: np.zeros((3,3,3,3))
# C["TnueLL"] = lambda C, p: np.zeros((3,3,3,3))
# C["SnueLR"] = lambda C, p: np.zeros((3,3,3,3))
#
# C["SnuuLL"] = lambda C, p: np.zeros((3,3,3,3))
# C["TnuuLL"] = lambda C, p: np.zeros((3,3,3,3))
# C["SnuuLR"] = lambda C, p: np.zeros((3,3,3,3))
# C["SnudLL"] = lambda C, p: np.zeros((3,3,3,3))
# C["TnudLL"] = lambda C, p: np.zeros((3,3,3,3))
# C["SnudLR"] = lambda C, p: np.zeros((3,3,3,3))
# C["SnueduLL"] = lambda C, p: np.zeros((3,3,3,3))
# C["TnueduLL"] = lambda C, p: np.zeros((3,3,3,3))
# C["SnueduLR"] = lambda C, p: np.zeros((3,3,3,3))
# C["VnueduRL"] = lambda C, p: np.zeros((3,3,3,3))
# C["VnueduRR"] = lambda C, p: np.zeros((3,3,3,3))

# Table 20
C["SuddLL"] = lambda C, p: -C["qqql"]-np.einsum('rpst',C["qqql"])
C["SduuLL"] = lambda C, p: -C["qqql"]-np.einsum('rpst',C["qqql"])
C["SuudLR"] = lambda C, p: np.zeros((3,3,3,3))
C["SduuLR"] = lambda C, p: -C["qque"]-np.einsum('rpst',C["qque"])
C["SuudRL"] = lambda C, p: np.zeros((3,3,3,3))
C["SduuRL"] = lambda C, p: C["duql"]
C["SdudRL"] = lambda C, p: -C["duql"]
C["SdduRL"] = lambda C, p: np.zeros((3,3,3,3))
C["SduuRR"] = lambda C, p: C["duue"]

# # Table 21
# C["SdddLL"] = lambda C, p: np.zeros((3,3,3,3))
# C["SuddLR"] = lambda C, p: np.zeros((3,3,3,3))
# C["SdduLR"] = lambda C, p: np.zeros((3,3,3,3))
# C["SdddLR"] = lambda C, p: np.zeros((3,3,3,3))
# C["SdddRL"] = lambda C, p: np.zeros((3,3,3,3))
# C["SuddRR"] = lambda C, p: np.zeros((3,3,3,3))
# C["SdddRR"] = lambda C, p: np.zeros((3,3,3,3))

def match_all_array(C_SMEFT, p):
    # generate a dictionary with 0 Wilson coefficients = Standard Model
    C_SMEFT_0 = {k: 0*v for k, v in C_SMEFT.items()}
    # compute the SMEFT matching contribution but subtract the SM part
    return {k: f(C_SMEFT, p) - f(C_SMEFT_0, p) for k, f in C.items()}

def match_all(d_SMEFT, parameters=None):
    """Match the SMEFT Warsaw basis onto the WET JMS basis."""
    p = default_parameters.copy()
    if parameters is not None:
        # if parameters are passed in, overwrite the default values
        p.update(parameters)
    C = wilson.translate.smeft.wcxf2arrays(d_SMEFT)
    C = smeftutil.symmetrize(C)
    C = smeftutil.scale_dict(C)
    C = smeftutil.add_missing(C)
    C['vT'] = 246.22
    C_WET = match_all_array(C, p)
    C_WET = wilson.translate.wet.rotate_down(C_WET, p)
    C_WET = wetutil.unscale_dict_wet(C_WET)
    d_WET = wilson.translate.smeft.arrays2wcxf(C_WET)
    basis = wcxf.Basis['WET', 'JMS']
    keys = set(d_WET.keys()) & set(basis.all_wcs)
    d_WET = {k: d_WET[k] for k in keys}
    return d_WET
