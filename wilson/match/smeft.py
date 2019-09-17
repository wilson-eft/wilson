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

def _match_all_array(C, p):

    # AUXILIARY FUNCTIONS

    # Eq. (6.4)
    GF = p['GF']
    GFx = GF - sqrt(2)/4 * ( -C['ll'][1,0,0,1] - C['ll'][0,1,1,0] + 2*C['phil3'][1,1] + 2*C['phil3'][0,0] )

    vT = sqrt(1/sqrt(2)/abs(GFx))
    eps = C["phiWB"] * vT**2
    g2b = 2*p["m_W"]/vT

    alpha_e = p['alpha_e']
    eb = sqrt(4*pi*alpha_e)
    g1b = eb*g2b/sqrt(g2b**2-eb**2) + eb**2*g2b/(g2b**2-eb**2) * eps

    sb = g1b/sqrt(g1b**2+g2b**2) * (1 + eps/2. * g2b/g1b * ((g2b**2-g1b**2)/(g1b**2+g2b**2)))
    cb = g2b/sqrt(g1b**2+g2b**2) * (1 - eps/2. * g1b/g2b * ((g2b**2-g1b**2)/(g1b**2+g2b**2)))

    # Eq. (2.26)
    eb = g2b*sb - 1/2.*cb*g2b* vT**2*C["phiWB"]
    gzb = eb/(sb*cb) * (1 + (g1b**2+g2b**2)/(2*g1b*g2b)*vT**2*C["phiWB"])

    # Eq. (2.30)
    wl = np.eye(3)+vT**2*C["phil3"]
    wq = np.eye(3)+vT**2*C["phiq3"]
    wr = 1/2.*vT**2*C["phiud"]
    znu = np.eye(3)*1/2.-1/2.*vT**2*C["phil1"]+1/2.*vT**2*C["phil3"]
    zel = np.eye(3)*(-1/2.+sb**2)-1/2.*vT**2*C["phil1"]-1/2.*vT**2*C["phil3"]
    zer = np.eye(3)*sb**2-1/2.*vT**2*C["phie"]
    zul = np.eye(3)*(1/2.-2./3*sb**2)-1/2.*vT**2*C["phiq1"]+1/2.*vT**2*C["phiq3"]
    zur = np.eye(3)*(-2./3)*sb**2-1/2.*vT**2*C["phiu"]
    zdl = np.eye(3)*(-1/2.+1/3.*sb**2)-1/2.*vT**2*C["phiq1"]-1/2.*vT**2*C["phiq3"]
    zdr = np.eye(3)*(1/3.)*sb**2-1/2.*vT**2*C["phid"]

    # MATCHING CONDITIONS
    c = {}

    # Table 9
    # c["nu"] = 1/2.*C["llphiphi"]* vT**2

    # Table 10
    # c["nugamma"] = np.zeros((3,3))

    # Table 11
    c["egamma"] = 1/sqrt(2) * (-C["eW"] * sb + C["eB"] * cb) * vT

    c["ugamma"] = 1/sqrt(2) * (C["uW"] * sb + C["uB"] * cb) * vT
    c["dgamma"] = 1/sqrt(2) * (-C["dW"] * sb + C["dB"] * cb) * vT
    c["uG"] = 1/sqrt(2) * C["uG"] * vT
    c["dG"] = 1/sqrt(2) * C["dG"] * vT

    #Table 12
    c["G"] = C["G"]
    c["Gtilde"] = C["Gtilde"]

    # Table 13
    c["VnunuLL"] = C["ll"]-gzb**2/(4*p["m_Z"]**2)*np.einsum('pr,st',znu,znu)-gzb**2/(4*p["m_Z"]**2)*np.einsum('pt,sr',znu,znu)
    c["VeeLL"] = C["ll"]-gzb**2/(4*p["m_Z"]**2)*np.einsum('pr,st',zel,zel)-gzb**2/(4*p["m_Z"]**2)*np.einsum('pt,sr',zel,zel)
    c["VnueLL"] = C["ll"]+np.einsum('stpr',C["ll"])-g2b**2/(2*p["m_W"]**2)*np.einsum('pt,rs',wl,wl.conjugate())-gzb**2/(p["m_Z"]**2)*np.einsum('pr,st',znu,zel)

    c["VnuuLL"] = C["lq1"]+C["lq3"]-gzb**2/(p["m_Z"]**2)*np.einsum('pr,st',znu,zul)
    c["VnudLL"] = C["lq1"]-C["lq3"]-gzb**2/(p["m_Z"]**2)*np.einsum('pr,st',znu,zdl)
    c["VeuLL"] = C["lq1"]-C["lq3"]-gzb**2/(p["m_Z"]**2)*np.einsum('pr,st',zel,zul)
    c["VedLL"] = C["lq1"]+C["lq3"]-gzb**2/(p["m_Z"]**2)*np.einsum('pr,st',zel,zdl)
    c["VnueduLL"] = 2*C["lq3"]-g2b**2/(2*p["m_W"]**2)*np.einsum('pr,ts',wl,wq.conjugate())
    # + h.c.

    c["VuuLL"] = C["qq1"]+C["qq3"]-gzb**2/(2*p["m_Z"]**2)*np.einsum('pr,st',zul,zul)
    c["VddLL"] = C["qq1"]+C["qq3"]-gzb**2/(2*p["m_Z"]**2)*np.einsum('pr,st',zdl,zdl)
    c["V1udLL"] = C["qq1"]+np.einsum('stpr',C["qq1"])-C["qq3"]-np.einsum('stpr',C["qq3"])+2/Nc*np.einsum('ptsr',C["qq3"])+2/Nc*np.einsum('srpt',C["qq3"])-g2b**2/(2*p["m_W"]**2)*np.einsum('pt,rs',wq,wq.conjugate())/Nc-gzb**2/(p["m_Z"]**2)*np.einsum('pr,st',zul,zdl)
    c["V8udLL"] = 4*np.einsum('ptsr',C["qq3"])+4*np.einsum('srpt',C["qq3"])-g2b**2/(p["m_W"]**2)*np.einsum('pt,rs',wq,wq.conjugate())


    # Table 14
    c["VeeRR"] = C["ee"]-gzb**2/(4*p["m_Z"]**2)*np.einsum('pr,st',zer,zer)-gzb**2/(4*p["m_Z"]**2)*np.einsum('pt,sr',zer,zer)

    c["VeuRR"] = C["eu"]-gzb**2/(p["m_Z"]**2)*np.einsum('pr,st',zer,zur)
    c["VedRR"] = C["ed"]-gzb**2/(p["m_Z"]**2)*np.einsum('pr,st',zer,zdr)

    c["VuuRR"] = C["uu"]-gzb**2/(2*p["m_Z"]**2)*np.einsum('pr,st',zur,zur)
    c["VddRR"] = C["dd"]-gzb**2/(2*p["m_Z"]**2)*np.einsum('pr,st',zdr,zdr)
    c["V1udRR"] = C["ud1"]-g2b**2/(2*p["m_W"]**2)*np.einsum('pt,rs',wr,wr.conjugate())/Nc-gzb**2/(p["m_Z"]**2)*np.einsum('pr,st',zur,zdr)
    c["V8udRR"] = C["ud8"]-g2b**2/(p["m_W"]**2)*np.einsum('pt,rs',wr,wr.conjugate())


    # Table 15
    c["VnueLR"] = C["le"]-gzb**2/(p["m_Z"]**2)*np.einsum('pr,st',znu,zer)
    c["VeeLR"] = C["le"]-gzb**2/(p["m_Z"]**2)*np.einsum('pr,st',zel,zer)

    c["VnuuLR"] = C["lu"]-gzb**2/(p["m_Z"]**2)*np.einsum('pr,st',znu,zur)
    c["VnudLR"] = C["ld"]-gzb**2/(p["m_Z"]**2)*np.einsum('pr,st',znu,zdr)
    c["VeuLR"] = C["lu"]-gzb**2/(p["m_Z"]**2)*np.einsum('pr,st',zel,zur)
    c["VedLR"] = C["ld"]-gzb**2/(p["m_Z"]**2)*np.einsum('pr,st',zel,zdr)
    c["VueLR"] = C["qe"]-gzb**2/(p["m_Z"]**2)*np.einsum('pr,st',zul,zer)
    c["VdeLR"] = C["qe"]-gzb**2/(p["m_Z"]**2)*np.einsum('pr,st',zdl,zer)
    c["VnueduLR"] = -g2b**2/(2*p["m_W"]**2)*np.einsum('pr,ts',wl,wr.conjugate())
    #+ h.c.

    c["V1uuLR"] = C["qu1"]-gzb**2/(p["m_Z"]**2)*np.einsum('pr,st',zul,zur)
    c["V8uuLR"] = C["qu8"]
    c["V1udLR"] = C["qd1"]-gzb**2/(p["m_Z"]**2)*np.einsum('pr,st',zul,zdr)
    c["V8udLR"] = C["qd8"]
    c["V1duLR"] = C["qu1"]-gzb**2/(p["m_Z"]**2)*np.einsum('pr,st',zdl,zur)
    c["V8duLR"] = C["qu8"]
    c["V1ddLR"] = C["qd1"]-gzb**2/(p["m_Z"]**2)*np.einsum('pr,st',zdl,zdr)
    c["V8ddLR"] = C["qd8"]
    c["V1udduLR"] = -g2b**2/(2*p["m_W"]**2)*np.einsum('pr,ts',wq,wr.conjugate())
    c["V8udduLR"] = np.zeros((3,3,3,3))


    # Table 16
    c["SeuRL"] = np.zeros((3,3,3,3))
    c["SedRL"] = C["ledq"]
    c["SnueduRL"] = C["ledq"]


    # Table 17
    c["SeeRR"] = np.zeros((3,3,3,3))

    c["SeuRR"] = -C["lequ1"]
    c["TeuRR"] = -C["lequ3"]
    c["SedRR"] = np.zeros((3,3,3,3))
    c["TedRR"] = np.zeros((3,3,3,3))
    c["SnueduRR"] = C["lequ1"]
    c["TnueduRR"] = C["lequ3"]

    c["S1uuRR"] = np.zeros((3,3,3,3))
    c["S8uuRR"] = np.zeros((3,3,3,3))
    c["S1udRR"] = C["quqd1"]
    c["S8udRR"] = C["quqd8"]
    c["S1ddRR"] = np.zeros((3,3,3,3))
    c["S8ddRR"] = np.zeros((3,3,3,3))
    c["S1udduRR"] = -np.einsum('stpr',C["quqd1"])
    c["S8udduRR"] = -np.einsum('stpr',C["quqd8"])

    # # Table 18
    # c["SnunuLL"] = np.zeros((3,3,3,3))
    #
    # # Table 19
    # c["SnueLL"] = np.zeros((3,3,3,3))
    # c["TnueLL"] = np.zeros((3,3,3,3))
    # c["SnueLR"] = np.zeros((3,3,3,3))
    #
    # c["SnuuLL"] = np.zeros((3,3,3,3))
    # c["TnuuLL"] = np.zeros((3,3,3,3))
    # c["SnuuLR"] = np.zeros((3,3,3,3))
    # c["SnudLL"] = np.zeros((3,3,3,3))
    # c["TnudLL"] = np.zeros((3,3,3,3))
    # c["SnudLR"] = np.zeros((3,3,3,3))
    # c["SnueduLL"] = np.zeros((3,3,3,3))
    # c["TnueduLL"] = np.zeros((3,3,3,3))
    # c["SnueduLR"] = np.zeros((3,3,3,3))
    # c["VnueduRL"] = np.zeros((3,3,3,3))
    # c["VnueduRR"] = np.zeros((3,3,3,3))

    # Table 20
    c["SuddLL"] = -C["qqql"]-np.einsum('rpst',C["qqql"])
    c["SduuLL"] = -C["qqql"]-np.einsum('rpst',C["qqql"])
    c["SuudLR"] = np.zeros((3,3,3,3))
    c["SduuLR"] = -C["qque"]-np.einsum('rpst',C["qque"])
    c["SuudRL"] = np.zeros((3,3,3,3))
    c["SduuRL"] = C["duql"]
    c["SdudRL"] = -C["duql"]
    c["SdduRL"] = np.zeros((3,3,3,3))
    c["SduuRR"] = C["duue"]

    # # Table 21
    # c["SdddLL"] = np.zeros((3,3,3,3))
    # c["SuddLR"] = np.zeros((3,3,3,3))
    # c["SdduLR"] = np.zeros((3,3,3,3))
    # c["SdddLR"] = np.zeros((3,3,3,3))
    # c["SdddRL"] = np.zeros((3,3,3,3))
    # c["SuddRR"] = np.zeros((3,3,3,3))
    # c["SdddRR"] = np.zeros((3,3,3,3))
    return c


def match_all_array(C_SMEFT, p):
    # generate a dictionary with 0 Wilson coefficients = Standard Model
    C_SMEFT_0 = {k: 0*v for k, v in C_SMEFT.items()}
    # compute the SMEFT matching contribution but subtract the SM part
    match_C = _match_all_array(C_SMEFT, p)
    match_C0 = _match_all_array(C_SMEFT_0, p)
    return {k: match_C[k] - match_C0[k] for k in match_C}


def match_all(d_SMEFT, parameters=None):
    """Match the SMEFT Warsaw basis onto the WET JMS basis."""
    p = default_parameters.copy()
    if parameters is not None:
        # if parameters are passed in, overwrite the default values
        p.update(parameters)
    C = wilson.util.smeftutil.wcxf2arrays_symmetrized(d_SMEFT)
    C['vT'] = 246.22
    C_WET = match_all_array(C, p)
    C_WET = wilson.translate.wet.rotate_down(C_WET, p)
    C_WET = wetutil.unscale_dict_wet(C_WET)
    d_WET = wilson.util.smeftutil.arrays2wcxf(C_WET)
    basis = wcxf.Basis['WET', 'JMS']
    keys = set(d_WET.keys()) & set(basis.all_wcs)
    d_WET = {k: d_WET[k] for k in keys}
    return d_WET
