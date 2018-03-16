from math import pi, sqrt
import numpy as np
from wilson.parameters import p as default_parameters
from wilson.util.qcd import alpha_s, m_b, m_s, m_c
import ckmutil.ckm, ckmutil.diag
import wcxf
import pkgutil
import json

# CONSTANTS

Nc = 3.
Qu = 2 / 3.
Qd = -1 / 3.


# flavour indices

dflav = {'d': 0, 's': 1, 'b': 2}
uflav = {'u': 0, 'c': 1}
lflav = {'e': 0, 'mu': 1, 'tau': 2}
llflav = {'e': 0, 'm': 1, 't': 2}

# WET with b,c,s,d,u

## Class I (DeltaF = 2)##

def _JMS_to_Bern_I(C, qq):
    """From JMS to BernI basis (= traditional SUSY basis in this case)
    for $\Delta F=2$ operators.
    `qq` should be 'sb', 'db', 'ds' or 'cu'"""
    if qq in ['sb', 'db', 'ds']:
        ij = tuple(dflav[q] for q in qq)
        ji = (ij[1], ij[0])
        return {
            '1' + 2 * qq : C["VddLL"][ij + ij],
            '2' + 2 * qq : C["S1ddRR"][ji + ji].conj()
                           - C["S8ddRR"][ji + ji].conj() / (2 * Nc),
            '3' + 2 * qq : C["S8ddRR"][ji + ji].conj() / 2,
            '4' + 2 * qq : -C["V8ddLR"][ij + ij],
            '5' + 2 * qq : -2 * C["V1ddLR"][ij + ij]
                           + C["V8ddLR"][ij + ij] / Nc,
            '1p' + 2 * qq : C["VddRR"][ij + ij],
            '2p' + 2 * qq : C["S1ddRR"][ij + ij]
                           - C["S8ddRR"][ij + ij] / (2 * Nc),
            '3p' + 2 * qq : C["S8ddRR"][ij + ij] / 2
                }
    elif qq == 'cu':
        ij = tuple(uflav[q] for q in qq)
        ji = (ij[1], ij[0])
        return {
            '1' + 2 * qq : C["VuuLL"][ij + ij],
            '2' + 2 * qq : C["S1uuRR"][ji + ji].conj()
                           - C["S8uuRR"][ji + ji].conj() / (2 * Nc),
            '3' + 2 * qq : C["S8uuRR"][ji + ji].conj() / 2,
            '4' + 2 * qq : -C["V8uuLR"][ij + ij],
            '5' + 2 * qq : -2 * C["V1uuLR"][ij + ij]
                           + C["V8uuLR"][ij + ij] / Nc,
            '1p' + 2 * qq : C["VuuRR"][ij + ij],
            '2p' + 2 * qq : C["S1uuRR"][ij + ij]
                            - C["S8uuRR"][ij + ij]/(2 * Nc),
            '3p' + 2 * qq : C["S8uuRR"][ij + ij] / 2
                }
    else:
        return "not in Bern_I"


def _BernI_to_Flavio_I(C, qq):
    """From BernI to FlavioI basis for $\Delta F=2$ operators.
    `qq` should be 'sb', 'db', 'ds' or 'uc'"""
    qqf = qq[::-1]  # flavio uses "bs" instead of "sb" etc.
    if qq in ['sb', 'db', 'ds', 'cu']:
        return {
            'CVLL_' + 2*qqf: C["1" + 2*qq],
            'CSLL_' + 2*qqf: C["2" + 2*qq] - 1 / 2 * C["3" + 2*qq],
            'CTLL_' + 2*qqf: -1 / 8 * C["3" + 2*qq],
            'CVLR_' + 2*qqf: -1 / 2 * C["5" + 2*qq],
            'CVRR_' + 2*qqf: C["1p" + 2*qq],
            'CSRR_' + 2*qqf: C["2p" + 2*qq] - 1 / 2 * C["3p" + 2*qq],
            'CTRR_' + 2*qqf: -1 / 8 * C["3p" + 2*qq],
            'CSLR_' + 2*qqf: C["4" + 2*qq]
            }
    else:
        return "not in Flavio_I"


def _FlavioI_to_Bern_I(C, qq):
    """From FlavioI to BernI basis for $\Delta F=2$ operators.
    `qq` should be 'sb', 'db', 'ds' or 'uc'"""
    qqb = qq[::-1]  # flavio uses "bs" instead of "sb" etc.
    if qq in ['bs', 'bd', 'sd', 'uc']:
        return {
            '1' + 2*qqb: C["CVLL_" + 2*qq],
            '2' + 2*qqb: C["CSLL_" + 2*qq] - 4 * C["CTLL_" + 2*qq],
            '3' + 2*qqb: -8 * C["CTLL_" + 2*qq],
            '4' + 2*qqb: C["CSLR_" + 2*qq],
            '5' + 2*qqb: -2 * C["CVLR_" + 2*qq],
            '1p' + 2*qqb: C["CVRR_" + 2*qq],
            '2p' + 2*qqb: C["CSRR_" + 2*qq] - 4 * C["CTRR_" + 2*qq],
            '3p' + 2*qqb: -8 * C["CTRR_" + 2*qq],
            }
    else:
        return "not in Bern_I"


def _BernI_to_FormFlavor_I(C, qq):
    """From BernI to FormFlavorI basis for $\Delta F=2$ operators.
    `qq` should be 'sb', 'db', 'ds' or 'uc'"""
    qqf = qq[::-1] # FormFlavour uses "bs" instead of "sb" etc.
    if qq in ['sb', 'db', 'ds']:
        return {
            'CVLL_' + 2*qqf: C["1" + 2*qq],
            'CSLL_' + 2*qqf: C["2" + 2*qq] + 1 / 2 * C["3" + 2*qq],
            'CTLL_' + 2*qqf: -1 / 8 * C["3" + 2*qq],
            'CVLR_' + 2*qqf: -1 / 2 * C["5" + 2*qq],
            'CVRR_' + 2*qqf: C["1p" + 2*qq],
            'CSRR_' + 2*qqf: C["2p" + 2*qq] + 1 / 2 * C["3p" + 2*qq],
            'CTRR_' + 2*qqf: -1 / 8 * C["3p" + 2*qq],
            'CSLR_' + 2*qqf: C["4" + 2*qq]
            }
    elif qq == 'cu':
        return {
            'CVLL_' + 2*qq: C["1" + 2*qq].conjugate(),
            'CSLL_' + 2*qq: C["2" + 2*qq] + 1 / 2 * C["3" + 2*qq].conjugate(),
            'CTLL_' + 2*qq: -1 / 8 * C["3" + 2*qq].conjugate(),
            'CVLR_' + 2*qq: -1 / 2 * C["5" + 2*qq].conjugate(),
            'CVRR_' + 2*qq: C["1p" + 2*qq].conjugate(),
            'CSRR_' + 2*qq: C["2p" + 2*qq].conjugate() + 1 / 2 * C["3p" + 2*qq].conjugate(),
            'CTRR_' + 2*qq: -1 / 8 * C["3p" + 2*qq],
            'CSLR_' + 2*qq: C["4" + 2*qq].conjugate()
            }
    else:
        raise ValueError("{} not in FormFlavor_I".format(qq))


## Class II ##

def _JMS_to_Bern_II(C, udlnu):
    """From JMS to BernII basis for charged current process semileptonic
    operators. `udlnu` should be of the form 'udl_enu_tau', 'cbl_munu_e' etc."""
    u = uflav[udlnu[0]]
    d = dflav[udlnu[1]]
    l = lflav[udlnu[4:udlnu.find('n')]]
    lp = lflav[udlnu[udlnu.find('_',5)+1:len(udlnu)]]
    ind = udlnu[0]+udlnu[1]+udlnu[4:udlnu.find('n')]+udlnu[udlnu.find('_',5)+1
                                                                    :len(udlnu)]
    return {
        '1' + ind : C["VnueduLL"][lp, l, d, u].conj(),
        '5' + ind : C["SnueduRL"][lp, l, d, u].conj(),
        '1p' + ind : C["VnueduLR"][lp, l, d, u].conj(),
        '5p' + ind : C["SnueduRR"][lp, l, d, u].conj(),
        '7p' + ind : C["TnueduRR"][lp, l, d, u].conj()
        }


def _BernII_to_Flavio_II(C, udlnu, parameters):
    """From BernII to FlavioII basis
    for charged current process semileptonic operators.
    `udlnu` should be of the form 'udl_enu_tau', 'cbl_munu_e' etc."""
    p = parameters
    u = uflav[udlnu[0]]
    d = dflav[udlnu[1]]
    l = lflav[udlnu[4:udlnu.find('n')]]
    lp = lflav[udlnu[udlnu.find('_',5)+1:len(udlnu)]]
    ind = udlnu[0]+udlnu[1]+udlnu[4:udlnu.find('n')]+udlnu[udlnu.find('_',5)+1
                                                                    :len(udlnu)]
    ind2 = udlnu[1]+udlnu[0]+udlnu[4:udlnu.find('n')]+'nu'+udlnu[
                                                udlnu.find('_',5)+1:len(udlnu)]
    dic = {
        'CVL_' + ind2 : C['1' + ind],
        'CVR_'+ ind2 : C['1p' + ind],
        'CSR_'+ ind2 : C['5' + ind],
        'CSL_'+ ind2 : C['5p' + ind],
        'CT_'+ ind2 : C['7p' + ind]
        }
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["gamma"])
    prefactor = -sqrt(2) / p['GF'] / V[u, d] / 4
    return {k: prefactor * v for k, v in dic.items()}


def _FlavioII_to_BernII(C, udlnu, parameters):
    """From FlavioII to BernII basis
    for charged current process semileptonic operators.
    `udlnu` should be of the form 'udl_enu_tau', 'cbl_munu_e' etc."""
    p = parameters
    u = uflav[udlnu[0]]
    d = dflav[udlnu[1]]
    l = lflav[udlnu[4:udlnu.find('n')]]
    lp = lflav[udlnu[udlnu.find('_',5)+1:len(udlnu)]]
    ind = udlnu[0]+udlnu[1]+udlnu[4:udlnu.find('n')]+udlnu[udlnu.find('_',5)+1
                                                                    :len(udlnu)]
    ind2 = udlnu[1]+udlnu[0]+udlnu[4:udlnu.find('n')]+'nu'+udlnu[
                                                udlnu.find('_',5)+1:len(udlnu)]
    dic = {
        '1' + ind: C['CVL_' + ind2],
        '1p' + ind: C['CVR_' + ind2],
        '5' + ind: C['CSR_' + ind2],
        '5p' + ind: C['CSL_' + ind2],
        '7p' + ind: C['CT_' + ind2],
        }
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["gamma"])
    prefactor = -sqrt(2) / p['GF'] / V[u, d] / 4
    return {k: v / prefactor for k, v in dic.items()}


def _BernII_to_EOS_II(C, udlnu, parameters):
    """From BernII to EOS  basis
    for charged current process semileptonic operators.
    `udlnu` should be of the form 'udl_enu_tau', 'cbl_munu_e' etc."""
    p = parameters
    u = uflav[udlnu[0]]
    d = dflav[udlnu[1]]
    l = lflav[udlnu[4:udlnu.find('n')]]
    lp = lflav[udlnu[udlnu.find('_',5)+1:len(udlnu)]]
    ind = udlnu[0]+udlnu[1]+udlnu[4:udlnu.find('n')]+udlnu[udlnu.find('_',5)+1
                                                                    :len(udlnu)]
    ind2 = udlnu[0]+udlnu[4:udlnu.find('n')]+'nu'+udlnu[
                                                udlnu.find('_',5)+1:len(udlnu)]
    dic = {
        'b->' + ind2 + '::cVL': C['1' + ind],
        'b->' + ind2 + '::cVR': C['1p' + ind],
        'b->' + ind2 + '::cSR': C['5' + ind],
        'b->' + ind2 + '::cSL': C['5p' + ind],
        'b->' + ind2 + '::cT': C['7p' + ind]
        }
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["gamma"])
    prefactor = -sqrt(2) / p['GF'] / V[u, d] / 4
    return {k: prefactor * v for k,v in dic.items()}


## Class III ##

def _JMS_to_Fierz_III_IV_V(C, qqqq):
    """From JMS to 4-quark Fierz basis for Classes III, IV and V.
    `qqqq` should be of the form 'sbuc', 'sdcc', 'ucuu' etc."""
    #case dduu
    classIII = ['sbuc', 'sbcu', 'dbuc', 'dbcu', 'dsuc', 'dscu']
    classVdduu = ['sbuu' , 'dbuu', 'dsuu', 'sbcc' , 'dbcc', 'dscc']
    if qqqq in classIII + classVdduu:
        f1 = dflav[qqqq[0]]
        f2 = dflav[qqqq[1]]
        f3 = uflav[qqqq[2]]
        f4 = uflav[qqqq[3]]
        return {
            'F' + qqqq + '1' : C["V1udLL"][f3, f4, f1, f2]
                                - C["V8udLL"][f3, f4, f1, f2] / (2 * Nc),
            'F' + qqqq + '2' : C["V8udLL"][f3, f4, f1, f2] / 2,
            'F' + qqqq + '3' : C["V1duLR"][f1, f2, f3, f4]
                                - C["V8duLR"][f1, f2, f3, f4] / (2 * Nc),
            'F' + qqqq + '4' : C["V8duLR"][f1, f2, f3, f4] / 2,
            'F' + qqqq + '5' : C["S1udRR"][f3, f4, f1, f2]
                                - C["S8udduRR"][f3, f2, f1, f4] / 4
                                - C["S8udRR"][f3, f4, f1, f2] / (2 * Nc),
            'F' + qqqq + '6' : -C["S1udduRR"][f3, f2, f1, f4] / 2
                                + C["S8udduRR"][f3, f2, f1, f4] /(4 * Nc)
                                + C["S8udRR"][f3, f4, f1, f2] / 2,
            'F' + qqqq + '7' : -C["V8udduLR"][f4, f1, f2, f3].conj(),
            'F' + qqqq + '8' : -2 * C["V1udduLR"][f4, f1, f2, f3].conj()
                                + C["V8udduLR"][f4, f1, f2, f3].conj() / Nc,
            'F' + qqqq + '9' : -C["S8udduRR"][f3, f2, f1, f4] / 16,
            'F' + qqqq + '10' : -C["S1udduRR"][f3, f2, f1, f4] / 8
                                + C["S8udduRR"][f3, f2, f1, f4] / (16 * Nc),
            'F' + qqqq + '1p' : C["V1udRR"][f3, f4, f1, f2]
                                - C["V8udRR"][f3, f4, f1, f2] / (2 * Nc),
            'F' + qqqq + '2p' : C["V8udRR"][f3, f4, f1, f2] / 2,
            'F' + qqqq + '3p' : C["V1udLR"][f3, f4, f1, f2]
                                - C["V8udLR"][f3, f4, f1, f2] / (2 * Nc),
            'F' + qqqq + '4p' : C["V8udLR"][f3, f4, f1, f2] / 2,
            'F' + qqqq + '5p' : C["S1udRR"][f4, f3, f2, f1].conj() -
                                C["S8udduRR"][f4, f1, f2, f3].conj() / 4
                                - C["S8udRR"][f4, f3, f2, f1].conj() / (2 * Nc),
            'F' + qqqq + '6p' : -C["S1udduRR"][f4, f1, f2, f3].conj() / 2 +
                                C["S8udduRR"][f4, f1, f2, f3].conj()/(4 * Nc)
                                + C["S8udRR"][f4, f3, f2, f1].conj() / 2,
            'F' + qqqq + '7p' : -C["V8udduLR"][f3, f2, f1, f4],
            'F' + qqqq + '8p' : - 2 * C["V1udduLR"][f3, f2, f1, f4]
                                + C["V8udduLR"][f3, f2, f1, f4] / Nc,
            'F' + qqqq + '9p' : -C["S8udduRR"][f4, f1, f2, f3].conj() / 16,
            'F' + qqqq + '10p' : -C["S1udduRR"][f4, f1, f2, f3].conj() / 8
                                + C["S8udduRR"][f4, f1, f2, f3].conj() / 16 / Nc
                            }
    #case dddd
    classIV = ['sbsd', 'dbds', 'bsbd']
    classVdddd = ['sbss', 'dbdd', 'dsdd', 'sbbb', 'dbbb', 'dsss']
    classVddddind = ['sbdd', 'dsbb', 'dbss']
    if qqqq in classIV + classVdddd + classVddddind:
        f1 = dflav[qqqq[0]]
        f2 = dflav[qqqq[1]]
        f3 = dflav[qqqq[2]]
        f4 = dflav[qqqq[3]]
        return {
                'F'+ qqqq +'1' : C["VddLL"][f3, f4, f1, f2],
                 'F'+ qqqq +'2' : C["VddLL"][f1, f4, f3, f2],
                 'F'+ qqqq +'3' : C["V1ddLR"][f1, f2, f3, f4]
                                  - C["V8ddLR"][f1, f2, f3, f4]/(2 * Nc),
                 'F'+ qqqq +'4' : C["V8ddLR"][f1, f2, f3, f4] / 2,
                 'F'+ qqqq +'5' : C["S1ddRR"][f3, f4, f1, f2]
                                  - C["S8ddRR"][f3, f2, f1,f4] / 4
                                  - C["S8ddRR"][f3, f4, f1, f2] / (2 * Nc),
                 'F'+ qqqq +'6' : -C["S1ddRR"][f1, f4, f3, f2] / 2
                                  + C["S8ddRR"][f3, f2, f1, f4] / (4 * Nc)
                                  + C["S8ddRR"][f3, f4, f1, f2] / 2,
                 'F'+ qqqq +'7' : -C["V8ddLR"][f1, f4, f3, f2],
                 'F'+ qqqq +'8' : -2 * C["V1ddLR"][f1, f4, f3, f2]
                                  + C["V8ddLR"][f1, f4, f3, f2] / Nc,
                 'F'+ qqqq +'9' : -C["S8ddRR"][f3, f2, f1, f4] / 16,
                 'F'+ qqqq +'10' : -C["S1ddRR"][f1, f4, f3, f2] / 8
                                   + C["S8ddRR"][f3, f2, f1, f4] / (16 * Nc),
                 'F'+ qqqq +'1p' : C["VddRR"][f3, f4, f1, f2],
                 'F'+ qqqq +'2p' : C["VddRR"][f1, f3, f4, f2],
                 'F'+ qqqq +'3p' : C["V1ddLR"][f3, f4, f1, f2]
                                   - C["V8ddLR"][f3, f4, f1,f2] / (2 * Nc),
                 'F'+ qqqq +'4p' : C["V8ddLR"][f3, f4, f1, f2] / 2,
                 'F'+ qqqq +'5p' : C["S1ddRR"][f4, f3, f2, f1].conj() -
                                   C["S8ddRR"][f4, f1, f2, f3].conj() / 4
                                   -C["S8ddRR"][f4, f3, f2, f1].conj() / 2 / Nc,
                 'F'+ qqqq +'6p' : -C["S1ddRR"][f4, f1, f2, f3].conj() / 2 +
                                    C["S8ddRR"][f4, f1, f2, f3].conj() / 4 / Nc
                                    + C["S8ddRR"][f4, f3, f2, f1].conj() / 2,
                 'F'+ qqqq +'7p' : -C["V8ddLR"][f3, f2, f1, f4],
                 'F'+ qqqq +'8p' : -2 * C["V1ddLR"][f3, f2, f1, f4]
                                    + C["V8ddLR"][f3, f2, f1, f4] / Nc,
                 'F'+ qqqq +'9p' : -C["S8ddRR"][f4, f1, f2, f3].conj() / 16,
                 'F'+ qqqq +'10p' : -C["S1ddRR"][f4, f1, f2, f3].conj() / 8 +
                                    C["S8ddRR"][f4, f1, f2, f3].conj() / 16 / Nc
                                    }
    #case uuuu
    classVuuuu = ['ucuu', 'cucc']
    if qqqq in classVuuuu:
        f1 = uflav[qqqq[0]]
        f2 = uflav[qqqq[1]]
        f3 = uflav[qqqq[2]]
        f4 = uflav[qqqq[3]]
        return {
                'F' + qqqq + '1' : C["VuuLL"][f3, f4, f1, f2],
                'F' + qqqq + '2' : C["VuuLL"][f1, f4, f3, f2],
                'F' + qqqq + '3' : C["V1uuLR"][f1, f2, f3, f4]
                                   - C["V8uuLR"][f1, f2, f3, f4] / (2 * Nc),
                'F' + qqqq + '4' : C["V8uuLR"][f1, f2, f3, f4] / 2,
                'F' + qqqq + '5' : C["S1uuRR"][f3, f4, f1, f2]
                                    - C["S8uuRR"][f3, f2, f1, f4] / 4
                                    - C["S8uuRR"][f3, f4, f1, f2] / (2 * Nc),
                'F' + qqqq + '6' : -C["S1uuRR"][f1, f4, f3, f2] / 2
                                    + C["S8uuRR"][f3, f2, f1, f4] / (4 * Nc)
                                    + C["S8uuRR"][f3, f4, f1, f2] / 2,
                'F' + qqqq + '7' : -C["V8uuLR"][f1, f4, f3, f2],
                'F' + qqqq + '8' : -2 * C["V1uuLR"][f1, f4, f3, f2]
                                    + C["V8uuLR"][f1, f4, f3, f2] / Nc,
                'F' + qqqq + '9' : -C["S8uuRR"][f3, f2, f1, f4] / 16,
                'F' + qqqq + '10' : -C["S1uuRR"][f1, f4, f3, f2] / 8
                                    + C["S8uuRR"][f3, f2, f1, f4] / (16 * Nc),
                'F'+ qqqq + '1p': C["VuuRR"][f3, f4, f1, f2],
                'F' + qqqq + '2p': C["VuuRR"][f1, f3, f4, f2],
                'F' + qqqq + '3p' : C["V1uuLR"][f3, f4, f1, f2]
                                    - C["V8uuLR"][f3, f4, f1,f2] / (2 * Nc),
                'F' + qqqq + '4p' : C["V8uuLR"][f3, f4, f1, f2] / 2,
                'F' + qqqq + '5p' : C["S1uuRR"][f4, f3, f2, f1].conj() -
                                    C["S8uuRR"][f4, f1, f2, f3].conj() / 4 -
                                    C["S8uuRR"][f4, f3, f2, f1].conj() / 2 / Nc,
                'F' + qqqq + '6p' : -C["S1uuRR"][f4, f1, f2, f3].conj() / 2 +
                                    C["S8uuRR"][f4, f1, f2, f3].conj() / 4 / Nc
                                    + C["S8uuRR"][f4, f3, f2, f1].conj() / 2,
                'F' + qqqq + '7p' : -C["V8uuLR"][f3, f2, f1, f4],
                'F' + qqqq + '8p' : -2 * C["V1uuLR"][f3, f2, f1, f4]
                                    + C["V8uuLR"][f3, f2, f1, f4] / Nc,
                'F' + qqqq + '9p' : -C["S8uuRR"][f4, f1, f2, f3].conj() / 16,
                'F' + qqqq + '10p' : -C["S1uuRR"][f4, f1, f2, f3].conj() / 8 +
                                    C["S8uuRR"][f4, f1, f2, f3].conj() / 16 / Nc
                                    }
    else:
        "not in Fqqqq"


def _Fierz_to_Bern_III_IV_V(Fqqqq, qqqq):
    """From Fierz to 4-quark Bern basis for Classes III, IV and V.
    `qqqq` should be of the form 'sbuc', 'sdcc', 'ucuu' etc."""
    # 2nd != 4th, color-octet redundant
    if qqqq in ['sbss', 'dbdd', 'dbds', 'sbsd', 'bsbd', 'dsdd']:
        return {
        '1' + qqqq : -Fqqqq['F' + qqqq + '1'] / 3
                        + 4 * Fqqqq['F' + qqqq + '3'] / 3,
        '3' + qqqq : Fqqqq['F' + qqqq + '1'] / 12 - Fqqqq['F' + qqqq + '3'] / 12,
        '5' + qqqq : -Fqqqq['F' + qqqq + '5p'] / 3
                    + 4 * Fqqqq['F' + qqqq + '7p'] / 3,
        '7' + qqqq : Fqqqq['F' + qqqq + '5p'] / 3 - Fqqqq['F' + qqqq + '7p'] / 3
                    + Fqqqq['F' + qqqq + '9p'],
        '9' + qqqq : Fqqqq['F' + qqqq + '5p'] / 48
                     - Fqqqq['F' + qqqq + '7p'] / 48,

        '1p' + qqqq : -Fqqqq['F' + qqqq + '1p'] / 3
                      + 4 * Fqqqq['F' + qqqq + '3p'] / 3,
        '3p' + qqqq : Fqqqq['F' + qqqq + '1p'] / 12
                      - Fqqqq['F' + qqqq + '3p'] / 12,
        '5p' + qqqq : -Fqqqq['F' + qqqq + '5'] / 3
                      + 4 * Fqqqq['F' + qqqq + '7'] / 3,
        '7p' + qqqq : Fqqqq['F' + qqqq + '5'] / 3 - Fqqqq['F' + qqqq + '7'] / 3
                      + Fqqqq['F' + qqqq + '9'],
        '9p' + qqqq : Fqqqq['F' + qqqq + '5'] / 48
                      - Fqqqq['F' + qqqq + '7'] / 48
                            }
    if qqqq in ['dbbb', 'sbbb', 'dsss']:  # 2nd = 4th, color-octet redundant
        return {
        '1' + qqqq : -Fqqqq['F' + qqqq + '1'] / 3
                        + 4 * Fqqqq['F' + qqqq + '3'] / 3,
        '3' + qqqq : Fqqqq['F' + qqqq + '1'] / 12 - Fqqqq['F' + qqqq + '3'] / 12,
        '5' + qqqq : -Fqqqq['F' + qqqq + '5'] / 3
                    + 4 * Fqqqq['F' + qqqq + '7'] / 3,
        '7' + qqqq : Fqqqq['F' + qqqq + '5'] / 3 - Fqqqq['F' + qqqq + '7'] / 3
                    + Fqqqq['F' + qqqq + '9'],
        '9' + qqqq : Fqqqq['F' + qqqq + '5'] / 48
                     - Fqqqq['F' + qqqq + '7'] / 48,

        '1p' + qqqq : -Fqqqq['F' + qqqq + '1p'] / 3
                      + 4 * Fqqqq['F' + qqqq + '3p'] / 3,
        '3p' + qqqq : Fqqqq['F' + qqqq + '1p'] / 12
                      - Fqqqq['F' + qqqq + '3p'] / 12,
        '5p' + qqqq : -Fqqqq['F' + qqqq + '5p'] / 3
                      + 4 * Fqqqq['F' + qqqq + '7p'] / 3,
        '7p' + qqqq : Fqqqq['F' + qqqq + '5p'] / 3 - Fqqqq['F' + qqqq + '7p'] / 3
                      + Fqqqq['F' + qqqq + '9p'],
        '9p' + qqqq : Fqqqq['F' + qqqq + '5p'] / 48
                      - Fqqqq['F' + qqqq + '7p'] / 48
                            }
    # generic case
    if qqqq in ['sbuu', 'sbdd', 'sbuu', 'sbuc', 'sbcu', 'sbcc',
                'dbuu', 'dbss', 'dbuu', 'dbuc', 'dbcu', 'dbcc',
                'dsuu', 'dsbb', 'dsuu', 'dsuc', 'dscu', 'dscc',]:
        return {
        '1'+qqqq : -Fqqqq['F' + qqqq + '1']/3 + 4 * Fqqqq['F' + qqqq + '3'] / 3
                   - Fqqqq['F' + qqqq + '2']/(3 * Nc)
                   + 4 * Fqqqq['F' + qqqq + '4'] / (3 * Nc),
        '2'+qqqq : -2 * Fqqqq['F' + qqqq + '2'] / 3
                    + 8 * Fqqqq['F' + qqqq + '4'] / 3,
        '3'+qqqq : Fqqqq['F' + qqqq + '1'] / 12
                   - Fqqqq['F' + qqqq + '3'] / 12
                   + Fqqqq['F' + qqqq + '2'] / (12 * Nc)
                   - Fqqqq['F' + qqqq + '4'] / (12 * Nc),
        '4'+ qqqq : Fqqqq['F' + qqqq + '2'] / 6 - Fqqqq['F' + qqqq + '4'] / 6,
        '5'+ qqqq : -Fqqqq['F' + qqqq + '5'] / 3
                    + 4 * Fqqqq['F' + qqqq + '7'] / 3
                    - Fqqqq['F' + qqqq + '6']/(3 * Nc)
                    + 4 * Fqqqq['F' + qqqq + '8']/(3 * Nc),
        '6'+qqqq : -2 * Fqqqq['F' + qqqq + '6'] / 3
                   + 8 * Fqqqq['F' + qqqq + '8'] / 3,
        '7'+qqqq : Fqqqq['F' + qqqq + '5'] / 3 - Fqqqq['F' + qqqq + '7'] / 3
                   + Fqqqq['F' + qqqq + '9'] + Fqqqq['F' + qqqq + '10'] / Nc
                   + Fqqqq['F' + qqqq + '6']/(3 * Nc)
                   - Fqqqq['F' + qqqq + '8']/(3 * Nc),
        '8'+qqqq : 2*Fqqqq['F' + qqqq + '10'] + 2 * Fqqqq['F' + qqqq + '6'] / 3
                   -2 * Fqqqq['F' + qqqq + '8'] / 3,
        '9'+qqqq : Fqqqq['F' + qqqq + '5'] / 48 - Fqqqq['F' + qqqq + '7'] / 48
                   + Fqqqq['F' + qqqq + '6'] / (48 * Nc)
                   - Fqqqq['F' + qqqq + '8'] / (48 * Nc),
        '10'+qqqq : Fqqqq['F' + qqqq + '6'] / 24 - Fqqqq['F' + qqqq + '8'] / 24,
        '1p'+qqqq : -Fqqqq['F' + qqqq + '1p'] / 3
                    + 4 * Fqqqq['F' + qqqq + '3p'] / 3
                    - Fqqqq['F' + qqqq + '2p'] / (3 * Nc)
                    + 4 * Fqqqq['F' + qqqq + '4p'] / (3 * Nc),
        '2p'+qqqq : -2 * Fqqqq['F' + qqqq + '2p'] / 3
                    + 8 * Fqqqq['F' + qqqq + '4p'] / 3,
        '3p'+qqqq : Fqqqq['F' + qqqq + '1p'] / 12
                    - Fqqqq['F' + qqqq + '3p'] / 12
                    + Fqqqq['F' + qqqq + '2p'] / (12 * Nc)
                    - Fqqqq['F' + qqqq + '4p'] / (12 * Nc),
        '4p'+qqqq : Fqqqq['F' + qqqq + '2p'] / 6 - Fqqqq['F' + qqqq + '4p'] / 6,
        '5p'+qqqq : -Fqqqq['F' + qqqq + '5p'] / 3
                    + 4 * Fqqqq['F' + qqqq + '7p'] / 3
                    - Fqqqq['F' + qqqq + '6p'] / (3 * Nc)
                    + 4 * Fqqqq['F' + qqqq + '8p'] / (3 * Nc),
        '6p'+qqqq : -2 * Fqqqq['F' + qqqq + '6p'] / 3
                    + 8 * Fqqqq['F' + qqqq + '8p'] / 3,
        '7p'+qqqq : Fqqqq['F' + qqqq + '5p'] / 3 - Fqqqq['F' + qqqq + '7p'] / 3
                    + Fqqqq['F' + qqqq + '9p'] + Fqqqq['F' + qqqq + '10p'] / Nc
                    + Fqqqq['F' + qqqq + '6p']/(3 * Nc)
                    - Fqqqq['F' + qqqq + '8p']/(3 * Nc),
        '8p'+qqqq : 2 * Fqqqq['F' + qqqq + '10p']
                    + 2 * Fqqqq['F' + qqqq + '6p'] / 3
                    - 2 * Fqqqq['F' + qqqq + '8p'] / 3,
        '9p'+qqqq : Fqqqq['F' + qqqq + '5p'] / 48
                    - Fqqqq['F' + qqqq + '7p'] / 48
                    + Fqqqq['F' + qqqq + '6p'] / (48 * Nc)
                    - Fqqqq['F' + qqqq + '8p'] / (48 * Nc),
        '10p'+qqqq : Fqqqq['F' + qqqq + '6p'] / 24
                     - Fqqqq['F' + qqqq + '8p'] / 24
                    }
    raise ValueError("Case not implemented: {}".format(qqqq))

def _Bern_to_Fierz_III_IV_V(C, qqqq):
    """From Bern to 4-quark Fierz basis for Classes III, IV and V.
    `qqqq` should be of the form 'sbuc', 'sdcc', 'ucuu' etc."""
    # 2nd != 4th, color-octet redundant
    if qqqq in ['sbss', 'dbdd', 'dbds', 'sbsd', 'bsbd', 'dsdd']:
        return {
                'F' + qqqq + '1': C['1' + qqqq] + 16 * C['3' + qqqq],
                'F' + qqqq + '1p': C['1p' + qqqq] + 16 * C['3p' + qqqq],
                'F' + qqqq + '3': C['1' + qqqq] + 4 * C['3' + qqqq],
                'F' + qqqq + '3p': C['1p' + qqqq] + 4 * C['3p' + qqqq],
                'F' + qqqq + '5': C['5p' + qqqq] + 64 * C['9p' + qqqq],
                'F' + qqqq + '5p': C['5' + qqqq] + 64 * C['9' + qqqq],
                'F' + qqqq + '7': C['5p' + qqqq] + 16 * C['9p' + qqqq],
                'F' + qqqq + '7p': C['5' + qqqq] + 16 * C['9' + qqqq],
                'F' + qqqq + '9': C['7p' + qqqq] - 16 * C['9p' + qqqq],
                'F' + qqqq + '9p': C['7' + qqqq] - 16 * C['9' + qqqq],
                }
    if qqqq in ['dbbb', 'sbbb', 'dsss']:  # 2nd = 4th, color-octet redundant
        return {
                'F' + qqqq + '1': C['1' + qqqq] + 16 * C['3' + qqqq],
                'F' + qqqq + '1p': C['1p' + qqqq] + 16 * C['3p' + qqqq],
                'F' + qqqq + '3': C['1' + qqqq] + 4 * C['3' + qqqq],
                'F' + qqqq + '3p': C['1p' + qqqq] + 4 * C['3p' + qqqq],
                'F' + qqqq + '5': C['5' + qqqq] + 64 * C['9' + qqqq],
                'F' + qqqq + '5p': C['5p' + qqqq] + 64 * C['9p' + qqqq],
                'F' + qqqq + '7': C['5' + qqqq] + 16 * C['9' + qqqq],
                'F' + qqqq + '7p': C['5p' + qqqq] + 16 * C['9p' + qqqq],
                'F' + qqqq + '9': C['7' + qqqq] - 16 * C['9' + qqqq],
                'F' + qqqq + '9p': C['7p' + qqqq] - 16 * C['9p' + qqqq],
                }
    # generic case
    if qqqq in ['sbuu', 'sbdd', 'sbuu', 'sbuc', 'sbcu', 'sbcc',
                'dbuu', 'dbss', 'dbuu', 'dbuc', 'dbcu', 'dbcc',
                'dsuu', 'dsbb', 'dsuu', 'dsuc', 'dscu', 'dscc',]:
        return {
                'F' + qqqq + '1': C['1' + qqqq] - C['2' + qqqq] / 6 + 16 * C['3' + qqqq] - (8 * C['4' + qqqq]) / 3,
                'F' + qqqq + '10': -8 * C['10' + qqqq] + C['8' + qqqq] / 2,
                'F' + qqqq + '10p': -8 * C['10p' + qqqq] + C['8p' + qqqq] / 2,
                'F' + qqqq + '1p': C['1p' + qqqq] - C['2p' + qqqq] / 6 + 16 * C['3p' + qqqq] - (8 * C['4p' + qqqq]) / 3,
                'F' + qqqq + '2': C['2' + qqqq] / 2 + 8 * C['4' + qqqq],
                'F' + qqqq + '2p': C['2p' + qqqq] / 2 + 8 * C['4p' + qqqq],
                'F' + qqqq + '3': C['1' + qqqq] - C['2' + qqqq] / 6 + 4 * C['3' + qqqq] - (2 * C['4' + qqqq]) / 3,
                'F' + qqqq + '3p': C['1p' + qqqq] - C['2p' + qqqq] / 6 + 4 * C['3p' + qqqq] - (2 * C['4p' + qqqq]) / 3,
                'F' + qqqq + '4': C['2' + qqqq] / 2 + 2 * C['4' + qqqq],
                'F' + qqqq + '4p': C['2p' + qqqq] / 2 + 2 * C['4p' + qqqq],
                'F' + qqqq + '5': -((32 * C['10' + qqqq]) / 3) + C['5' + qqqq] - C['6' + qqqq] / 6 + 64 * C['9' + qqqq],
                'F' + qqqq + '5p': -((32 * C['10p' + qqqq]) / 3) + C['5p' + qqqq] - C['6p' + qqqq] / 6 + 64 * C['9p' + qqqq],
                'F' + qqqq + '6': 32 * C['10' + qqqq] + C['6' + qqqq] / 2,
                'F' + qqqq + '6p': 32 * C['10p' + qqqq] + C['6p' + qqqq] / 2,
                'F' + qqqq + '7': -((8 * C['10' + qqqq]) / 3) + C['5' + qqqq] - C['6' + qqqq] / 6 + 16 * C['9' + qqqq],
                'F' + qqqq + '7p': -((8 * C['10p' + qqqq]) / 3) + C['5p' + qqqq] - C['6p' + qqqq] / 6 + 16 * C['9p' + qqqq],
                'F' + qqqq + '8': 8 * C['10' + qqqq] + C['6' + qqqq] / 2,
                'F' + qqqq + '8p': 8 * C['10p' + qqqq] + C['6p' + qqqq] / 2,
                'F' + qqqq + '9': (8 * C['10' + qqqq]) / 3 + C['7' + qqqq] - C['8' + qqqq] / 6 - 16 * C['9' + qqqq],
                'F' + qqqq + '9p': (8 * C['10p' + qqqq]) / 3 + C['7p' + qqqq] - C['8p' + qqqq] / 6 - 16 * C['9p' + qqqq],
                }
    raise ValueError("Case not implemented: {}".format(qqqq))

def _Fierz_to_Flavio_V(Fqqqq, qqqq, parameters):
    p = parameters
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["gamma"])
    if qqqq[:2] == 'sb':
        xi = V[2, 2] * V[2, 1].conj()
    elif qqqq[:2] == 'db':
        xi = V[2, 2] * V[2, 0].conj()
    elif qqqq[:2] == 'ds':
        xi = V[2, 1] * V[2, 0].conj()
    else:
        raise ValueError("Unexpected flavours: {}".format(qqqq[:2]))
    pf = sqrt(2) / p['GF'] / xi / 4
    qqqq_fl = qqqq[1] + qqqq[0] + qqqq[2:]  # 1st two indices flipped for flavio
    if qqqq in ['dsss', 'dsdd', 'dbbb', 'dbdd', 'sbss', 'sbbb']:
        return {
            'CVLL_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '1'],
            'CVLR_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '3'],
            'CSRR_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '5'],
            'CSRL_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '7'],
            'CTRR_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '9'],
            'CVRR_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '1p'],
            'CVRL_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '3p'],
            'CSLL_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '5p'],
            'CSLR_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '7p'],
            'CTLL_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '9p'],
        }
    elif qqqq in ['dsuu', 'dscc', 'dsbb', 'dbuu', 'dbcc', 'dbss', 'sbuu', 'sbcc', 'sbdd']:
        return {
            'CVLL_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '1'],
            'CVLLt_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '2'],
            'CVLR_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '3'],
            'CVLRt_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '4'],
            'CSRR_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '5'],
            'CSRRt_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '6'],
            'CSRL_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '7'],
            'CSRLt_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '8'],
            'CTRR_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '9'],
            'CTRRt_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '10'],
            'CVRR_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '1p'],
            'CVRRt_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '2p'],
            'CVRL_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '3p'],
            'CVRLt_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '4p'],
            'CSLL_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '5p'],
            'CSLLt_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '6p'],
            'CSLR_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '7p'],
            'CSLRt_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '8p'],
            'CTLL_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '9p'],
            'CTLLt_' + qqqq_fl: pf * Fqqqq['F' + qqqq + '10p'],
        }
    else:
        raise ValueError("Sector not implemented: {}".format(qqqq))


def _Flavio_to_Fierz_V(Cqqqq, qqqq, parameters):
    p = parameters
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["gamma"])
    if qqqq[:2] == 'sb':
        xi = V[2, 2] * V[2, 1].conj()
    elif qqqq[:2] == 'db':
        xi = V[2, 2] * V[2, 0].conj()
    elif qqqq[:2] == 'ds':
        xi = V[2, 1] * V[2, 0].conj()
    else:
        raise ValueError("Unexpected flavours: {}".format(qqqq[:2]))
    pf = 4 * p['GF'] /sqrt(2) * xi
    qqqq_fl = qqqq[1] + qqqq[0] + qqqq[2:]  # 1st two indices flipped for flavio
    if qqqq in ['dsss', 'dsdd', 'dbbb', 'dbdd', 'sbss', 'sbbb']:
        return {
            'F' + qqqq + '1': pf * Cqqqq['CVLL_' + qqqq_fl],
            'F' + qqqq + '3': pf * Cqqqq['CVLR_' + qqqq_fl],
            'F' + qqqq + '5': pf * Cqqqq['CSRR_' + qqqq_fl],
            'F' + qqqq + '7': pf * Cqqqq['CSRL_' + qqqq_fl],
            'F' + qqqq + '9': pf * Cqqqq['CTRR_' + qqqq_fl],
            'F' + qqqq + '1p': pf * Cqqqq['CVRR_' + qqqq_fl],
            'F' + qqqq + '3p': pf * Cqqqq['CVRL_' + qqqq_fl],
            'F' + qqqq + '5p': pf * Cqqqq['CSLL_' + qqqq_fl],
            'F' + qqqq + '7p': pf * Cqqqq['CSLR_' + qqqq_fl],
            'F' + qqqq + '9p': pf * Cqqqq['CTLL_' + qqqq_fl],
        }
    elif qqqq in ['dsuu', 'dscc', 'dsbb', 'dbuu', 'dbcc', 'dbss', 'sbuu', 'sbcc', 'sbdd']:
        return {
            'F' + qqqq + '1': pf * Cqqqq['CVLL_' + qqqq_fl],
            'F' + qqqq + '2': pf * Cqqqq['CVLLt_' + qqqq_fl],
            'F' + qqqq + '3': pf * Cqqqq['CVLR_' + qqqq_fl],
            'F' + qqqq + '4': pf * Cqqqq['CVLRt_' + qqqq_fl],
            'F' + qqqq + '5': pf * Cqqqq['CSRR_' + qqqq_fl],
            'F' + qqqq + '6': pf * Cqqqq['CSRRt_' + qqqq_fl],
            'F' + qqqq + '7': pf * Cqqqq['CSRL_' + qqqq_fl],
            'F' + qqqq + '8': pf * Cqqqq['CSRLt_' + qqqq_fl],
            'F' + qqqq + '9': pf * Cqqqq['CTRR_' + qqqq_fl],
            'F' + qqqq + '10': pf * Cqqqq['CTRRt_' + qqqq_fl],
            'F' + qqqq + '1p': pf * Cqqqq['CVRR_' + qqqq_fl],
            'F' + qqqq + '2p': pf * Cqqqq['CVRRt_' + qqqq_fl],
            'F' + qqqq + '3p': pf * Cqqqq['CVRL_' + qqqq_fl],
            'F' + qqqq + '4p': pf * Cqqqq['CVRLt_' + qqqq_fl],
            'F' + qqqq + '5p': pf * Cqqqq['CSLL_' + qqqq_fl],
            'F' + qqqq + '6p': pf * Cqqqq['CSLLt_' + qqqq_fl],
            'F' + qqqq + '7p': pf * Cqqqq['CSLR_' + qqqq_fl],
            'F' + qqqq + '8p': pf * Cqqqq['CSLRt_' + qqqq_fl],
            'F' + qqqq + '9p': pf * Cqqqq['CTLL_' + qqqq_fl],
            'F' + qqqq + '10p': pf * Cqqqq['CTLLt_' + qqqq_fl],
        }
    else:
        raise ValueError("Sector not implemented: {}".format(qqqq))


def _Fierz_to_EOS_V(Fsbuu,Fsbdd,Fsbcc,Fsbss,Fsbbb,parameters):
    p = parameters
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["gamma"])
    Vtb = V[2,2]
    Vts = V[2,1]
    """From Fierz to the EOS basis for b -> s transitions.
    The arguments are dictionaries of the corresponding Fierz bases """
    dic = {
    'b->s::c1' :  -2 * Fsbbb['Fsbbb1'] / 5 + Fsbcc['Fsbcc1'] -
                  6 * Fsbdd['Fsbdd1'] / 5 + 4 * Fsbdd['Fsbdd2'] / 5
                  -2 * Fsbss['Fsbss1'] / 5 + Fsbuu['Fsbuu1'],
     'b->s::c2' : -4 * Fsbbb['Fsbbb1'] / 15 + Fsbcc['Fsbcc1'] / 6
                  + Fsbcc['Fsbcc2'] / 2 + Fsbdd['Fsbdd1'] / 5
                  - 7 * Fsbdd['Fsbdd2'] / 15 - 4 * Fsbss['Fsbss1'] / 15
                  + Fsbuu['Fsbuu1'] / 6 + Fsbuu['Fsbuu2'] / 2,
     'b->s::c3' : -4 * Fsbbb['Fsbbb1'] / 45 + 4 * Fsbbb['Fsbbb3'] / 15 +
                    4 * Fsbbb['Fsbbb4'] / 45 + 4 * Fsbcc['Fsbcc3'] / 15 +
                    4 * Fsbcc['Fsbcc4'] / 45 - 7 * Fsbdd['Fsbdd1'] / 45
                    + Fsbdd['Fsbdd2'] / 15 + 4 * Fsbdd['Fsbdd3'] / 15
                    + 4 * Fsbdd['Fsbdd4'] / 45 - 4 * Fsbss['Fsbss1'] / 45
                    + 4 * Fsbss['Fsbss3'] / 15 + 4 * Fsbss['Fsbss4'] / 45
                    + 4 * Fsbuu['Fsbuu3'] / 15 + 4 * Fsbuu['Fsbuu4'] / 45,
     'b->s::c4' : - 2 * Fsbbb['Fsbbb1'] / 15 + 8 * Fsbbb['Fsbbb4'] / 15
                    + 8 * Fsbcc['Fsbcc4'] / 15 + 4 * Fsbdd['Fsbdd1'] / 15
                    - 2 * Fsbdd['Fsbdd2'] / 5 + 8 * Fsbdd['Fsbdd4'] / 15
                    - 2 * Fsbss['Fsbss1'] / 15 + 8 * Fsbss['Fsbss4'] / 15
                    + 8 * Fsbuu['Fsbuu4'] / 15,
     'b->s::c5' : Fsbbb['Fsbbb1'] / 45 - Fsbbb['Fsbbb3'] / 60
                  -Fsbbb['Fsbbb4'] / 180 - Fsbcc['Fsbcc3'] / 60
                  - Fsbcc['Fsbcc4'] / 180 + 7 * Fsbdd['Fsbdd1'] / 180
                  - Fsbdd['Fsbdd2'] / 60 - Fsbdd['Fsbdd3'] / 60
                  -Fsbdd['Fsbdd4'] / 180 + Fsbss['Fsbss1'] / 45
                  - Fsbss['Fsbss3'] / 60 -Fsbss['Fsbss4'] / 180
                  - Fsbuu['Fsbuu3'] / 60 - Fsbuu['Fsbuu4'] / 180,
     'b->s::c6' : Fsbbb['Fsbbb1'] / 30 - Fsbbb['Fsbbb4'] / 30
                  -Fsbcc['Fsbcc4'] / 30 - Fsbdd['Fsbdd1'] / 15
                  + Fsbdd['Fsbdd2'] / 10 -Fsbdd['Fsbdd4'] / 30
                  + Fsbss['Fsbss1'] / 30 - Fsbss['Fsbss4'] / 30
                  -Fsbuu['Fsbuu4'] / 30
                  }
    prefactor = sqrt(2)/p['GF']/Vtb/Vts.conj()/4
    return {k: prefactor * v for k,v in dic.items()}


# semileptonic operators
# arguments are of the form ddl_lnu_l' to simplify the notation

def JMS_to_Fierz_lep(C, ddll):
    """From JMS to semileptonic Fierz basis for Class V.
    `ddll` should be of the form 'sbl_enu_tau', 'dbl_munu_e' etc."""
    s = dflav[ddll[0]]
    b = dflav[ddll[1]]
    l = lflav[ddll[4:ddll.find('n')]]
    lp = lflav[ddll[ddll.find('_',5)+1:len(ddll)]]
    ind = ddll.replace('l_','').replace('nu_','')
    return {
        'F' + ind + '9' : C["VdeLR"][s, b, l, lp] / 2
                          + C["VedLL"][l, lp, s, b] / 2,
        'F' + ind + '10' : C["VdeLR"][s, b, l, lp] / 2
                           - C["VedLL"][l, lp, s, b] / 2,
        'F' + ind + 'S' : C["SedRL"][lp, l, b, s].conj() / 2
                          + C["SedRR"][l, lp, s, b] / 2,
        'F' + ind + 'P' : - C["SedRL"][lp, l, b, s].conj() / 2
                          + C["SedRR"][l, lp, s, b] / 2,
        'F' + ind + 'T' : C["TedRR"][l, lp, s, b] / 2
                          + C["TedRR"][lp, l, b, s].conj() / 2,
        'F' + ind + 'T5' : C["TedRR"][l, lp, s, b] / 2
                           - C["TedRR"][lp, l, b, s].conj() / 2,
        'F' + ind + '9p' : C["VedLR"][l, lp, s, b] / 2
                           + C["VedRR"][l, lp, s, b] / 2,
        'F' + ind + '10p' : -C["VedLR"][l, lp, s, b] / 2
                            + C["VedRR"][l, lp, s, b] / 2,
        'F' + ind + 'Sp' : C["SedRL"][l, lp, s, b] / 2
                           + C["SedRR"][lp, l, b, s].conj() / 2,
        'F' + ind + 'Pp' : C["SedRL"][l, lp, s, b] / 2
                            - C["SedRR"][lp, l, b, s].conj() / 2,
        'F' + ind + 'nu' : C["VnudLL"][l, lp, s, b],
        'F' + ind + 'nup' : C["VnudLR"][l, lp, s, b]
        }


def Fierz_to_Bern_lep(C, ddll, include_charged=True):
    """From semileptonic Fierz basis to Bern semileptonic basis for Class V.
    C should be the corresponding leptonic Fierz basis and
    `ddll` should be of the form 'sbl_enu_tau', 'dbl_munu_e' etc."""
    ind = ddll.replace('l_','').replace('nu_','')
    dic =  {'nu1' + ind : C['F' + ind + 'nu'],
            'nu1p' + ind : C['F' + ind + 'nup']
            }
    if include_charged:
        dic.update({
            '1' + ind : 5 * C['F'+ ind + '10'] / 3 + C['F'+ ind + '9'],
            '3' + ind : -C['F' + ind + '10'] / 6,
            '5' + ind : C['F' + ind + 'S'] - 5 * C['F' + ind + 'P'] / 3,
            '7' + ind : 2 * C['F' + ind + 'P'] / 3 + C['F' + ind + 'T']
                        + C['F' + ind + 'T5'],
            '9' + ind : C['F' + ind + 'P'] / 24,
            '1p' + ind : C['F' + ind + '9p'] - 5 * C['F' + ind + '10p'] / 3,
            '3p' + ind : C['F' + ind + '10p'] / 6,
            '5p' + ind : 5 * C['F' + ind + 'Pp'] / 3 + C['F' + ind + 'Sp'],
            '7p' + ind : -2 * C['F' + ind + 'Pp'] / 3 + C['F' + ind + 'T']
                        - C['F' + ind + 'T5'],
            '9p' + ind : -C['F' + ind + 'Pp'] / 24,
            'nu1' + ind : C['F' + ind + 'nu'],
            'nu1p' + ind : C['F' + ind + 'nup']
            })
    return dic


def Bern_to_Fierz_lep(C,ddll):
    """From semileptonic Bern basis to Fierz semileptonic basis for Class V.
    C should be the corresponding leptonic Fierz basis and
    `ddll` should be of the form 'sbl_enu_tau', 'dbl_munu_e' etc."""
    ind = ddll.replace('l_','').replace('nu_','')
    return {'F' + ind + '9': C['1' + ind] + 10 * C['3' + ind],
            'F' + ind + '10': -6 * C['3' + ind],
            'F' + ind + 'S': C['5' + ind] + 40 * C['9' + ind],
            'F' + ind + 'P': 24 * C['9' + ind],
            'F' + ind + 'T': C['7' + ind] / 2 + C['7p' + ind] / 2 - 8 * C['9' + ind] - 8 * C['9p' + ind],
            'F' + ind + 'T5': C['7' + ind] / 2 - C['7p' + ind] / 2 - 8 * C['9' + ind] + 8 * C['9p' + ind],
            'F' + ind + '9p': C['1p' + ind] + 10 * C['3p' + ind],
            'F' + ind + '10p': 6 * C['3p' + ind],
            'F' + ind + 'Sp': C['5p' + ind] + 40 * C['9p' + ind],
            'F' + ind + 'Pp': -24 * C['9p' + ind],
            'F' + ind + 'nu': C['nu1' + ind],
            'F' + ind + 'nup': C['nu1p' + ind],
            }


def Fierz_to_Flavio_lep(C, ddll, parameters, include_charged=True, norm_gf=True):
    """From semileptonic Fierz basis to Flavio semileptonic basis for Class V.
    C should be the corresponding leptonic Fierz basis and
    `ddll` should be of the form 'sbl_enu_tau', 'dbl_munu_e' etc."""
    p = parameters
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["gamma"])
    if ddll[:2] == 'sb':
        xi = V[2, 2] * V[2, 1].conj()
    elif ddll[:2] == 'db':
        xi = V[2, 2] * V[2, 0].conj()
    elif ddll[:2] == 'ds':
        xi = V[2, 1] * V[2, 0].conj()
    else:
        raise ValueError("Unexpected flavours: {}".format(ddll[:2]))
    ind = ddll.replace('l_','').replace('nu_','')
    indfl = ddll[1::-1]+ind[2:] # flavio has first two indices inverted
    indnu = ddll[1::-1]+ddll.replace('l_','nu').replace('nu_','nu')[2:]
    e = sqrt(4* pi * parameters['alpha_e'])
    mb = parameters['m_b']
    dic = {
        "CL_" + indnu : (8 * pi**2) / e**2 * C['F' + ind + 'nu'],
        "CR_" + indnu : (8 * pi**2) / e**2 * C['F' + ind + 'nup']
    }
    if include_charged:
        dic.update({
            "C9_" + indfl : (16 * pi**2) / e**2 * C['F' + ind + '9'],
            "C9p_" + indfl : (16 * pi**2) / e**2 * C['F' + ind + '9p'],
            "C10_" + indfl : (16 * pi**2) / e**2 * C['F' + ind + '10'],
            "C10p_" + indfl : (16 * pi**2) / e**2 * C['F' + ind + '10p'],
            "CS_" + indfl : (16 * pi**2) / e**2 / mb * C['F' + ind + 'S'],
            "CSp_" + indfl : (16 * pi**2) / e**2 / mb * C['F' + ind + 'Sp'],
            "CP_" + indfl : (16 * pi**2) / e**2 / mb * C['F' + ind + 'P'],
            "CPp_" + indfl : (16 * pi**2) / e**2 / mb * C['F' + ind + 'Pp'],
        })
    if norm_gf:
        prefactor = sqrt(2)/p['GF']/xi/4
    else:
        prefactor = 1 / xi
    return {k: prefactor * v for k,v in dic.items()}


def Flavio_to_Fierz_lep(C, ddll, parameters, include_charged=True, norm_gf=True):
    """From  Flavio semileptonic basis to semileptonic Fierz basis for Class V.
    C should be the corresponding leptonic Fierz basis and
    `ddll` should be of the form 'sbl_enu_tau', 'dbl_munu_e' etc."""
    p = parameters
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["gamma"])
    if ddll[:2] == 'sb':
        xi = V[2, 2] * V[2, 1].conj()
    elif ddll[:2] == 'db':
        xi = V[2, 2] * V[2, 0].conj()
    elif ddll[:2] == 'ds':
        xi = V[2, 1] * V[2, 0].conj()
    else:
        raise ValueError("Unexpected flavours: {}".format(ddll[:2]))
    ind = ddll.replace('l_','').replace('nu_','')
    indfl = ddll[1::-1]+ind[2:] # flavio has first two indices inverted
    indnu = ddll[1::-1]+ddll.replace('l_','nu').replace('nu_','nu')[2:]
    e = sqrt(4* pi * parameters['alpha_e'])
    mb = parameters['m_b']
    dic = {
        'F' + ind + 'nu': C["CL_" + indnu] / ((8 * pi**2) / e**2),
        'F' + ind + 'nup': C["CR_" + indnu] / ((8 * pi**2) / e**2),
        }
    if include_charged:
        dic.update({
        'F' + ind + '9': C["C9_" + indfl] / ((16 * pi**2) / e**2),
        'F' + ind + '9p': C["C9p_" + indfl] / ((16 * pi**2) / e**2),
        'F' + ind + '10': C["C10_" + indfl] / ((16 * pi**2) / e**2),
        'F' + ind + '10p': C["C10p_" + indfl] / ((16 * pi**2) / e**2),
        'F' + ind + 'S': C["CS_" + indfl] / ((16 * pi**2) / e**2 / mb),
        'F' + ind + 'Sp': C["CSp_" + indfl] / ((16 * pi**2) / e**2 / mb),
        'F' + ind + 'P': C["CP_" + indfl] / ((16 * pi**2) / e**2 / mb),
        'F' + ind + 'Pp': C["CPp_" + indfl] / ((16 * pi**2) / e**2 / mb),
        'F' + ind + 'T': 0,  # tensors not implemented in flavio basis yet
        'F' + ind + 'T5': 0,  # tensors not implemented in flavio basis yet
        })
    if norm_gf:
        prefactor = sqrt(2)/p['GF']/xi/4
    else:
        prefactor = 1 / xi
    return {k: v / prefactor for k, v in dic.items()}


def Fierz_to_EOS_lep(C, ddll, parameters):
    """From semileptonic Fierz basis to EOS semileptonic basis for Class V.
    C should be the corresponding leptonic Fierz basis and
    `ddll` should be of the form 'sbl_enu_tau', 'dbl_munu_e' etc."""
    p = parameters
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["gamma"])
    Vtb = V[2,2]
    Vts = V[2,1]
    ind = ddll.replace('l_','').replace('nu_','')
    ind2 = ddll.replace('l_','').replace('nu_','')[2::]
    e = sqrt(4* pi * parameters['alpha_e'])
    dic = {
        'b->s' + ind2 + '::c9' : (16 * pi**2) / e**2 * C['F' + ind + '9'],
        'b->s' + ind2 + "::c9'" : (16 * pi**2) / e**2 * C['F' + ind + '9p'],
        'b->s' + ind2 + "::c10" : (16 * pi**2) / e**2 * C['F' + ind + '10'],
        'b->s' + ind2 + "::c10'" : (16 * pi**2) / e**2 * C['F' + ind + '10p'],
        'b->s' + ind2 + "::cS" : (16 * pi**2) / e**2  * C['F' + ind + 'S'],
        'b->s' + ind2 + "::cS'" : (16 * pi**2) / e**2 * C['F' + ind + 'Sp'],
        'b->s' + ind2 + "::cP" : (16 * pi**2) / e**2  * C['F' + ind + 'P'],
        'b->s' + ind2 + "::cP'" : (16 * pi**2) / e**2  * C['F' + ind + 'Pp'],
        'b->s' + ind2 + "::cT"  : (16 * pi**2) / e**2  * C['F' + ind + 'T'],
        'b->s' + ind2 + "::cT5"  : (16 * pi**2) / e**2  * C['F' + ind + 'T5']
        }
    prefactor = sqrt(2)/p['GF']/Vtb/Vts.conj()/4
    return {k: prefactor * v for k,v in dic.items()}

def JMS_to_FormFlavor_lep(C, dd):
    """From JMS to semileptonic Fierz basis for Classes V.
    C should be the JMS basis and `ddll` should be of the
    form 'sbl_eni_tau', 'dbl_munu_e' etc."""
    b = dflav[dd[0]]
    s = dflav[dd[1]]
    return {
        'CVLL_' + dd + 'mm' : C["VedLL"][1, 1, s, b],
        'CVRR_' + dd + 'mm' :  C["VedRR"][1, 1, s, b],
        'CVLR_' + dd + 'mm' : C["VdeLR"][s, b, 1, 1],
        'CVRL_' + dd + 'mm' : C["VedLR"][1, 1, s, b],
        'CSLL_' + dd + 'mm' : C["SedRR"][1, 1, b, s].conj(),
        'CSRR_' + dd + 'mm' : C["SedRR"][1, 1, s, b],
        'CSLR_' + dd + 'mm' : C["SedRL"][1, 1, s, b],
        'CSRL_' + dd + 'mm' : C["SedRL"][1, 1, b, s].conj(),
        'CTLL_' + dd + 'mm' : C["TedRR"][1, 1, b, s].conj(),
        'CTRR_' + dd + 'mm' : C["TedRR"][1, 1, s, b],
        'CVLL_sdnn' : 1 / 3 * C["VnudLL"][0, 0, s-1, s]
                      + 1 / 3 * C["VnudLL"][1, 1, s-1, s]
                      + 1 / 3 * C["VnudLL"][2, 2, s-1, s],
        'CVRL_sdnn' : 1 / 3 * C["VnudLR"][0, 0, s-1, s]
                      + 1 / 3 * C["VnudLR"][1, 1, s-1, s]
                      + 1 / 3 * C["VnudLR"][2, 2, s-1, s]
            }

# chromomagnetic operators
def JMS_to_Fierz_chrom(C, dd):
    """From JMS to chromomagnetic Fierz basis for Class V.
    qq should be of the form 'sb', 'ds' etc."""
    s = dflav[dd[0]]
    b = dflav[dd[1]]
    return {
            'F7gamma' + dd : C['dgamma'][s, b],
            'F8g' + dd : C['dG'][s, b],
            'F7pgamma' + dd : C['dgamma'][b, s].conj(),
            'F8pg' + dd : C['dG'][b, s].conj()
                }


def Fierz_to_Bern_chrom(C, dd, parameters):
    """From Fierz to chromomagnetic Bern basis for Class V.
    dd should be of the form 'sb', 'ds' etc."""
    e = sqrt(4 * pi * parameters['alpha_e'])
    gs = sqrt(4 * pi * parameters['alpha_s'])
    mb = parameters['m_b']
    return {
        '7gamma' + dd : gs**2 / e / mb * C['F7gamma' + dd ],
        '8g' + dd : gs / mb * C['F8g' + dd ],
        '7pgamma' + dd : gs**2 / e /mb * C['F7pgamma' + dd],
        '8pg' + dd : gs / mb * C['F8pg' + dd]
            }


def Bern_to_Fierz_chrom(C, dd, parameters):
    """From Bern to chromomagnetic Fierz basis for Class V.
    dd should be of the form 'sb', 'ds' etc."""
    e = sqrt(4 * pi * parameters['alpha_e'])
    gs = sqrt(4 * pi * parameters['alpha_s'])
    mb = parameters['m_b']
    return {
        'F7gamma' + dd : C['7gamma' + dd] / (gs**2 / e / mb),
        'F8g' + dd : C['8g' + dd] / (gs / mb),
        'F7pgamma' + dd: C['7pgamma' + dd] / (gs**2 / e /mb),
        'F8pg' + dd: C['8pg' + dd] / (gs / mb)
            }


def Fierz_to_Flavio_chrom(C, dd, parameters):
    """From Fierz to chromomagnetic Flavio basis for Class V.
    dd should be of the form 'sb', 'db' etc."""
    p = parameters
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["gamma"])
    if dd == 'sb':
        xi = V[2, 2] * V[2, 1].conj()
    elif dd == 'db':
        xi = V[2, 2] * V[2, 0].conj()
    elif dd == 'ds':
        xi = V[2, 1] * V[2, 0].conj()
    else:
        raise ValueError("Unexpected flavours: {}".format(dd))
    ddfl = dd[::-1]
    e = sqrt(4 * pi * parameters['alpha_e'])
    gs = sqrt(4 * pi * parameters['alpha_s'])
    mb = parameters['m_b']
    dic = {
        "C7_" + ddfl : (16 * pi**2) / e / mb * C['F7gamma' + dd],
        "C8_" + ddfl : (16 * pi**2) / gs / mb * C['F8g' + dd],
        "C7p_" + ddfl : (16 * pi**2) / e / mb * C['F7pgamma' + dd],
        "C8p_" + ddfl : (16 * pi**2) / gs / mb * C['F8pg' + dd]
            }
    prefactor = sqrt(2)/p['GF']/xi/4
    return {k: prefactor * v for k, v in dic.items()}


def Flavio_to_Fierz_chrom(C, dd, parameters):
    """From Flavio to chromomagnetic Fierz basis for Class V.
    dd should be of the form 'sb', 'db' etc."""
    p = parameters
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["gamma"])
    if dd == 'sb':
        xi = V[2, 2] * V[2, 1].conj()
    elif dd == 'db':
        xi = V[2, 2] * V[2, 0].conj()
    elif dd == 'ds':
        xi = V[2, 1] * V[2, 0].conj()
    else:
        raise ValueError("Unexpected flavours: {}".format(dd))
    ddfl = dd[::-1]
    e = sqrt(4 * pi * parameters['alpha_e'])
    gs = sqrt(4 * pi * parameters['alpha_s'])
    mb = parameters['m_b']
    dic = {
        'F7gamma' + dd: C["C7_" + ddfl] / ((16 * pi**2) / e / mb),
        'F8g' + dd: C["C8_" + ddfl] / ((16 * pi**2) / gs / mb),
        'F7pgamma' + dd: C["C7p_" + ddfl] / ((16 * pi**2) / e / mb),
        'F8pg' + dd: C["C8p_" + ddfl] / ((16 * pi**2) / gs / mb)
            }
    prefactor = sqrt(2)/p['GF']/xi/4
    return {k: v / prefactor for k, v in dic.items()}


def Fierz_to_EOS_chrom(C, dd, parameters):
    """From Fierz to chromomagnetic EOS basis for Class V.
    dd should be of the form 'sb', 'ds' etc."""
    p = parameters
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["gamma"])
    Vtb = V[2,2]
    Vts = V[2,1]
    e = sqrt(4 * pi * parameters['alpha_e'])
    gs = sqrt(4 * pi * parameters['alpha_s'])
    mb = parameters['m_b']
    ms = parameters['m_s']
    dic = {"b->s::c7": 16 * pi**2 * mb / e * C["F7gamma" + dd] / (mb**2 - ms**2)
                       -16 * pi**2 * ms / e * C["F7pgamma" + dd]
                                                              / (mb**2 - ms**2),
            "b->s::c7'": - 16 * pi**2  * ms / e * C["F7gamma" + dd]
                                                    / (mb**2 - ms**2)
                        + 16 * pi**2  * mb / e * C["F7pgamma" + dd]
                                                        / (mb**2 - ms**2),
            "b->s::c8": 16 * pi**2 * mb / gs * C["F8g" + dd] / (mb**2 - ms**2)
                        - 16 * pi**2 * ms / gs * C["F8pg" + dd]
                                                            / (mb**2 - ms**2),
            "b->s::c8'": - 16 * pi**2 * ms / gs * C["F8g" + dd] / (mb**2 - ms**2)
                        + 16 * pi**2 * mb / gs * C["F8pg" + dd] / (mb**2 - ms**2)
                }
    prefactor = sqrt(2)/p['GF']/Vtb/Vts.conj()/4
    return {k: prefactor * v for k,v in dic.items()}


def JMS_to_FormFlavor_chrom(C, qq, parameters):
    """From JMS to chromomagnetic FormFlavor basis for Class V.
    qq should be of the form 'sb', 'ds', 'uu', mt (mu tau), em (e mu) etc."""
    e = sqrt(4 * pi * parameters['alpha_e'])
    gs = sqrt(4 * pi * parameters['alpha_s'])
    if qq[0] in dflav.keys():
        s = dflav[qq[0]]
        b = dflav[qq[1]]
        return {
            'CAR_' + qq : C['dgamma'][s, b] / e,
            'CGR_' + qq : C['dG'][s, b] / gs,
            'CAL_'  + qq : C['dgamma'][b, s].conj() / e,
            'CGL_' + qq : C['dG'][b, s].conj() / gs,
                }
    if qq[0] in llflav.keys():
        l1 = llflav[qq[0]]
        l2 = llflav[qq[1]]
        return {
            'CAR_' + qq : C['egamma'][l1, l2] / e,
            'CAL_' + qq : C['egamma'][l2, l1].conj() / gs,
                }
    if qq[0] in uflav.keys():
        u = uflav[qq[0]]
        c = uflav[qq[1]]
        return {
            'CAR_' + qq : C['ugamma'][u, c] / e,
            'CGR_' + qq : C['uG'][u, c] / gs,
            'CAL_'  + qq : C['ugamma'][c, u].conj() / e,
            'CGL_' + qq : C['uG'][c, u].conj() / gs,
                }
    else:
        return 'not in FormFlav_chrom'


def _JMS_to_Bern_VII(Cflat, parameters):
    """From JMS to Bern basis for class VII, i.e. flavour blind operators
    mixing with the quark EDMs and CEDMs."""
    d = {}
    k_unchanged = ['TeuRR_1111', 'TeuRR_1122', 'TeuRR_2211', 'TeuRR_2222',
    'TeuRR_3311', 'TeuRR_3322', 'TedRR_1111', 'TedRR_1122', 'TedRR_1133',
    'TedRR_2211', 'TedRR_2222', 'TedRR_2233', 'TedRR_3311', 'TedRR_3322',
    'TedRR_3333', 'S1uuRR_1111', 'S1uuRR_1122', 'S1uuRR_1221', 'S1uuRR_2222',
    'S8uuRR_1111', 'S8uuRR_1122', 'S8uuRR_1221', 'S8uuRR_2222', 'S1udRR_1111',
    'S1udRR_1122', 'S1udRR_1133', 'S1udRR_2211', 'S1udRR_2222', 'S1udRR_2233',
    'S8udRR_1111', 'S8udRR_1122', 'S8udRR_1133', 'S8udRR_2211', 'S8udRR_2222',
    'S8udRR_2233', 'S1ddRR_1111', 'S1ddRR_1122', 'S1ddRR_1133', 'S1ddRR_1221',
    'S1ddRR_1331', 'S1ddRR_2222', 'S1ddRR_2233', 'S1ddRR_2332', 'S1ddRR_3333',
    'S8ddRR_1111', 'S8ddRR_1122', 'S8ddRR_1133', 'S8ddRR_1221', 'S8ddRR_1331',
    'S8ddRR_2222', 'S8ddRR_2233', 'S8ddRR_2332', 'S8ddRR_3333', 'S1udduRR_1111',
    'S1udduRR_1221', 'S1udduRR_1331', 'S1udduRR_2112', 'S1udduRR_2222',
    'S1udduRR_2332', 'S8udduRR_1111', 'S8udduRR_1221', 'S8udduRR_1331',
    'S8udduRR_2112', 'S8udduRR_2222', 'S8udduRR_2332']
    e = sqrt(4 * pi * parameters['alpha_e'])
    gs = sqrt(4 * pi * parameters['alpha_s'])
    mu = parameters['m_u']
    md = parameters['m_d']
    ms = parameters['m_s']
    mc = parameters['m_c']
    mb = parameters['m_b']
    for k in k_unchanged:
        if k in Cflat:
            d[k] = Cflat[k]
    d['G'] = gs * Cflat.get('G', 0)
    d['Gtilde'] = gs * Cflat.get('Gtilde', 0)
    d['uG_11'] = gs / mu * Cflat.get('uG_11', 0)
    d['uG_22'] = gs / mc * Cflat.get('uG_22', 0)
    d['dG_11'] = gs / md * Cflat.get('dG_11', 0)
    d['dG_22'] = gs / ms * Cflat.get('dG_22', 0)
    d['dG_33'] = gs / mb * Cflat.get('dG_33', 0)
    d['ugamma_11'] = gs**2 / e / mu * Cflat.get('ugamma_11', 0)
    d['ugamma_22'] = gs**2 / e / mc * Cflat.get('ugamma_22', 0)
    d['dgamma_11'] = gs**2 / e / md * Cflat.get('dgamma_11', 0)
    d['dgamma_22'] = gs**2 / e / ms * Cflat.get('dgamma_22', 0)
    d['dgamma_33'] = gs**2 / e / mb * Cflat.get('dgamma_33', 0)
    return d


def _Bern_to_Flavio_VII(C, parameters):
    """From Bern to flavio basis for class VII, i.e. flavour blind operators
    mixing with the quark EDMs and CEDMs."""
    d = {}
    dtrans = json.loads(pkgutil.get_data('wcxf', 'data/bern_flavio_vii.json').decode('utf8'))
    for cb, cf in dtrans.items():
        d[cf] = C.get(cb, 0)
    gs = sqrt(4 * pi * parameters['alpha_s'])
    d['CG'] = 1 / gs * C.get('G', 0)
    d['CGtilde'] = 1 / gs * C.get('Gtilde', 0)
    pre = 16 * pi**2 / gs**2
    d['C8_uu'] = pre * C.get('uG_11', 0)
    d['C8_cc'] = pre * C.get('uG_22', 0)
    d['C8_dd'] = pre * C.get('dG_11', 0)
    d['C8_ss'] = pre * C.get('dG_22', 0)
    d['C8_bb'] = pre * C.get('dG_33', 0)
    d['C7_uu'] = pre * C.get('ugamma_11', 0)
    d['C7_cc'] = pre * C.get('ugamma_22', 0)
    d['C7_dd'] = pre * C.get('dgamma_11', 0)
    d['C7_ss'] = pre * C.get('dgamma_22', 0)
    d['C7_bb'] = pre * C.get('dgamma_33', 0)
    # note that this prefactor is removed below in Bern_to_flavio!
    preGF = sqrt(2) / parameters['GF'] / 4
    return {k: preGF * v for k,v in d.items()}


def _Flavio_to_Bern_VII(C, parameters):
    """From flavio to Bern basis for class VII, i.e. flavour blind operators
    mixing with the quark EDMs and CEDMs."""
    d = {}
    dtrans = json.loads(pkgutil.get_data('wcxf', 'data/bern_flavio_vii.json').decode('utf8'))
    for cb, cf in dtrans.items():
        d[cb] = C.get(cf, 0)
    gs = sqrt(4 * pi * parameters['alpha_s'])
    d['G'] = gs * C.get('CG', 0)
    d['Gtilde'] = gs * C.get('CGtilde', 0)
    pre = gs**2 / (16 * pi**2)
    d['uG_11'] = pre * C.get('C8_uu', 0)
    d['uG_22'] = pre * C.get('C8_cc', 0)
    d['dG_11'] = pre * C.get('C8_dd', 0)
    d['dG_22'] = pre * C.get('C8_ss', 0)
    d['dG_33'] = pre * C.get('C8_bb', 0)
    d['ugamma_11'] = pre * C.get('C7_uu', 0)
    d['ugamma_22'] = pre * C.get('C7_cc', 0)
    d['dgamma_11'] = pre * C.get('C7_dd', 0)
    d['dgamma_22'] = pre * C.get('C7_ss', 0)
    d['dgamma_33'] = pre * C.get('C7_bb', 0)
    # note that this prefactor is removed below in flavio_to_Bern!
    preGF = sqrt(2) / parameters['GF'] / 4
    return {k: v / preGF for k,v in d.items()}


# symmetrize JMS basis

def _scalar2array(d):
    """Convert a dictionary with scalar elements and string indices '_1234'
    to a dictionary of arrays. Unspecified entries are np.nan."""
    da = {}
    for k, v in d.items():
        if '_' not in k:
            da[k] = v
        else:
            name = ''.join(k.split('_')[:-1])
            ind = k.split('_')[-1]
            dim = len(ind)
            if name not in da:
                shape = tuple(3 for i in range(dim))
                da[name] = np.empty(shape, dtype=complex)
                da[name][:] = np.nan
            da[name][tuple(int(i) - 1 for i in ind)] = v
    return da


def _symm_herm(C):
    """To get rid of NaNs produced by _scalar2array, symmetrize operators
    where C_ijkl = C_jilk*"""
    nans = np.isnan(C)
    C[nans] = np.einsum('jilk', C)[nans].conj()
    return C


def _symm_current(C):
    """To get rid of NaNs produced by _scalar2array, symmetrize operators
    where C_ijkl = C_klij"""
    nans = np.isnan(C)
    C[nans] = np.einsum('klij', C)[nans]
    return C



def _JMS_to_array(C):
    """For a dictionary with JMS Wilson coefficients, return a dictionary
    of arrays."""
    wc_keys = wcxf.Basis['WET', 'JMS'].all_wcs
    # fill in zeros for missing coefficients
    C_complete = {k: C.get(k, 0) for k in wc_keys}
    Ca = _scalar2array(C_complete)
    for k in Ca:
        if k in ["VnueLL", "VnuuLL", "VnudLL", "VeuLL", "VedLL", "V1udLL",
                "V8udLL", "VeuRR", "VedRR", "V1udRR", "V8udRR", "VnueLR",
                "VeeLR", "VnuuLR", "VnudLR", "VeuLR", "VedLR", "VueLR", "VdeLR",
                "V1uuLR", "V8uuLR", "V1udLR", "V8udLR", "V1duLR", "V8duLR",
                "V1ddLR", "V8ddLR"]:
            Ca[k] = _symm_herm(Ca[k])
        if k in ["S1uuRR", "S8uuRR", "S1ddRR", "S8ddRR"]:
            Ca[k] = _symm_current(Ca[k])
        if k in ["VuuLL", "VddLL", "VuuRR", "VddRR"]:
            Ca[k] = _symm_herm(_symm_current(Ca[k]))
    return Ca


def rotate_down(C_in, p):
    """Redefinition of all Wilson coefficients in the JMS basis when rotating
    down-type quark fields from the flavour to the mass basis.

    C_in is expected to be an array-valued dictionary containg a key
    for all Wilson coefficient matrices."""
    C = C_in.copy()
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["gamma"])
    UdL = V
    # type dL dL dL dL
    for k in ['VddLL']:
        C[k] = np.einsum('ia,jb,kc,ld,ijkl->abcd',
                         UdL.conj(), UdL, UdL.conj(), UdL,
                         C_in[k])
    # type X X dL dL
    for k in ['V1udLL', 'V8udLL', 'VedLL', 'VnudLL']:
        C[k] = np.einsum('kc,ld,ijkl->ijcd',
                         UdL.conj(), UdL,
                         C_in[k])
    # type dL dL X X
    for k in ['V1ddLR', 'V1duLR', 'V8ddLR', 'V8duLR', 'VdeLR']:
        C[k] = np.einsum('ia,jb,ijkl->abkl',
                         UdL.conj(), UdL,
                         C_in[k])
    # type dL X dL X
    for k in ['S1ddRR', ]:
        C[k] = np.einsum('ia,kc,ijkl->ajcl',
                         UdL.conj(), UdL.conj(),
                         C_in[k])
    # type X dL X X
    for k in ['V1udduLR', 'V8udduLR']:
        C[k] = np.einsum('jb,ijkl->ibkl',
                         UdL,
                         C_in[k])
    # type X X dL X
    for k in ['VnueduLL', 'SedRR', 'TedRR', 'SnueduRR', 'TnueduRR',
              'S1udRR',  'S8udRR', 'S1udduRR',  'S8udduRR', ]:
        C[k] = np.einsum('kc,ijkl->ijcl',
                         UdL.conj(),
                         C_in[k])
    # type X X X dL
    for k in ['SedRL', ]:
        C[k] = np.einsum('ld,ijkl->ijkd',
                         UdL,
                         C_in[k])
    return C


def get_parameters(scale, f=5, input_parameters=None):
    """Get parameters (masses, coupling constants, ...) at the scale
    `scale` in QCD with `f` dynamical quark flavours. Optionally takes a
    dictionary of inputs (otherwise, defaults are used)."""
    p = default_parameters.copy()
    if input_parameters is not None:
        # if parameters are passed in, overwrite the default values
        p.update(input_parameters)
    parameters = {}
    # running quark masses and alpha_s
    parameters['m_b'] = m_b(p['m_b'], scale, f, p['alpha_s'])
    parameters['m_c'] = m_c(p['m_c'], scale, f, p['alpha_s'])
    parameters['m_s'] = m_s(p['m_s'], scale, f, p['alpha_s'])
    parameters['m_u'] = m_s(p['m_u'], scale, f, p['alpha_s'])
    parameters['m_d'] = m_s(p['m_d'], scale, f, p['alpha_s'])
    parameters['alpha_s'] = alpha_s(scale, f, p['alpha_s'])
    # no running is performed for these parameters
    for k in ['m_W', 'm_Z', 'GF',
              'alpha_e',
              'Vus', 'Vub', 'Vcb', 'gamma',
              'm_e', 'm_mu', 'm_tau', ]:
        parameters[k] = p[k]
    return parameters


# final dicitonaries

def JMS_to_EOS(Cflat, scale, parameters=None):
    p = get_parameters(scale, f=5, input_parameters=parameters)
    C = _JMS_to_array(Cflat)
    d={}

    # Class II
    for l in ['e','mu']:
        d.update(_BernII_to_EOS_II(_JMS_to_Bern_II(C, 'ub'+'l_'+l+'nu_'+l),
                                          'ub'+'l_'+l+'nu_'+l,
                                        p))

    # Class V
    Fsbuu = _JMS_to_Fierz_III_IV_V(C, 'sbuu')
    Fsbdd = _JMS_to_Fierz_III_IV_V(C, 'sbdd')
    Fsbcc = _JMS_to_Fierz_III_IV_V(C, 'sbcc')
    Fsbss = _JMS_to_Fierz_III_IV_V(C, 'sbss')
    Fsbbb = _JMS_to_Fierz_III_IV_V(C, 'sbbb')

    d.update(_Fierz_to_EOS_V(Fsbuu,Fsbdd,Fsbcc,Fsbss,Fsbbb,p))

    # Class V semileptonic
    d.update(Fierz_to_EOS_lep(JMS_to_Fierz_lep(C, 'sbl_enu_e'),'sbl_enu_e', p))
    d.update(Fierz_to_EOS_lep(JMS_to_Fierz_lep(C, 'sbl_munu_mu'),'sbl_munu_mu'
                                                      , p))

    # Class V chromomagnetic
    d.update(Fierz_to_EOS_chrom(JMS_to_Fierz_chrom(C, 'sb'), 'sb', p))
    return d


def JMS_to_flavio(Cflat, scale, parameters=None):
    p = get_parameters(scale, f=5, input_parameters=parameters)
    C = _JMS_to_array(Cflat)
    d={}

    # Class I
    for qq in ['sb', 'db', 'ds', 'cu']:
        d.update(_BernI_to_Flavio_I(_JMS_to_Bern_I(C, qq), qq))

    # Class II
    for l in lflav.keys():
        for lp in lflav.keys():
            for qq in ['cb', 'ub', 'us', 'cs', 'cd', 'ud']:
                d.update(_BernII_to_Flavio_II(_JMS_to_Bern_II(C,
                                              qq+'l_'+l+'nu_'+lp),
                                              qq+'l_'+l+'nu_'+lp, p))

    # Class V semileptonic
    for l in lflav.keys():
        for lp in lflav.keys():
            d.update(Fierz_to_Flavio_lep(JMS_to_Fierz_lep(C,
                                        'sb'+'l_'+l+'nu_'+lp),
                                        'sb'+'l_'+l+'nu_'+lp, p,
                                        norm_gf=True))
            d.update(Fierz_to_Flavio_lep(JMS_to_Fierz_lep(C,
                                        'db'+'l_'+l+'nu_'+lp),
                                         'db'+'l_'+l+'nu_'+lp, p,
                                         norm_gf=True))
            d.update(Fierz_to_Flavio_lep(JMS_to_Fierz_lep(C,
                                        'ds'+'l_'+l+'nu_'+lp),
                                        'ds'+'l_'+l+'nu_'+lp, p,
                                        include_charged=(l==lp),
                                        norm_gf=True))  # l+l- only for l=l'

    # Class V non-leptonic
    for qq1 in ['ds', 'sb', 'db']:
        for qq2 in ['uu', 'dd', 'ss', 'cc', 'bb']:
            qqqq = qq1 + qq2
            d.update(_Fierz_to_Flavio_V(_JMS_to_Fierz_III_IV_V(C, qqqq),
                                        qqqq, p))

    # Class V chromomagnetic
    d.update(Fierz_to_Flavio_chrom(JMS_to_Fierz_chrom(C, 'sb'), 'sb', p))
    d.update(Fierz_to_Flavio_chrom(JMS_to_Fierz_chrom(C, 'db'), 'db', p))
    d.update(Fierz_to_Flavio_chrom(JMS_to_Fierz_chrom(C, 'ds'), 'ds', p))

    # Class VII
    d.update(_Bern_to_Flavio_VII(_JMS_to_Bern_VII(Cflat, p), p))

    return d


def Bern_to_flavio(C_incomplete, scale, parameters=None):
    p = get_parameters(scale, f=5, input_parameters=parameters)
    # fill in zeros for missing coefficients
    wc_keys = wcxf.Basis['WET', 'Bern'].all_wcs
    C = {k: C_incomplete.get(k, 0) for k in wc_keys}
    d = {}

    # Class I
    for qq in ['sb', 'db', 'ds', 'cu']:
        d.update(_BernI_to_Flavio_I(C, qq))

    # Class II
    for l in lflav.keys():
        for lp in lflav.keys():
            for qq in ['cb', 'ub', 'us', 'cs', 'cd', 'ud']:
                d.update(_BernII_to_Flavio_II(C, qq+'l_'+l+'nu_'+lp, p))

    # Class V semileptonic
    for l in lflav.keys():
        for lp in lflav.keys():
            d.update(Fierz_to_Flavio_lep(Bern_to_Fierz_lep(C,
                                        'sb'+'l_'+l+'nu_'+lp),
                                        'sb'+'l_'+l+'nu_'+lp, p,
                                        norm_gf=True))
            d.update(Fierz_to_Flavio_lep(Bern_to_Fierz_lep(C,
                                        'db'+'l_'+l+'nu_'+lp),
                                         'db'+'l_'+l+'nu_'+lp, p,
                                         norm_gf=True))
            d.update(Fierz_to_Flavio_lep(Bern_to_Fierz_lep(C,
                                        'ds'+'l_'+l+'nu_'+lp),
                                         'ds'+'l_'+l+'nu_'+lp, p,
                                         include_charged=(l==lp),  # l+l- only for l=l'
                                         norm_gf=True))


    # Class V non-leptonic
    for qq1 in ['ds', 'sb', 'db']:
        for qq2 in ['uu', 'dd', 'ss', 'cc', 'bb']:
            qqqq = qq1 + qq2
            d.update(_Fierz_to_Flavio_V(_Bern_to_Fierz_III_IV_V(C, qqqq),
                                        qqqq, p))

    # Class V chromomagnetic
    d.update(Fierz_to_Flavio_chrom(Bern_to_Fierz_chrom(C, 'sb', p), 'sb', p))
    d.update(Fierz_to_Flavio_chrom(Bern_to_Fierz_chrom(C, 'db', p), 'db', p))
    d.update(Fierz_to_Flavio_chrom(Bern_to_Fierz_chrom(C, 'ds', p), 'ds', p))

    # Class VII
    d.update(_Bern_to_Flavio_VII(C, p))

    prefactor = sqrt(2)/p['GF']/4
    return {k: v / prefactor for k,v in d.items()}



def flavio_to_Bern(C_incomplete, scale, parameters=None):
    p = get_parameters(scale, f=5, input_parameters=parameters)
    # fill in zeros for missing coefficients
    wc_keys = wcxf.Basis['WET', 'flavio'].all_wcs
    C = {k: C_incomplete.get(k, 0) for k in wc_keys}
    d = {}

    # Class I
    for qq in ['bs', 'bd', 'sd', 'uc']:
        d.update(_FlavioI_to_Bern_I(C, qq))

    # Class II
    for l in lflav.keys():
        for lp in lflav.keys():
            for qq in ['cb', 'ub', 'us', 'cs', 'cd', 'ud']:
                d.update(_FlavioII_to_BernII(C, qq+'l_'+l+'nu_'+lp, p))

    # Class V semileptonic
    for l in lflav.keys():
        for lp in lflav.keys():
            d.update(Fierz_to_Bern_lep(Flavio_to_Fierz_lep(C,
                                        'sb'+'l_'+l+'nu_'+lp, p,
                                        norm_gf=True),
                                        'sb'+'l_'+l+'nu_'+lp))
            d.update(Fierz_to_Bern_lep(Flavio_to_Fierz_lep(C,
                                        'db'+'l_'+l+'nu_'+lp, p,
                                        norm_gf=True),
                                        'db'+'l_'+l+'nu_'+lp))
            d.update(Fierz_to_Bern_lep(Flavio_to_Fierz_lep(C,
                                        'ds'+'l_'+l+'nu_'+lp, p,
                                        include_charged=(l==lp),
                                        norm_gf=True),  # l+l- only for l=l'
                                        'ds'+'l_'+l+'nu_'+lp,
                                        include_charged=(l==lp)),  # l+l- only for l=l'
                                        )


    # Class V non-leptonic
    for qq1 in ['ds', 'sb', 'db']:
        for qq2 in ['uu', 'dd', 'ss', 'cc', 'bb']:
            qqqq = qq1 + qq2
            d.update(_Fierz_to_Bern_III_IV_V(_Flavio_to_Fierz_V(C, qqqq, p),
                                             qqqq))

    # Class V chromomagnetic
    d.update(Fierz_to_Bern_chrom(Flavio_to_Fierz_chrom(C, 'sb', p), 'sb', p))
    d.update(Fierz_to_Bern_chrom(Flavio_to_Fierz_chrom(C, 'db', p), 'db', p))
    d.update(Fierz_to_Bern_chrom(Flavio_to_Fierz_chrom(C, 'ds', p), 'ds', p))

    # Class VII
    d.update(_Flavio_to_Bern_VII(C, p))

    prefactor = sqrt(2)/p['GF']/4
    return {k: prefactor * v for k,v in d.items()}

def JMS_to_FormFlavor(Cflat, scale, parameters=None):
    p = get_parameters(scale, f=5, input_parameters=parameters)
    C = _JMS_to_array(Cflat)
    d={}

    # Class I
    for qq in ['sb', 'db', 'ds', 'cu']:
        d.update(_BernI_to_FormFlavor_I(_JMS_to_Bern_I(C, qq), qq))

    # Class V semileptonic
    d.update(JMS_to_FormFlavor_lep(C, 'bs'))
    d.update(JMS_to_FormFlavor_lep(C, 'bd'))

    # Class V chromomagnetic
    for ind in ['sb', 'db', 'uu', 'dd', 'mt', 'em', 'et']:
        d.update(JMS_to_FormFlavor_chrom(C, ind, p))
    return d


def JMS_to_Bern(Cflat, scale, parameters=None):
    p = get_parameters(scale, f=5, input_parameters=parameters)
    C = _JMS_to_array(Cflat)
    d={}

    # Class I
    for qq in ['sb', 'db', 'ds', 'cu']:
        d.update(_JMS_to_Bern_I(C, qq))

    # Class II
    for l in lflav.keys():
        for lp in lflav.keys():
            for qq in ['cb', 'ub', 'us', 'cs', 'cd', 'ud']:
                d.update(_JMS_to_Bern_II(C, qq+'l_'+l+'nu_'+lp))

    # Class V
    for u1 in uflav.keys():
        for u2 in uflav.keys():
            d.update(_Fierz_to_Bern_III_IV_V(_JMS_to_Fierz_III_IV_V(C,
                                                      'sb'+u1+u2), 'sb'+u1+u2))

            d.update(_Fierz_to_Bern_III_IV_V(_JMS_to_Fierz_III_IV_V(C,
                                                      'db'+u1+u2), 'db'+u1+u2))

            d.update(_Fierz_to_Bern_III_IV_V(_JMS_to_Fierz_III_IV_V(C,
                                                      'ds'+u1+u2), 'ds'+u1+u2))

    for qqqq in ['sbdd', 'sbss', 'dbdd', 'dbss', 'dbbb', 'sbbb',
                 'dbds', 'sbsd', 'dsbb',
                 'dsss', 'dsdd',
                 ]:
        d.update(_Fierz_to_Bern_III_IV_V(_JMS_to_Fierz_III_IV_V(C, qqqq), qqqq))

    # Class V semileptonic
    for l in lflav.keys():
        for lp in lflav.keys():
            d.update(Fierz_to_Bern_lep(JMS_to_Fierz_lep(C, 'sb'+'l_'+l+'nu_'+lp)
                                                         ,'sb'+'l_'+l+'nu_'+lp))
            d.update(Fierz_to_Bern_lep(JMS_to_Fierz_lep(C, 'db'+'l_'+l+'nu_'+lp)
                                                         ,'db'+'l_'+l+'nu_'+lp))
            d.update(Fierz_to_Bern_lep(JMS_to_Fierz_lep(C, 'ds'+'l_'+l+'nu_'+lp)
                                                         ,'ds'+'l_'+l+'nu_'+lp))

    # Class V chromomagnetic
    d.update(Fierz_to_Bern_chrom(JMS_to_Fierz_chrom(C, 'sb'), 'sb', p))
    d.update(Fierz_to_Bern_chrom(JMS_to_Fierz_chrom(C, 'db'), 'db', p))
    d.update(Fierz_to_Bern_chrom(JMS_to_Fierz_chrom(C, 'ds'), 'ds', p))

    # Class VII
    d.update(_JMS_to_Bern_VII(Cflat, p))

    prefactor = sqrt(2)/p['GF']/4
    return {k: prefactor * v for k,v in d.items()}


def FlavorKit_to_JMS(C, scale, parameters=None):
    p = get_parameters(scale, f=5, input_parameters=parameters)
    d = json.loads(pkgutil.get_data('wcxf', 'data/flavorkit_jms.json').decode('utf8'))
    d_conj = json.loads(pkgutil.get_data('wcxf', 'data/flavorkit_jms_conj.json').decode('utf8'))
    C_out = {}
    for k, v in C.items():
        if k in d:
            C_out[d[k]] = v
        elif k in d_conj:
            C_out[d_conj[k]] = v.conjugate()
        elif k == 'AVLL_2231':
            C_out['VeeLL_1223'] = v
        elif k == 'AVRR_2231':
            C_out['VeeRR_1223'] = v
        elif k[:4] == 'K2R_':
            ind = k[4:][::-1]
            e = sqrt(4* pi * p['alpha_e'])
            if ind[1] == '1':
                m = p['m_e']
            if ind[1] == '2':
                m = p['m_mu']
            if ind[1] == '3':
                m = p['m_tau']
            C_out['egamma_' + ind] = 1/2 * e * m * v
        else:
            raise ValueError("Unexpected key: {}".format(k))
    return C_out


def JMS_to_FlavorKit(C, scale, parameters=None):
    p = get_parameters(scale, f=5, input_parameters=parameters)
    d = json.loads(pkgutil.get_data('wcxf', 'data/flavorkit_jms.json').decode('utf8'))
    d = {v: k for k, v in d.items()}  # revert dict
    d_conj = json.loads(pkgutil.get_data('wcxf', 'data/flavorkit_jms_conj.json').decode('utf8'))
    d_conj = {v: k for k, v in d_conj.items()}  # revert dict
    C_out = {}
    for k, v in C.items():
        if k in d:
            C_out[d[k]] = v
        elif k in d_conj:
            C_out[d_conj[k]] = v.conjugate()
        elif k == 'VeeLL_1223':
            C_out['AVLL_2231'] = v
        elif k == 'VeeRR_1223':
            C_out['AVRR_2231'] = v
        elif k.split('_')[0] == 'egamma':
            ind = k.split('_')[1][::-1]
            if ind[0] == ind[1]:
                continue  # diagonal dipoles are not in basis
            e = sqrt(4* pi * p['alpha_e'])
            if ind[0] == '1':
                m = p['m_e']
            if ind[0] == '2':
                m = p['m_mu']
            if ind[0] == '3':
                m = p['m_tau']
            C_out['K2R_' + ind] = 2 / e / m * v
        else:
            pass  # FlavorKit is not complete, so there will be unknown keys
    return C_out
