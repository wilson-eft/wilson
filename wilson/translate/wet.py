from math import pi, sqrt
import numpy as np
from wilson.parameters import p as default_parameters
from wilson.util.qcd import alpha_s, m_b, m_s, m_c
from wilson.util.wetutil import rotate_down, symmetrize_JMS_dict, JMS_to_array
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
        dd = 'dd'
        ij = tuple(dflav[q] for q in qq)
    elif qq == 'cu':
        dd = 'uu'
        ij = tuple(uflav[q] for q in qq)
    else:
        raise ValueError("not in Bern_I: ".format(qq))
    ji = (ij[1], ij[0])
    d = {
        '1' + 2 * qq : C["V{}LL".format(dd)][ij + ij],
        '2' + 2 * qq : C["S1{}RR".format(dd)][ji + ji].conj()
                       - C["S8{}RR".format(dd)][ji + ji].conj() / (2 * Nc),
        '3' + 2 * qq : C["S8{}RR".format(dd)][ji + ji].conj() / 2,
        '4' + 2 * qq : -C["V8{}LR".format(dd)][ij + ij],
        '5' + 2 * qq : -2 * C["V1{}LR".format(dd)][ij + ij]
                       + C["V8{}LR".format(dd)][ij + ij] / Nc,
        '1p' + 2 * qq : C["V{}RR".format(dd)][ij + ij],
        '2p' + 2 * qq : C["S1{}RR".format(dd)][ij + ij]
                       - C["S8{}RR".format(dd)][ij + ij] / (2 * Nc),
        '3p' + 2 * qq : C["S8{}RR".format(dd)][ij + ij] / 2
            }
    return d


def _Bern_to_JMS_I(C, qq):
    """From Bern to JMS basis for $\Delta F=2$ operators.
    `qq` should be 'sb', 'db', 'ds' or 'cu'"""
    if qq in ['sb', 'db', 'ds']:
        dd = 'dd'
        ij = '{}{}'.format(dflav[qq[0]] + 1, dflav[qq[1]] + 1)
    elif qq == 'cu':
        dd = 'uu'
        ij = '{}{}'.format(uflav[qq[0]] + 1, uflav[qq[1]] + 1)
    else:
        raise ValueError("not in Bern_I: ".format(qq))
    ji = ij[1] + ij[0]
    d = {"V{}LL_{}{}".format(dd, ij, ij): C['1' + 2 * qq],
         "S1{}RR_{}{}".format(dd, ji, ji): C['2' + 2 * qq].conjugate() + C['3' + 2 * qq].conjugate() / 3,
         "S8{}RR_{}{}".format(dd, ji, ji): 2 * C['3' + 2 * qq].conjugate(),
         "V1{}LR_{}{}".format(dd, ij, ij): -C['4' + 2 * qq] / 6 - C['5' + 2 * qq] / 2,
         "V8{}LR_{}{}".format(dd, ij, ij): -C['4' + 2 * qq],
         "V{}RR_{}{}".format(dd, ij, ij): C['1p' + 2 * qq],
         "S1{}RR_{}{}".format(dd, ij, ij): C['2p' + 2 * qq] + C['3p' + 2 * qq] / 3,
         "S8{}RR_{}{}".format(dd, ij, ij): 2 * C['3p' + 2 * qq],
         }
    if qq == 'cu':
        # here we need to convert some operators that are not in the basis
        for VXY in ['VuuRR', 'V1uuLR', 'V8uuLR', 'VuuLL']:
            d[VXY + '_1212'] = d.pop(VXY + '_2121').conjugate()
    return d


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
        raise ValueError("not in Flavio_I: ".format(qq))


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
        raise ValueError("not in Bern_I: ".format(qq))


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


def _Bern_to_JMS_II(C, udlnu):
    """From BernII to JMS basis for charged current process semileptonic
    operators. `udlnu` should be of the form 'udl_enu_tau', 'cbl_munu_e' etc."""
    u = uflav[udlnu[0]]
    d = dflav[udlnu[1]]
    l = lflav[udlnu[4:udlnu.find('n')]]
    lp = lflav[udlnu[udlnu.find('_',5)+1:len(udlnu)]]
    ind = udlnu[0]+udlnu[1]+udlnu[4:udlnu.find('n')]+udlnu[udlnu.find('_',5)+1
                                                                    :len(udlnu)]
    return {
        "VnueduLL_{}{}{}{}".format(lp + 1, l + 1, d + 1, u + 1): C['1' + ind].conjugate(),
        "SnueduRL_{}{}{}{}".format(lp + 1, l + 1, d + 1, u + 1): C['5' + ind].conjugate(),
        "VnueduLR_{}{}{}{}".format(lp + 1, l + 1, d + 1, u + 1): C['1p' + ind].conjugate(),
        "SnueduRR_{}{}{}{}".format(lp + 1, l + 1, d + 1, u + 1): C['5p' + ind].conjugate(),
        "TnueduRR_{}{}{}{}".format(lp + 1, l + 1, d + 1, u + 1): C['7p' + ind].conjugate()
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
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["delta"])
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
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["delta"])
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
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["delta"])
    prefactor = -sqrt(2) / p['GF'] / V[u, d] / 4
    return {k: prefactor * v for k,v in dic.items()}


## Class III ##

def _Fierz_to_JMS_III_IV_V(Fqqqq, qqqq):
    """From 4-quark Fierz to JMS basis for Classes III, IV and V.
    `qqqq` should be of the form 'sbuc', 'sdcc', 'ucuu' etc."""
    F = Fqqqq.copy()
    #case dduu
    classIII = ['sbuc', 'sbcu', 'dbuc', 'dbcu', 'dsuc', 'dscu']
    classVdduu = ['sbuu' , 'dbuu', 'dsuu', 'sbcc' , 'dbcc', 'dscc']
    if qqqq in classIII + classVdduu:
        f1 = str(dflav[qqqq[0]] + 1)
        f2 = str(dflav[qqqq[1]] + 1)
        f3 = str(uflav[qqqq[2]] + 1)
        f4 = str(uflav[qqqq[3]] + 1)
        d = {'V1udLL_' + f3 + f4 + f1 + f2: F['F' + qqqq + '1'] + F['F' + qqqq + '2'] / Nc,
            'V8udLL_' + f3 + f4 + f1 + f2: 2 * F['F' + qqqq + '2'],
            'V1duLR_' + f1 + f2 + f3 + f4: F['F' + qqqq + '3'] + F['F' + qqqq + '4'] / Nc,
            'V8duLR_' + f1 + f2 + f3 + f4: 2 * F['F' + qqqq + '4'],
            'S1udRR_' + f3 + f4 + f1 + f2: F['F' + qqqq + '5'] + F['F' + qqqq + '6'] / Nc - 4 * F['F' + qqqq + '9'] - (4 * F['F' + qqqq + '10']) / Nc,
            'S8udRR_' + f3 + f4 + f1 + f2: 2 * F['F' + qqqq + '6'] - 8 * F['F' + qqqq + '10'],
            'S1udduRR_' + f3 + f2 + f1 + f4: -((8 * F['F' + qqqq + '9']) / Nc) - 8 * F['F' + qqqq + '10'],
            'V8udduLR_' + f4 + f1 + f2 + f3: -F['F' + qqqq + '7'].conjugate(),
            'V1udduLR_' + f4 + f1 + f2 + f3: -(F['F' + qqqq + '7'].conjugate() / (2 * Nc)) - F['F' + qqqq + '8'].conjugate() / 2,
            'S8udduRR_' + f3 + f2 + f1 + f4: -16 * F['F' + qqqq + '9'],
            'V1udRR_' + f3 + f4 + f1 + f2: F['F' + qqqq + '1p'] + F['F' + qqqq + '2p'] / Nc,
            'V8udRR_' + f3 + f4 + f1 + f2: 2 * F['F' + qqqq + '2p'],
            'V1udLR_' + f3 + f4 + f1 + f2: F['F' + qqqq + '3p'] + F['F' + qqqq + '4p'] / Nc,
            'V8udLR_' + f3 + f4 + f1 + f2: 2 * F['F' + qqqq + '4p'],
            'S1udRR_' + f4 + f3 + f2 + f1: F['F' + qqqq + '5p'].conjugate() + F['F' + qqqq + '6p'].conjugate() / Nc - 4 * F['F' + qqqq + '9p'].conjugate() - (4 * F['F' + qqqq + '10p'].conjugate()) /  Nc,
            'S8udRR_' + f4 + f3 + f2 + f1: 2 * F['F' + qqqq + '6p'].conjugate() - 8 * F['F' + qqqq + '10p'].conjugate(),
            'S1udduRR_' + f4 + f1 + f2 + f3: -((8 * F['F' + qqqq + '9p'].conjugate()) / Nc) - 8 * F['F' + qqqq + '10p'].conjugate(),
            'V8udduLR_' + f3 + f2 + f1 + f4: -F['F' + qqqq + '7p'],
            'V1udduLR_' + f3 + f2 + f1 + f4: -(F['F' + qqqq + '7p'] / (2 * Nc)) - F['F' + qqqq + '8p'] / 2,
            'S8udduRR_' + f4 + f1 + f2 + f3: -16 * F['F' + qqqq + '9p'].conjugate(),
            }
        return symmetrize_JMS_dict(d)
    #case uudd
    classVuudd = ['ucdd', 'ucss','ucbb']
    if qqqq in classVuudd:
        f3 = str(uflav[qqqq[0]] + 1)
        f4 = str(uflav[qqqq[1]] + 1)
        f1 = str(dflav[qqqq[2]] + 1)
        f2 = str(dflav[qqqq[3]] + 1)
        d = {'V1udLL_' + f3 + f4 + f1 + f2: F['F' + qqqq + '1'] + F['F' + qqqq + '2'] / Nc,
            'V8udLL_' + f3 + f4 + f1 + f2: 2 * F['F' + qqqq + '2'],
            'V1duLR_' + f1 + f2 + f3 + f4: F['F' + qqqq + '3p'] + F['F' + qqqq + '4p'] / Nc,
            'V8duLR_' + f1 + f2 + f3 + f4: 2 * F['F' + qqqq + '4p'],
            'S1udRR_' + f3 + f4 + f1 + f2: F['F' + qqqq + '5'] + F['F' + qqqq + '6'] / Nc - 4 * F['F' + qqqq + '9'] - (4 * F['F' + qqqq + '10']) / Nc,
            'S8udRR_' + f3 + f4 + f1 + f2: 2 * F['F' + qqqq + '6'] - 8 * F['F' + qqqq + '10'],
            'S1udduRR_' + f3 + f2 + f1 + f4: -((8 * F['F' + qqqq + '9']) / Nc) - 8 * F['F' + qqqq + '10'],
            'V8udduLR_' + f4 + f1 + f2 + f3: -F['F' + qqqq + '7p'].conjugate(),
            'V1udduLR_' + f4 + f1 + f2 + f3: -(F['F' + qqqq + '7p'].conjugate() / (2 * Nc)) - F['F' + qqqq + '8p'].conjugate() / 2,
            'S8udduRR_' + f3 + f2 + f1 + f4: -16 * F['F' + qqqq + '9'],
            'V1udRR_' + f3 + f4 + f1 + f2: F['F' + qqqq + '1p'] + F['F' + qqqq + '2p'] / Nc,
            'V8udRR_' + f3 + f4 + f1 + f2: 2 * F['F' + qqqq + '2p'],
            'V1udLR_' + f3 + f4 + f1 + f2: F['F' + qqqq + '3'] + F['F' + qqqq + '4'] / Nc,
            'V8udLR_' + f3 + f4 + f1 + f2: 2 * F['F' + qqqq + '4'],
            'S1udRR_' + f4 + f3 + f2 + f1: F['F' + qqqq + '5p'].conjugate() + F['F' + qqqq + '6p'].conjugate() / Nc - 4 * F['F' + qqqq + '9p'].conjugate() - (4 * F['F' + qqqq + '10p'].conjugate()) /  Nc,
            'S8udRR_' + f4 + f3 + f2 + f1: 2 * F['F' + qqqq + '6p'].conjugate() - 8 * F['F' + qqqq + '10p'].conjugate(),
            'S1udduRR_' + f4 + f1 + f2 + f3: -((8 * F['F' + qqqq + '9p'].conjugate()) / Nc) - 8 * F['F' + qqqq + '10p'].conjugate(),
            'V8udduLR_' + f3 + f2 + f1 + f4: -F['F' + qqqq + '7'],
            'V1udduLR_' + f3 + f2 + f1 + f4: -(F['F' + qqqq + '7'] / (2 * Nc)) - F['F' + qqqq + '8'] / 2,
            'S8udduRR_' + f4 + f1 + f2 + f3: -16 * F['F' + qqqq + '9p'].conjugate(),
            }
        return symmetrize_JMS_dict(d)
    #case dddd
    classIV = ['sbsd', 'dbds', 'bsbd']
    classVdddd = ['sbss', 'dbdd', 'dsdd', 'sbbb', 'dbbb', 'dsss']
    classVddddind = ['sbdd', 'dsbb', 'dbss']
    classVuuuu = ['ucuu', 'cucc', 'uccc', 'cuuu']
    if qqqq in classVdddd + classIV + classVuuuu:
        # if 2nd and 4th or 1st and 3rd fields are the same, Fierz can be used
        # to express the even coeffs in terms of the odd ones
        for key in F:
            # to make sure we're not screwing things up, check that none
            # of the even WCs is actually present
            assert int(key[5:].replace('p', '')) % 2 == 1, "Unexpected key in Fierz basis: " + key
        for p in ['', 'p']:
            if qqqq in ['sbbb', 'dbbb', 'dsss', 'uccc']:
                F['F' + qqqq + '2' + p] = F['F' + qqqq + '1' + p]
                F['F' + qqqq + '4' + p] = -1 / 2 * F['F' + qqqq + '7' + p]
                F['F' + qqqq + '6' + p] = -1 / 2 * F['F' + qqqq + '5' + p] - 6 * F['F' + qqqq + '9' + p]
                F['F' + qqqq + '8' + p] = -2 * F['F' + qqqq + '3' + p]
                F['F' + qqqq + '10' + p] = -1 / 8 * F['F' + qqqq + '5' + p] + 1 / 2 * F['F' + qqqq + '9' + p]
            elif qqqq in ['sbss', 'dbdd', 'dsdd', 'sbsd', 'dbds', 'bsbd', 'ucuu']:
                notp = 'p' if p == '' else ''
                F['F' + qqqq + '2' + p] = F['F' + qqqq + '1' + p]
                F['F' + qqqq + '4' + p] = -1 / 2 * F['F' + qqqq + '7' + notp]
                F['F' + qqqq + '6' + notp] = -1 / 2 * F['F' + qqqq + '5' + notp] - 6 * F['F' + qqqq + '9' + notp]
                F['F' + qqqq + '8' + notp] = -2 * F['F' + qqqq + '3' + p]
                F['F' + qqqq + '10' + notp] = -1 / 8 * F['F' + qqqq + '5' + notp] + 1 / 2 * F['F' + qqqq + '9' + notp]
    if qqqq in classIV + classVdddd + classVddddind:
        f1 = str(dflav[qqqq[0]] + 1)
        f2 = str(dflav[qqqq[1]] + 1)
        f3 = str(dflav[qqqq[2]] + 1)
        f4 = str(dflav[qqqq[3]] + 1)
        d = {
        'VddLL_' + f3 + f4 + f1 + f2: F['F' + qqqq + '1'],
        'VddLL_' + f1 + f4 + f3 + f2: F['F' + qqqq + '2'],
        'V1ddLR_' + f1 + f2 + f3 + f4: F['F' + qqqq + '3'] + F['F' + qqqq + '4'] / Nc,
        'V8ddLR_' + f1 + f2 + f3 + f4: 2 * F['F' + qqqq + '4'],
        'S1ddRR_' + f3 + f4 + f1 + f2: F['F' + qqqq + '5'] + F['F' + qqqq + '6'] / Nc - 4 * F['F' + qqqq + '9'] - (4 * F['F' + qqqq + '10']) / Nc,
        'S8ddRR_' + f3 + f4 + f1 + f2: 2 * F['F' + qqqq + '6'] - 8 * F['F' + qqqq + '10'],
        'V8ddLR_' + f1 + f4 + f3 + f2: -F['F' + qqqq + '7'],
        'V1ddLR_' + f1 + f4 + f3 + f2: -(F['F' + qqqq + '7'] / (2 * Nc)) - F['F' + qqqq + '8'] /  2,
        'S1ddRR_' + f1 + f4 + f3 + f2: -((8 * F['F' + qqqq + '9']) / Nc) - 8 * F['F' + qqqq + '10'],
        'S8ddRR_' + f3 + f2 + f1 + f4: -16 * F['F' + qqqq + '9'],
        'VddRR_' + f3 + f4 + f1 + f2: F['F' + qqqq + '1p'],
        'VddRR_' + f1 + f4 + f3 + f2: F['F' + qqqq + '2p'],
        'V1ddLR_' + f3 + f4 + f1 + f2: F['F' + qqqq + '3p'] + F['F' + qqqq + '4p'] / Nc,
        'V8ddLR_' + f3 + f4 + f1 + f2: 2 * F['F' + qqqq + '4p'],
        'S1ddRR_' + f4 + f3 + f2 + f1: F['F' + qqqq + '5p'].conjugate() + F['F' + qqqq + '6p'].conjugate() / Nc - 4 * F['F' + qqqq + '9p'].conjugate() - (4 * F['F' + qqqq + '10p'].conjugate()) /  Nc,
        'S8ddRR_' + f4 + f3 + f2 + f1: 2 * F['F' + qqqq + '6p'].conjugate() - 8 * F['F' + qqqq + '10p'].conjugate(),
        'V8ddLR_' + f3 + f2 + f1 + f4: -F['F' + qqqq + '7p'],
        'V1ddLR_' + f3 + f2 + f1 + f4: -(F['F' + qqqq + '7p'] / (2 * Nc)) - F['F' + qqqq + '8p'] / 2,
        'S1ddRR_' + f4 + f1 + f2 + f3: -((8 * F['F' + qqqq + '9p'].conjugate()) / Nc) - 8 * F['F' + qqqq + '10p'].conjugate(),
        'S8ddRR_' + f4 + f1 + f2 + f3: -16 * F['F' + qqqq + '9p'].conjugate(),
        }
        return symmetrize_JMS_dict(d)
    #case uuuu
    if qqqq in classVuuuu:
        f1 = str(uflav[qqqq[0]] + 1)
        f2 = str(uflav[qqqq[1]] + 1)
        f3 = str(uflav[qqqq[2]] + 1)
        f4 = str(uflav[qqqq[3]] + 1)
        d = {
        'VuuLL_' + f3 + f4 + f1 + f2: F['F' + qqqq + '1'],
        'VuuLL_' + f1 + f4 + f3 + f2: F['F' + qqqq + '2'],
        'V1uuLR_' + f1 + f2 + f3 + f4: F['F' + qqqq + '3'] + F['F' + qqqq + '4'] / Nc,
        'V8uuLR_' + f1 + f2 + f3 + f4: 2 * F['F' + qqqq + '4'],
        'S1uuRR_' + f3 + f4 + f1 + f2: F['F' + qqqq + '5'] + F['F' + qqqq + '6'] / Nc - 4 * F['F' + qqqq + '9'] - (4 * F['F' + qqqq + '10']) / Nc,
        'S8uuRR_' + f3 + f4 + f1 + f2: 2 * F['F' + qqqq + '6'] - 8 * F['F' + qqqq + '10'],
        'V8uuLR_' + f1 + f4 + f3 + f2: -F['F' + qqqq + '7'],
        'V1uuLR_' + f1 + f4 + f3 + f2: -(F['F' + qqqq + '7'] / (2 * Nc)) - F['F' + qqqq + '8'] /  2,
        'S1uuRR_' + f1 + f4 + f3 + f2: -((8 * F['F' + qqqq + '9']) / Nc) - 8 * F['F' + qqqq + '10'],
        'S8uuRR_' + f3 + f2 + f1 + f4: -16 * F['F' + qqqq + '9'],
        'VuuRR_' + f3 + f4 + f1 + f2: F['F' + qqqq + '1p'],
        'VuuRR_' + f1 + f4 + f3 + f2: F['F' + qqqq + '2p'],
        'V1uuLR_' + f3 + f4 + f1 + f2: F['F' + qqqq + '3p'] + F['F' + qqqq + '4p'] / Nc,
        'V8uuLR_' + f3 + f4 + f1 + f2: 2 * F['F' + qqqq + '4p'],
        'S1uuRR_' + f4 + f3 + f2 + f1: F['F' + qqqq + '5p'].conjugate() + F['F' + qqqq + '6p'].conjugate() / Nc - 4 * F['F' + qqqq + '9p'].conjugate() - (4 * F['F' + qqqq + '10p'].conjugate()) /  Nc,
        'S8uuRR_' + f4 + f3 + f2 + f1: 2 * F['F' + qqqq + '6p'].conjugate() - 8 * F['F' + qqqq + '10p'].conjugate(),
        'V8uuLR_' + f3 + f2 + f1 + f4: -F['F' + qqqq + '7p'],
        'V1uuLR_' + f3 + f2 + f1 + f4: -(F['F' + qqqq + '7p'] / (2 * Nc)) - F['F' + qqqq + '8p'] / 2,
        'S1uuRR_' + f4 + f1 + f2 + f3: -((8 * F['F' + qqqq + '9p'].conjugate()) / Nc) - 8 * F['F' + qqqq + '10p'].conjugate(),
        'S8uuRR_' + f4 + f1 + f2 + f3: -16 * F['F' + qqqq + '9p'].conjugate()
        }
        return symmetrize_JMS_dict(d)
    raise ValueError("Case not implemented: {}".format(qqqq))

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
    classVuudd = ['ucdd', 'ucss', 'ucbb']
    if qqqq in classVuudd:
        f3 = uflav[qqqq[0]]
        f4 = uflav[qqqq[1]]
        f1 = dflav[qqqq[2]]
        f2 = dflav[qqqq[3]]
        return {
            'F' + qqqq + '1' : C["V1udLL"][f3, f4, f1, f2]
                                - C["V8udLL"][f3, f4, f1, f2] / (2 * Nc),
            'F' + qqqq + '2' : C["V8udLL"][f3, f4, f1, f2] / 2,
            'F' + qqqq + '3p' : C["V1duLR"][f1, f2, f3, f4]
                                - C["V8duLR"][f1, f2, f3, f4] / (2 * Nc),
            'F' + qqqq + '4p' : C["V8duLR"][f1, f2, f3, f4] / 2,
            'F' + qqqq + '5' : C["S1udRR"][f3, f4, f1, f2]
                                - C["S8udduRR"][f3, f2, f1, f4] / 4
                                - C["S8udRR"][f3, f4, f1, f2] / (2 * Nc),
            'F' + qqqq + '6' : -C["S1udduRR"][f3, f2, f1, f4] / 2
                                + C["S8udduRR"][f3, f2, f1, f4] /(4 * Nc)
                                + C["S8udRR"][f3, f4, f1, f2] / 2,
            'F' + qqqq + '7p' : -C["V8udduLR"][f4, f1, f2, f3].conj(),
            'F' + qqqq + '8p' : -2 * C["V1udduLR"][f4, f1, f2, f3].conj()
                                + C["V8udduLR"][f4, f1, f2, f3].conj() / Nc,
            'F' + qqqq + '9' : -C["S8udduRR"][f3, f2, f1, f4] / 16,
            'F' + qqqq + '10' : -C["S1udduRR"][f3, f2, f1, f4] / 8
                                + C["S8udduRR"][f3, f2, f1, f4] / (16 * Nc),
            'F' + qqqq + '1p' : C["V1udRR"][f3, f4, f1, f2]
                                - C["V8udRR"][f3, f4, f1, f2] / (2 * Nc),
            'F' + qqqq + '2p' : C["V8udRR"][f3, f4, f1, f2] / 2,
            'F' + qqqq + '3' : C["V1udLR"][f3, f4, f1, f2]
                                - C["V8udLR"][f3, f4, f1, f2] / (2 * Nc),
            'F' + qqqq + '4' : C["V8udLR"][f3, f4, f1, f2] / 2,
            'F' + qqqq + '5p' : C["S1udRR"][f4, f3, f2, f1].conj() -
                                C["S8udduRR"][f4, f1, f2, f3].conj() / 4
                                - C["S8udRR"][f4, f3, f2, f1].conj() / (2 * Nc),
            'F' + qqqq + '6p' : -C["S1udduRR"][f4, f1, f2, f3].conj() / 2 +
                                C["S8udduRR"][f4, f1, f2, f3].conj()/(4 * Nc)
                                + C["S8udRR"][f4, f3, f2, f1].conj() / 2,
            'F' + qqqq + '7' : -C["V8udduLR"][f3, f2, f1, f4],
            'F' + qqqq + '8' : - 2 * C["V1udduLR"][f3, f2, f1, f4]
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
                 'F'+ qqqq +'2p' : C["VddRR"][f1, f4, f3, f2],
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
    classVuuuu = ['ucuu', 'cucc', 'cuuu', 'uccc']
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
        raise ValueError("Case not implemented: {}".format(qqqq))


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
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["delta"])
    if qqqq[:2] == 'sb':
        xi = V[2, 2] * V[2, 1].conj()
    elif qqqq[:2] == 'db':
        xi = V[2, 2] * V[2, 0].conj()
    elif qqqq[:2] == 'ds':
        xi = V[2, 1] * V[2, 0].conj()
    elif qqqq[:2] == 'uc':
        xi = V[1, 2].conj() * V[0, 2]
    else:
        raise ValueError("Unexpected flavours: {}".format(qqqq[:2]))
    pf = sqrt(2) / p['GF'] / xi / 4
    qqqq_fl = qqqq[1] + qqqq[0] + qqqq[2:]  # 1st two indices flipped for flavio
    if qqqq in ['dsss', 'dsdd', 'dbbb', 'dbdd', 'sbss', 'sbbb', 'ucuu', 'uccc']:
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
    elif qqqq in ['dsuu', 'dscc', 'dsbb', 'dbuu', 'dbcc', 'dbss', 'sbuu', 'sbcc', 'sbdd', 'ucdd', 'ucss', 'ucbb']:
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
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["delta"])
    if qqqq[:2] == 'sb':
        xi = V[2, 2] * V[2, 1].conj()
    elif qqqq[:2] == 'db':
        xi = V[2, 2] * V[2, 0].conj()
    elif qqqq[:2] == 'ds':
        xi = V[2, 1] * V[2, 0].conj()
    elif qqqq[:2] == 'uc':
        xi = V[1, 2].conj() * V[0, 2]
    else:
        raise ValueError("Unexpected flavours: {}".format(qqqq[:2]))
    pf = 4 * p['GF'] /sqrt(2) * xi
    qqqq_fl = qqqq[1] + qqqq[0] + qqqq[2:]  # 1st two indices flipped for flavio
    if qqqq in ['dsss', 'dsdd', 'dbbb', 'dbdd', 'sbss', 'sbbb', 'ucuu', 'uccc']:
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
    elif qqqq in ['dsuu', 'dscc', 'dsbb', 'dbuu', 'dbcc', 'dbss', 'sbuu', 'sbcc', 'sbdd', 'ucdd', 'ucss', 'ucbb']:
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
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["delta"])
    Vtb = V[2,2]
    Vts = V[2,1]
    """From Fierz to the EOS basis for b -> s transitions.
    The arguments are dictionaries of the corresponding Fierz bases """
    dic = {
    'b->s::c1' :  -Fsbbb['Fsbbb1']/3 + 2*Fsbcc['Fsbcc1']
                    - 2 * Fsbdd['Fsbdd1'] / 3 + Fsbdd['Fsbdd2']/3 -
                    Fsbss['Fsbss1'] / 3 - 2 * Fsbuu['Fsbuu1'] / 3
                    + Fsbuu['Fsbuu2'] / 3,
     'b->s::c2' : -2 * Fsbbb['Fsbbb1'] / 9 + Fsbcc['Fsbcc1'] / 3
                    + Fsbcc['Fsbcc2'] + Fsbdd['Fsbdd1'] / 18
                    - 5 * Fsbdd['Fsbdd2'] / 18 - 2 * Fsbss['Fsbss1'] / 9
                    + Fsbuu['Fsbuu1'] / 18 - 5 * Fsbuu['Fsbuu2'] / 18,
     'b->s::c3' : -2 * Fsbbb['Fsbbb1'] / 27 + 4 * Fsbbb['Fsbbb3'] / 15
                    + 4 * Fsbbb['Fsbbb4'] / 45 + 4 * Fsbcc['Fsbcc3'] / 15
                    + 4 * Fsbcc['Fsbcc4'] / 45 - 5 * Fsbdd['Fsbdd1'] / 54
                    + Fsbdd['Fsbdd2'] / 54 + 4 * Fsbdd['Fsbdd3'] / 15
                    + 4 * Fsbdd['Fsbdd4'] / 45 - 2 * Fsbss['Fsbss1'] / 27
                    + 4 * Fsbss['Fsbss3'] / 15 + 4 * Fsbss['Fsbss4'] / 45
                    - 5 * Fsbuu['Fsbuu1'] / 54 + Fsbuu['Fsbuu2'] / 54
                    + 4 * Fsbuu['Fsbuu3'] / 15 + 4 * Fsbuu['Fsbuu4'] / 45,
     'b->s::c4' : -Fsbbb['Fsbbb1'] / 9 + 8 * Fsbbb['Fsbbb4'] / 15
                    + 8 * Fsbcc['Fsbcc4'] / 15 + Fsbdd['Fsbdd1'] / 9
                    - 2 * Fsbdd['Fsbdd2'] / 9 + 8 * Fsbdd['Fsbdd4'] / 15
                    - Fsbss['Fsbss1'] / 9 + 8 * Fsbss['Fsbss4'] / 15
                    + Fsbuu['Fsbuu1'] / 9 - 2 * Fsbuu['Fsbuu2'] / 9
                    + 8 * Fsbuu['Fsbuu4'] / 15,
     'b->s::c5' : Fsbbb['Fsbbb1'] / 54 - Fsbbb['Fsbbb3'] / 60
                 - Fsbbb['Fsbbb4'] / 180 - Fsbcc['Fsbcc3'] / 60
                 - Fsbcc['Fsbcc4'] / 180 + 5 * Fsbdd['Fsbdd1'] / 216
                 - Fsbdd['Fsbdd2'] / 216 - Fsbdd['Fsbdd3'] / 60
                 - Fsbdd['Fsbdd4'] / 180 + Fsbss['Fsbss1'] / 54
                 - Fsbss['Fsbss3'] / 60 - Fsbss['Fsbss4'] / 180
                 + 5 * Fsbuu['Fsbuu1'] / 216 - Fsbuu['Fsbuu2'] / 216
                 - Fsbuu['Fsbuu3'] / 60 - Fsbuu['Fsbuu4'] / 180,
     'b->s::c6' : Fsbbb['Fsbbb1'] / 36 - Fsbbb['Fsbbb4'] / 30
                  - Fsbcc['Fsbcc4'] / 30 - Fsbdd['Fsbdd1'] / 36
                  + Fsbdd['Fsbdd2'] / 18 - Fsbdd['Fsbdd4'] / 30
                  + Fsbss['Fsbss1'] / 36 - Fsbss['Fsbss4'] / 30
                  - Fsbuu['Fsbuu1'] / 36 + Fsbuu['Fsbuu2'] / 18
                  - Fsbuu['Fsbuu4'] / 30
                  }
    prefactor = sqrt(2)/p['GF']/Vtb/Vts.conj()/4
    return {k: prefactor * v for k,v in dic.items()}


# semileptonic operators
# arguments are of the form ddl_lnu_l' to simplify the notation

def JMS_to_Fierz_lep(C, ddll):
    """From JMS to semileptonic Fierz basis for Class V.
    `ddll` should be of the form 'sbl_enu_tau', 'dbl_munu_e' etc."""
    if ddll[:2] == 'uc':
        s = uflav[ddll[0]]
        b = uflav[ddll[1]]
        q = 'u'
    else:
        s = dflav[ddll[0]]
        b = dflav[ddll[1]]
        q = 'd'
    l = lflav[ddll[4:ddll.find('n')]]
    lp = lflav[ddll[ddll.find('_',5)+1:len(ddll)]]
    ind = ddll.replace('l_','').replace('nu_','')
    return {
        'F' + ind + '9' : C["V" + q + "eLR"][s, b, l, lp] / 2
                          + C["Ve" + q + "LL"][l, lp, s, b] / 2,
        'F' + ind + '10' : C["V" + q + "eLR"][s, b, l, lp] / 2
                           - C["Ve" + q + "LL"][l, lp, s, b] / 2,
        'F' + ind + 'S' : C["Se" + q + "RL"][lp, l, b, s].conj() / 2
                          + C["Se" + q + "RR"][l, lp, s, b] / 2,
        'F' + ind + 'P' : - C["Se" + q + "RL"][lp, l, b, s].conj() / 2
                          + C["Se" + q + "RR"][l, lp, s, b] / 2,
        'F' + ind + 'T' : C["Te" + q + "RR"][l, lp, s, b] / 2
                          + C["Te" + q + "RR"][lp, l, b, s].conj() / 2,
        'F' + ind + 'T5' : C["Te" + q + "RR"][l, lp, s, b] / 2
                           - C["Te" + q + "RR"][lp, l, b, s].conj() / 2,
        'F' + ind + '9p' : C["Ve" + q + "LR"][l, lp, s, b] / 2
                           + C["Ve" + q + "RR"][l, lp, s, b] / 2,
        'F' + ind + '10p' : -C["Ve" + q + "LR"][l, lp, s, b] / 2
                            + C["Ve" + q + "RR"][l, lp, s, b] / 2,
        'F' + ind + 'Sp' : C["Se" + q + "RL"][l, lp, s, b] / 2
                           + C["Se" + q + "RR"][lp, l, b, s].conj() / 2,
        'F' + ind + 'Pp' : C["Se" + q + "RL"][l, lp, s, b] / 2
                            - C["Se" + q + "RR"][lp, l, b, s].conj() / 2,
        }

def JMS_to_Fierz_nunu(C, ddll):
    """From JMS to semileptonic Fierz basis for Class V.
    `ddll` should be of the form 'sbl_enu_tau', 'dbl_munu_e' etc."""
    s = dflav[ddll[0]]
    b = dflav[ddll[1]]
    l = lflav[ddll[4:ddll.find('n')]]
    lp = lflav[ddll[ddll.find('_',5)+1:len(ddll)]]
    ind = ddll.replace('l_','').replace('nu_','')
    return {
        'F' + ind + 'nu' : C["VnudLL"][l, lp, s, b],
        'F' + ind + 'nup' : C["VnudLR"][l, lp, s, b]
        }


def Fierz_to_JMS_lep(C, ddll):
    """From Fierz to semileptonic JMS basis for Class V.
    `ddll` should be of the form 'sbl_enu_tau', 'dbl_munu_e' etc."""
    if ddll[:2] == 'uc':
        s = str(uflav[ddll[0]] + 1)
        b = str(uflav[ddll[1]] + 1)
        q = 'u'
    else:
        s = str(dflav[ddll[0]] + 1)
        b = str(dflav[ddll[1]] + 1)
        q = 'd'
    l = str(lflav[ddll[4:ddll.find('n')]] + 1)
    lp = str(lflav[ddll[ddll.find('_',5)+1:len(ddll)]] + 1)
    ind = ddll.replace('l_','').replace('nu_','')
    d = {
        "Ve" + q + "LL" + '_' + l + lp + s + b  : -C['F' + ind + '10'] + C['F' + ind + '9'],
        "V" + q + "eLR" + '_' + s + b + l + lp : C['F' + ind + '10'] + C['F' + ind + '9'],
        "Se" + q + "RR" + '_' + l + lp + s + b : C['F' + ind + 'P'] + C['F' + ind + 'S'],
        "Se" + q + "RL" + '_' + lp + l + b + s : -C['F' + ind + 'P'].conjugate() + C['F' + ind + 'S'].conjugate(),
        "Te" + q + "RR" + '_' + lp + l + b + s : C['F' + ind + 'T'].conjugate() - C['F' + ind + 'T5'].conjugate(),
        "Te" + q + "RR" + '_' + l + lp + s + b : C['F' + ind + 'T'] + C['F' + ind + 'T5'],
        "Ve" + q + "LR" + '_' + l + lp + s + b : -C['F' + ind + '10p'] + C['F' + ind + '9p'],
        "Ve" + q + "RR" + '_' + l + lp + s + b : C['F' + ind + '10p'] + C['F' + ind + '9p'],
        "Se" + q + "RL" + '_' + l + lp + s + b : C['F' + ind + 'Pp'] + C['F' + ind + 'Sp'],
        "Se" + q + "RR" + '_' + lp + l + b + s : -C['F' + ind + 'Pp'].conjugate() + C['F' + ind + 'Sp'].conjugate(),
    }
    return symmetrize_JMS_dict(d)

def Fierz_to_JMS_nunu(C, ddll):
    """From Fierz to semileptonic JMS basis for Class V.
    `ddll` should be of the form 'sbl_enu_tau', 'dbl_munu_e' etc."""
    s = str(dflav[ddll[0]] + 1)
    b = str(dflav[ddll[1]] + 1)
    l = str(lflav[ddll[4:ddll.find('n')]] + 1)
    lp = str(lflav[ddll[ddll.find('_',5)+1:len(ddll)]] + 1)
    ind = ddll.replace('l_','').replace('nu_','')
    d = {
        "VnudLL" + '_' + l + lp + s + b : C['F' + ind + 'nu'],
        "VnudLR" + '_' + l + lp + s + b : C['F' + ind + 'nup']
    }
    return symmetrize_JMS_dict(d)


def Fierz_to_Bern_lep(C, ddll):
    """From semileptonic Fierz basis to Bern semileptonic basis for Class V.
    C should be the corresponding leptonic Fierz basis and
    `ddll` should be of the form 'sbl_enu_tau', 'dbl_munu_e' etc."""
    ind = ddll.replace('l_','').replace('nu_','')
    dic = {
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
    }
    return dic

def Fierz_to_Bern_nunu(C, ddll):
    """From semileptonic Fierz basis to Bern semileptonic basis for Class V.
    C should be the corresponding leptonic Fierz basis and
    `ddll` should be of the form 'sbl_enu_tau', 'dbl_munu_e' etc."""
    ind = ddll.replace('l_','').replace('nu_','')
    dic = {

        'nu1' + ind : C['F' + ind + 'nu'],
        'nu1p' + ind : C['F' + ind + 'nup']
    }
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
            }


def Bern_to_Fierz_nunu(C,ddll):
    """From semileptonic Bern basis to Fierz semileptonic basis for Class V.
    C should be the corresponding leptonic Fierz basis and
    `ddll` should be of the form 'sbl_enu_tau', 'dbl_munu_e' etc."""
    ind = ddll.replace('l_','').replace('nu_','')
    return {
            'F' + ind + 'nu': C['nu1' + ind],
            'F' + ind + 'nup': C['nu1p' + ind],
            }


def Fierz_to_Flavio_lep(C, ddll, parameters, norm_gf=True):
    """From semileptonic Fierz basis to Flavio semileptonic basis for Class V.
    C should be the corresponding leptonic Fierz basis and
    `ddll` should be of the form 'sbl_enu_tau', 'dbl_munu_e' etc."""
    p = parameters
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["delta"])
    if ddll[:2] == 'sb':
        xi = V[2, 2] * V[2, 1].conj()
    elif ddll[:2] == 'db':
        xi = V[2, 2] * V[2, 0].conj()
    elif ddll[:2] == 'ds':
        xi = V[2, 1] * V[2, 0].conj()
    elif ddll[:2] == 'uc':
        xi = V[1, 2].conj() * V[0, 2]
    else:
        raise ValueError("Unexpected flavours: {}".format(ddll[:2]))
    q1, q2 = ddll[:2]
    l1 = ddll[4:ddll.find('n')]
    l2 = ddll[ddll.find('_', 5) + 1:]
    ind = q1 + q2 + l1 + l2
    # flavio has indices within currents inverted
    indfl = q2 + q1 + l2 + l1
    e = sqrt(4* pi * parameters['alpha_e'])
    if ddll[:2] == 'sb' or ddll[:2] == 'db':
        mq = parameters['m_b']
    elif ddll[:2] == 'ds':
        mq = parameters['m_s']
    elif ddll[:2] == 'uc':
        mq = parameters['m_c']
    else:
        KeyError("Not sure what to do with quark mass for flavour {}".format(ddll[:2]))
    dic = {
        "C9_" + indfl : (16 * pi**2) / e**2 * C['F' + ind + '9'],
        "C9p_" + indfl : (16 * pi**2) / e**2 * C['F' + ind + '9p'],
        "C10_" + indfl : (16 * pi**2) / e**2 * C['F' + ind + '10'],
        "C10p_" + indfl : (16 * pi**2) / e**2 * C['F' + ind + '10p'],
        "CS_" + indfl : (16 * pi**2) / e**2 / mq * C['F' + ind + 'S'],
        "CSp_" + indfl : (16 * pi**2) / e**2 / mq * C['F' + ind + 'Sp'],
        "CP_" + indfl : (16 * pi**2) / e**2 / mq * C['F' + ind + 'P'],
        "CPp_" + indfl : (16 * pi**2) / e**2 / mq * C['F' + ind + 'Pp'],
    }
    if norm_gf:
        prefactor = sqrt(2)/p['GF']/xi/4
    else:
        prefactor = 1 / xi
    return {k: prefactor * v for k,v in dic.items()}


def Fierz_to_Flavio_nunu(C, ddll, parameters, norm_gf=True):
    """From semileptonic Fierz basis to Flavio semileptonic basis for Class V.
    C should be the corresponding leptonic Fierz basis and
    `ddll` should be of the form 'sbl_enu_tau', 'dbl_munu_e' etc."""
    p = parameters
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["delta"])
    if ddll[:2] == 'sb':
        xi = V[2, 2] * V[2, 1].conj()
    elif ddll[:2] == 'db':
        xi = V[2, 2] * V[2, 0].conj()
    elif ddll[:2] == 'ds':
        xi = V[2, 1] * V[2, 0].conj()
    else:
        raise ValueError("Unexpected flavours: {}".format(ddll[:2]))
    q1, q2 = ddll[:2]
    l1 = ddll[4:ddll.find('n')]
    l2 = ddll[ddll.find('_', 5) + 1:]
    ind = q1 + q2 + l1 + l2
    # flavio has indices within currents inverted
    indnu = q2 + q1 + 'nu' + l2 + 'nu' + l1
    e = sqrt(4* pi * parameters['alpha_e'])
    dic = {
        "CL_" + indnu : (8 * pi**2) / e**2 * C['F' + ind + 'nu'],
        "CR_" + indnu : (8 * pi**2) / e**2 * C['F' + ind + 'nup']
    }
    if norm_gf:
        prefactor = sqrt(2)/p['GF']/xi/4
    else:
        prefactor = 1 / xi
    return {k: prefactor * v for k,v in dic.items()}



def Flavio_to_Fierz_lep(C, ddll, parameters, norm_gf=True):
    """From  Flavio semileptonic basis to semileptonic Fierz basis for Class V.
    C should be the corresponding leptonic Fierz basis and
    `ddll` should be of the form 'sbl_enu_tau', 'dbl_munu_e' etc."""
    p = parameters
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["delta"])
    if ddll[:2] == 'sb':
        xi = V[2, 2] * V[2, 1].conj()
    elif ddll[:2] == 'db':
        xi = V[2, 2] * V[2, 0].conj()
    elif ddll[:2] == 'ds':
        xi = V[2, 1] * V[2, 0].conj()
    elif ddll[:2] == 'uc':
        xi = V[1, 2].conj() * V[0, 2]
    else:
        raise ValueError("Unexpected flavours: {}".format(ddll[:2]))
    q1, q2 = ddll[:2]
    l1 = ddll[4:ddll.find('n')]
    l2 = ddll[ddll.find('_', 5) + 1:]
    ind = q1 + q2 + l1 + l2
    # flavio has indices within currents inverted
    indfl = q2 + q1 + l2 + l1
    e = sqrt(4* pi * parameters['alpha_e'])
    if ddll[:2] == 'sb' or ddll[:2] == 'db':
        mq = parameters['m_b']
    elif ddll[:2] == 'ds':
        mq = parameters['m_s']
    elif ddll[:2] == 'uc':
        mq = parameters['m_c']
    else:
        KeyError("Not sure what to do with quark mass for flavour {}".format(ddll[:2]))
    dic = {
        'F' + ind + '9': C["C9_" + indfl] / ((16 * pi**2) / e**2),
        'F' + ind + '9p': C["C9p_" + indfl] / ((16 * pi**2) / e**2),
        'F' + ind + '10': C["C10_" + indfl] / ((16 * pi**2) / e**2),
        'F' + ind + '10p': C["C10p_" + indfl] / ((16 * pi**2) / e**2),
        'F' + ind + 'S': C["CS_" + indfl] / ((16 * pi**2) / e**2 / mq),
        'F' + ind + 'Sp': C["CSp_" + indfl] / ((16 * pi**2) / e**2 / mq),
        'F' + ind + 'P': C["CP_" + indfl] / ((16 * pi**2) / e**2 / mq),
        'F' + ind + 'Pp': C["CPp_" + indfl] / ((16 * pi**2) / e**2 / mq),
        'F' + ind + 'T': 0,  # tensors not implemented in flavio basis yet
        'F' + ind + 'T5': 0,  # tensors not implemented in flavio basis yet
    }
    if norm_gf:
        prefactor = sqrt(2)/p['GF']/xi/4
    else:
        prefactor = 1 / xi
    return {k: v / prefactor for k, v in dic.items()}


def Flavio_to_Fierz_nunu(C, ddll, parameters, norm_gf=True):
    """From  Flavio semileptonic basis to semileptonic Fierz basis for Class V.
    C should be the corresponding leptonic Fierz basis and
    `ddll` should be of the form 'sbl_enu_tau', 'dbl_munu_e' etc."""
    p = parameters
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["delta"])
    if ddll[:2] == 'sb':
        xi = V[2, 2] * V[2, 1].conj()
    elif ddll[:2] == 'db':
        xi = V[2, 2] * V[2, 0].conj()
    elif ddll[:2] == 'ds':
        xi = V[2, 1] * V[2, 0].conj()
    else:
        raise ValueError("Unexpected flavours: {}".format(ddll[:2]))
    q1, q2 = ddll[:2]
    l1 = ddll[4:ddll.find('n')]
    l2 = ddll[ddll.find('_', 5) + 1:]
    ind = q1 + q2 + l1 + l2
    # flavio has indices within currents inverted
    indnu = q2 + q1 + 'nu' + l2 + 'nu' + l1
    e = sqrt(4* pi * parameters['alpha_e'])
    dic = {
        'F' + ind + 'nu': C["CL_" + indnu] / ((8 * pi**2) / e**2),
        'F' + ind + 'nup': C["CR_" + indnu] / ((8 * pi**2) / e**2),
    }
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
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["delta"])
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
def JMS_to_Fierz_chrom(C, qq):
    """From JMS to chromomagnetic Fierz basis for Class V.
    qq should be of the form 'sb', 'ds' etc."""
    if qq[0] in dflav:
        s = dflav[qq[0]]
        b = dflav[qq[1]]
        return {
                'F7gamma' + qq : C['dgamma'][s, b],
                'F8g' + qq : C['dG'][s, b],
                'F7pgamma' + qq : C['dgamma'][b, s].conj(),
                'F8pg' + qq : C['dG'][b, s].conj()
                    }
    else:
        u = uflav[qq[0]]
        c = uflav[qq[1]]
        return {
                'F7gamma' + qq : C['ugamma'][u, c],
                'F8g' + qq : C['uG'][u, c],
                'F7pgamma' + qq : C['ugamma'][c, u].conj(),
                'F8pg' + qq : C['uG'][c, u].conj()
                    }


def Fierz_to_JMS_chrom(C, qq):
    """From chromomagnetic Fierz to JMS basis for Class V.
    qq should be of the form 'sb', 'ds' etc."""
    if qq[0] in dflav:
        s = dflav[qq[0]] + 1
        b = dflav[qq[1]] + 1
        return {'dgamma_{}{}'.format(s, b): C['F7gamma' + qq],
                'dG_{}{}'.format(s, b): C['F8g' + qq],
                'dgamma_{}{}'.format(b, s): C['F7pgamma' + qq].conjugate(),
                'dG_{}{}'.format(b, s): C['F8pg' + qq].conjugate(),
                }
    else:
        u = uflav[qq[0]] + 1
        c = uflav[qq[1]] + 1
        return {'ugamma_{}{}'.format(u, c): C['F7gamma' + qq],
                'uG_{}{}'.format(u, c): C['F8g' + qq],
                'ugamma_{}{}'.format(c, u): C['F7pgamma' + qq].conjugate(),
                'uG_{}{}'.format(c, u): C['F8pg' + qq].conjugate(),
                }


def Fierz_to_Bern_chrom(C, dd, parameters):
    """From Fierz to chromomagnetic Bern basis for Class V.
    dd should be of the form 'sb', 'ds' etc."""
    e = sqrt(4 * pi * parameters['alpha_e'])
    gs = sqrt(4 * pi * parameters['alpha_s'])
    if dd == 'sb' or dd == 'db':
        mq = parameters['m_b']
    elif dd == 'ds':
        mq = parameters['m_s']
    else:
        KeyError("Not sure what to do with quark mass for flavour {}".format(dd))
    return {
        '7gamma' + dd : gs**2 / e / mq * C['F7gamma' + dd ],
        '8g' + dd : gs / mq * C['F8g' + dd ],
        '7pgamma' + dd : gs**2 / e /mq * C['F7pgamma' + dd],
        '8pg' + dd : gs / mq * C['F8pg' + dd]
            }


def Bern_to_Fierz_chrom(C, dd, parameters):
    """From Bern to chromomagnetic Fierz basis for Class V.
    dd should be of the form 'sb', 'ds' etc."""
    e = sqrt(4 * pi * parameters['alpha_e'])
    gs = sqrt(4 * pi * parameters['alpha_s'])
    if dd == 'sb' or dd == 'db':
        mq = parameters['m_b']
    elif dd == 'ds':
        mq = parameters['m_s']
    else:
        KeyError("Not sure what to do with quark mass for flavour {}".format(dd))
    return {
        'F7gamma' + dd : C['7gamma' + dd] / (gs**2 / e / mq),
        'F8g' + dd : C['8g' + dd] / (gs / mq),
        'F7pgamma' + dd: C['7pgamma' + dd] / (gs**2 / e /mq),
        'F8pg' + dd: C['8pg' + dd] / (gs / mq)
            }


def Fierz_to_Flavio_chrom(C, qq, parameters):
    """From Fierz to chromomagnetic Flavio basis for Class V.
    qq should be of the form 'sb', 'db' etc."""
    p = parameters
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["delta"])
    if qq == 'sb':
        xi = V[2, 2] * V[2, 1].conj()
    elif qq == 'db':
        xi = V[2, 2] * V[2, 0].conj()
    elif qq == 'ds':
        xi = V[2, 1] * V[2, 0].conj()
    elif qq == 'uc':
        xi = V[1, 2].conj() * V[0, 2]
    else:
        raise ValueError("Unexpected flavours: {}".format(qq))
    qqfl = qq[::-1]
    e = sqrt(4 * pi * parameters['alpha_e'])
    gs = sqrt(4 * pi * parameters['alpha_s'])
    if qq == 'sb' or qq == 'db':
        mq = parameters['m_b']
    elif qq == 'ds':
        mq = parameters['m_s']
    elif qq == 'uc':
        mq = parameters['m_c']
    else:
        KeyError("Not sure what to do with quark mass for flavour {}".format(qq))
    dic = {
        "C7_" + qqfl : (16 * pi**2) / e / mq * C['F7gamma' + qq],
        "C8_" + qqfl : (16 * pi**2) / gs / mq * C['F8g' + qq],
        "C7p_" + qqfl : (16 * pi**2) / e / mq * C['F7pgamma' + qq],
        "C8p_" + qqfl : (16 * pi**2) / gs / mq * C['F8pg' + qq]
            }
    prefactor = sqrt(2)/p['GF']/xi/4
    return {k: prefactor * v for k, v in dic.items()}


def Flavio_to_Fierz_chrom(C, qq, parameters):
    """From Flavio to chromomagnetic Fierz basis for Class V.
    qq should be of the form 'sb', 'db' etc."""
    p = parameters
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["delta"])
    if qq == 'sb':
        xi = V[2, 2] * V[2, 1].conj()
    elif qq == 'db':
        xi = V[2, 2] * V[2, 0].conj()
    elif qq == 'ds':
        xi = V[2, 1] * V[2, 0].conj()
    elif qq == 'uc':
        xi = V[1, 2].conj() * V[0, 2]
    else:
        raise ValueError("Unexpected flavours: {}".format(qq))
    qqfl = qq[::-1]
    e = sqrt(4 * pi * parameters['alpha_e'])
    gs = sqrt(4 * pi * parameters['alpha_s'])
    if qq == 'sb' or qq == 'db':
        mq = parameters['m_b']
    elif qq == 'ds':
        mq = parameters['m_s']
    elif qq == 'uc':
        mq = parameters['m_c']
    else:
        KeyError("Not sure what to do with quark mass for flavour {}".format(qq))
    dic = {
        'F7gamma' + qq: C["C7_" + qqfl] / ((16 * pi**2) / e / mq),
        'F8g' + qq: C["C8_" + qqfl] / ((16 * pi**2) / gs / mq),
        'F7pgamma' + qq: C["C7p_" + qqfl] / ((16 * pi**2) / e / mq),
        'F8pg' + qq: C["C8p_" + qqfl] / ((16 * pi**2) / gs / mq)
            }
    prefactor = sqrt(2)/p['GF']/xi/4
    return {k: v / prefactor for k, v in dic.items()}


def Fierz_to_EOS_chrom(C, dd, parameters):
    """From Fierz to chromomagnetic EOS basis for Class V.
    dd should be of the form 'sb', 'ds' etc."""
    p = parameters
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["delta"])
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


def _JMS_to_Flavio_VII(C, parameters):
    """From JMS to flavio basis for class VII, i.e. flavour blind operators."""
    d = {}
    dtrans = json.loads(pkgutil.get_data('wilson', 'data/flavio_jms_vii.json').decode('utf8'))
    for cj, cf in dtrans.items():
        d[cf] = C.get(cj, 0)
    gs = sqrt(4 * pi * parameters['alpha_s'])
    e = sqrt(4 * pi * parameters['alpha_e'])
    preC7 = 16 * pi**2 / e
    preC8 = 16 * pi**2 / gs
    d['C8_uu'] = preC8 / parameters['m_u'] * C.get('uG_11', 0)
    d['C8_cc'] = preC8 / parameters['m_c'] * C.get('uG_22', 0)
    d['C8_dd'] = preC8 / parameters['m_d'] * C.get('dG_11', 0)
    d['C8_ss'] = preC8 / parameters['m_s'] * C.get('dG_22', 0)
    d['C8_bb'] = preC8 / parameters['m_b'] * C.get('dG_33', 0)
    d['C7_uu'] = preC7 / parameters['m_u'] * C.get('ugamma_11', 0)
    d['C7_cc'] = preC7 / parameters['m_c'] * C.get('ugamma_22', 0)
    d['C7_dd'] = preC7 / parameters['m_d'] * C.get('dgamma_11', 0)
    d['C7_ss'] = preC7 / parameters['m_s'] * C.get('dgamma_22', 0)
    d['C7_bb'] = preC7 / parameters['m_b'] * C.get('dgamma_33', 0)
    d['C7_ee'] = preC7 / parameters['m_e'] * C.get('egamma_11', 0)
    d['C7_mumu'] = preC7 / parameters['m_mu'] * C.get('egamma_22', 0)
    d['C7_tautau'] = preC7 / parameters['m_tau'] * C.get('egamma_33', 0)
    preGF = sqrt(2) / parameters['GF'] / 4
    return {k: preGF * v for k,v in d.items()}


def _Flavio_to_JMS_VII(C, parameters):
    """From flavio to JMS basis for class VII, i.e. flavour blind operators."""
    d = {}
    dtrans = json.loads(pkgutil.get_data('wilson', 'data/flavio_jms_vii.json').decode('utf8'))
    for cj, cf in dtrans.items():
        d[cj] = C.get(cf, 0)
    gs = sqrt(4 * pi * parameters['alpha_s'])
    e = sqrt(4 * pi * parameters['alpha_e'])
    preC7 = 16 * pi**2 / e
    preC8 = 16 * pi**2 / gs
    d['uG_11'] = parameters['m_u'] / preC8 * C.get('C8_uu', 0)
    d['uG_22'] = parameters['m_c'] / preC8 * C.get('C8_cc', 0)
    d['dG_11'] = parameters['m_d'] / preC8 * C.get('C8_dd', 0)
    d['dG_22'] = parameters['m_s'] / preC8 * C.get('C8_ss', 0)
    d['dG_33'] = parameters['m_b'] / preC8 * C.get('C8_bb', 0)
    d['ugamma_11'] = parameters['m_u'] / preC7 * C.get('C7_uu', 0)
    d['ugamma_22'] = parameters['m_c'] / preC7 * C.get('C7_cc', 0)
    d['dgamma_11'] = parameters['m_d'] / preC7 * C.get('C7_dd', 0)
    d['dgamma_22'] = parameters['m_s'] / preC7 * C.get('C7_ss', 0)
    d['dgamma_33'] = parameters['m_b'] / preC7 * C.get('C7_bb', 0)
    d['egamma_11'] = parameters['m_e'] / preC7 * C.get('C7_ee', 0)
    d['egamma_22'] = parameters['m_mu'] / preC7 * C.get('C7_mumu', 0)
    d['egamma_33'] = parameters['m_tau'] / preC7 * C.get('C7_tautau', 0)
    preGF = sqrt(2) / parameters['GF'] / 4
    return {k: v / preGF for k,v in d.items()}


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
              'Vus', 'Vub', 'Vcb', 'delta',
              'm_e', 'm_mu', 'm_tau', ]:
        parameters[k] = p[k]
    return parameters


# final dicitonaries

def JMS_to_EOS(Cflat, scale, parameters=None, sectors=None):
    p = get_parameters(scale, f=5, input_parameters=parameters)
    C = JMS_to_array(Cflat, sectors=sectors)
    d={}

    # Class II
    for l in ['e','mu']:
        d.update(_BernII_to_EOS_II(_JMS_to_Bern_II(C, 'ub'+'l_'+l+'nu_'+l),
                                          'ub'+'l_'+l+'nu_'+l,
                                        p))
        d.update(_BernII_to_EOS_II(_JMS_to_Bern_II(C, 'cb'+'l_'+l+'nu_'+l),
                                          'cb'+'l_'+l+'nu_'+l,
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


def JMS_to_flavio(Cflat, scale, parameters=None, sectors=None):
    p = get_parameters(scale, f=5, input_parameters=parameters)
    C = JMS_to_array(Cflat, sectors=sectors)
    d={}

    # Class I
    for qq in ['sb', 'db', 'ds', 'cu']:
        if sectors is None or 2*qq in sectors or (qq == 'ds' and 2*'sd' in sectors):
            d.update(_BernI_to_Flavio_I(_JMS_to_Bern_I(C, qq), qq))

    # Class II
    for l in lflav.keys():
        for lp in lflav.keys():
            for qq in ['cb', 'ub', 'us', 'cs', 'cd', 'ud']:
                if sectors is None or qq+l+'nu' in sectors:
                    d.update(_BernII_to_Flavio_II(_JMS_to_Bern_II(C,
                                                  qq+'l_'+l+'nu_'+lp),
                                                  qq+'l_'+l+'nu_'+lp, p))

    # Class V semileptonic
    for l in lflav.keys():
        for lp in lflav.keys():
            # ddnunu
            if sectors is None or 'sbnunu' in sectors:
                d.update(Fierz_to_Flavio_nunu(JMS_to_Fierz_nunu(C,
                                            'sb'+'l_'+l+'nu_'+lp),
                                            'sb'+'l_'+l+'nu_'+lp, p,
                                            norm_gf=True))
            if sectors is None or 'dbnunu' in sectors:
                d.update(Fierz_to_Flavio_nunu(JMS_to_Fierz_nunu(C,
                                            'db'+'l_'+l+'nu_'+lp),
                                            'db'+'l_'+l+'nu_'+lp, p,
                                            norm_gf=True))
            if sectors is None or 'sdnunu' in sectors:
                d.update(Fierz_to_Flavio_nunu(JMS_to_Fierz_nunu(C,
                                            'ds'+'l_'+l+'nu_'+lp),
                                            'ds'+'l_'+l+'nu_'+lp, p,
                                            norm_gf=True))
            # ddll
            if sectors is None or ('sb' in sectors and l == lp) or ('sb'+ l + lp in sectors):
                d.update(Fierz_to_Flavio_lep(JMS_to_Fierz_lep(C,
                                            'sb'+'l_'+l+'nu_'+lp),
                                            'sb'+'l_'+l+'nu_'+lp, p,
                                            norm_gf=True))
            if sectors is None or ('db' in sectors and l == lp) or ('db'+ l + lp in sectors):
                d.update(Fierz_to_Flavio_lep(JMS_to_Fierz_lep(C,
                                            'db'+'l_'+l+'nu_'+lp),
                                            'db'+'l_'+l+'nu_'+lp, p,
                                            norm_gf=True))
            # not how both sd<->ds and l,lp<->lp,l are interchanged!
            if sectors is None or ('sd' in sectors and l == lp) or ('sd'+ lp + l in sectors):
                d.update(Fierz_to_Flavio_lep(JMS_to_Fierz_lep(C,
                                            'ds'+'l_'+l+'nu_'+lp),
                                            'ds'+'l_'+l+'nu_'+lp, p,
                                            norm_gf=True))
            # uull
            if sectors is None or ('cu' in sectors and l == lp):
                if l == lp:
                    d.update(Fierz_to_Flavio_lep(JMS_to_Fierz_lep(C,
                                                'uc'+'l_'+l+'nu_'+lp),
                                                'uc'+'l_'+l+'nu_'+lp, p,
                                                norm_gf=True))

    # Class V non-leptonic
    for qq1 in ['ds', 'sb', 'db', 'uc']:
        if sectors is None or qq1 in sectors or (qq1 == 'ds' and 'sd' in sectors) or (qq1 == 'uc' and 'cu' in sectors):
            for qq2 in ['uu', 'dd', 'ss', 'cc', 'bb']:
                qqqq = qq1 + qq2
                d.update(_Fierz_to_Flavio_V(_JMS_to_Fierz_III_IV_V(C, qqqq),
                                        qqqq, p))

    # Class V chromomagnetic
    if sectors is None or 'sb' in sectors:
        d.update(Fierz_to_Flavio_chrom(JMS_to_Fierz_chrom(C, 'sb'), 'sb', p))
    if sectors is None or 'db' in sectors:
        d.update(Fierz_to_Flavio_chrom(JMS_to_Fierz_chrom(C, 'db'), 'db', p))
    if sectors is None or 'sd' in sectors:
        d.update(Fierz_to_Flavio_chrom(JMS_to_Fierz_chrom(C, 'ds'), 'ds', p))
    if sectors is None or 'cu' in sectors:
        d.update(Fierz_to_Flavio_chrom(JMS_to_Fierz_chrom(C, 'uc'), 'uc', p))

    # Class VII
    if sectors is None or 'dF=0' in sectors or 'ffnunu' in sectors:
        d.update(_JMS_to_Flavio_VII(Cflat, p))

    # LFV
    dlep = {}
    if sectors is None or bool(set(sectors) & {'nunumue', 'nunumutau', 'nunutaue'}):
        dlep.update(json.loads(pkgutil.get_data('wilson', 'data/flavio_jms_nunull.json').decode('utf8')))
    if sectors is None or bool(set(sectors) & {'mutau', 'mue', 'taue', 'tauetaue', 'taumutaumu', 'muemue', 'muemutau', 'etauemu', 'tauetaumu'}):
        dlep.update(json.loads(pkgutil.get_data('wilson', 'data/flavio_jms_lfv.json').decode('utf8')))
    for jkey, fkey in dlep.items():
        if jkey in Cflat:
            d[fkey] = Cflat[jkey]

    return d


def Bern_to_flavio(C_incomplete, scale, parameters=None, sectors=None):
    p = get_parameters(scale, f=5, input_parameters=parameters)
    # fill in zeros for missing coefficients
    wc_keys = set(wcxf.Basis['WET', 'Bern'].all_wcs)
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
                                         norm_gf=True))
            d.update(Fierz_to_Flavio_nunu(Bern_to_Fierz_nunu(C,
                                        'sb'+'l_'+l+'nu_'+lp),
                                        'sb'+'l_'+l+'nu_'+lp, p,
                                        norm_gf=True))
            d.update(Fierz_to_Flavio_nunu(Bern_to_Fierz_nunu(C,
                                        'db'+'l_'+l+'nu_'+lp),
                                         'db'+'l_'+l+'nu_'+lp, p,
                                         norm_gf=True))
            d.update(Fierz_to_Flavio_nunu(Bern_to_Fierz_nunu(C,
                                        'ds'+'l_'+l+'nu_'+lp),
                                         'ds'+'l_'+l+'nu_'+lp, p,
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

    prefactor = sqrt(2)/p['GF']/4
    return {k: v / prefactor for k,v in d.items()}



def flavio_to_Bern(C_incomplete, scale, parameters=None, sectors=None):
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
            if l == lp:  # l+l- only for l=l'
                d.update(Fierz_to_Bern_lep(Flavio_to_Fierz_lep(C,
                                            'ds'+'l_'+l+'nu_'+lp, p,
                                            norm_gf=True),
                                            'ds'+'l_'+l+'nu_'+lp))
            d.update(Fierz_to_Bern_nunu(Flavio_to_Fierz_nunu(C,
                                        'sb'+'l_'+l+'nu_'+lp, p,
                                        norm_gf=True),
                                        'sb'+'l_'+l+'nu_'+lp))
            d.update(Fierz_to_Bern_nunu(Flavio_to_Fierz_nunu(C,
                                        'db'+'l_'+l+'nu_'+lp, p,
                                        norm_gf=True),
                                        'db'+'l_'+l+'nu_'+lp))
            d.update(Fierz_to_Bern_nunu(Flavio_to_Fierz_nunu(C,
                                        'ds'+'l_'+l+'nu_'+lp, p,
                                        norm_gf=True),  # l+l- only for l=l'
                                        'ds'+'l_'+l+'nu_'+lp),  # l+l- only for l=l'
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

    prefactor = sqrt(2)/p['GF']/4
    return {k: prefactor * v for k,v in d.items()}

def JMS_to_FormFlavor(Cflat, scale, parameters=None, sectors=None):
    p = get_parameters(scale, f=5, input_parameters=parameters)
    C = JMS_to_array(Cflat, sectors=sectors)
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


def JMS_to_Bern(Cflat, scale, parameters=None, sectors=None):
    p = get_parameters(scale, f=5, input_parameters=parameters)
    C = JMS_to_array(Cflat, sectors=sectors)
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
            d.update(Fierz_to_Bern_nunu(JMS_to_Fierz_nunu(C, 'sb'+'l_'+l+'nu_'+lp)
                                                         ,'sb'+'l_'+l+'nu_'+lp))
            d.update(Fierz_to_Bern_nunu(JMS_to_Fierz_nunu(C, 'db'+'l_'+l+'nu_'+lp)
                                                         ,'db'+'l_'+l+'nu_'+lp))
            d.update(Fierz_to_Bern_nunu(JMS_to_Fierz_nunu(C, 'ds'+'l_'+l+'nu_'+lp)
                                                         ,'ds'+'l_'+l+'nu_'+lp))

    # Class V chromomagnetic
    d.update(Fierz_to_Bern_chrom(JMS_to_Fierz_chrom(C, 'sb'), 'sb', p))
    d.update(Fierz_to_Bern_chrom(JMS_to_Fierz_chrom(C, 'db'), 'db', p))
    d.update(Fierz_to_Bern_chrom(JMS_to_Fierz_chrom(C, 'ds'), 'ds', p))

    prefactor = sqrt(2)/p['GF']/4
    return {k: prefactor * v for k,v in d.items()}


def Bern_to_JMS(C_incomplete, scale, parameters=None, sectors=None):
    p = get_parameters(scale, f=5, input_parameters=parameters)
    # fill in zeros for missing coefficients
    wc_keys = wcxf.Basis['WET', 'Bern'].all_wcs
    C = {k: C_incomplete.get(k, 0) for k in wc_keys}
    d = {}

    # Class I
    for qq in ['sb', 'db', 'ds', 'cu']:
        d.update(_Bern_to_JMS_I(C, qq))

    # Class II
    for l in lflav.keys():
        for lp in lflav.keys():
            for qq in ['cb', 'ub', 'us', 'cs', 'cd', 'ud']:
                d.update(_Bern_to_JMS_II(C, qq+'l_'+l+'nu_'+lp))


    # Class V
    for u1 in uflav.keys():
        for u2 in uflav.keys():
            d.update(_Fierz_to_JMS_III_IV_V(_Bern_to_Fierz_III_IV_V(C,
                                                      'sb'+u1+u2), 'sb'+u1+u2))

            d.update(_Fierz_to_JMS_III_IV_V(_Bern_to_Fierz_III_IV_V(C,
                                                      'db'+u1+u2), 'db'+u1+u2))

            d.update(_Fierz_to_JMS_III_IV_V(_Bern_to_Fierz_III_IV_V(C,
                                                      'ds'+u1+u2), 'ds'+u1+u2))

    for qqqq in ['sbdd', 'sbss', 'dbdd', 'dbss', 'dbbb', 'sbbb',
                 'dbds', 'sbsd', 'dsbb',
                 'dsss', 'dsdd',
                 ]:
        d.update(_Fierz_to_JMS_III_IV_V(_Bern_to_Fierz_III_IV_V(C, qqqq), qqqq))


    # Class V semileptonic
    for l in lflav.keys():
        for lp in lflav.keys():
            for qq in ['sb', 'db', 'ds']:
                d.update(Fierz_to_JMS_lep(Bern_to_Fierz_lep(C,
                                            qq+'l_'+l+'nu_'+lp),
                                            qq+'l_'+l+'nu_'+lp))
                d.update(Fierz_to_JMS_nunu(Bern_to_Fierz_nunu(C,
                                            qq+'l_'+l+'nu_'+lp),
                                            qq+'l_'+l+'nu_'+lp))

    # Class V chromomagnetic
    for qq in ['sb', 'db', 'ds']:
        d.update(Fierz_to_JMS_chrom(Bern_to_Fierz_chrom(C, qq, p), qq))

    prefactor = 4 * p['GF'] / sqrt(2)
    return {k: prefactor * v for k,v in d.items()}


def flavio_to_JMS(C_incomplete, scale, parameters=None, sectors=None):
    p = get_parameters(scale, f=5, input_parameters=parameters)
    # fill in zeros for missing coefficients
    wc_keys = wcxf.Basis['WET', 'flavio'].all_wcs
    C = {k: C_incomplete.get(k, 0) for k in wc_keys}
    d = {}

    # Class I
    for qq in ['bs', 'bd', 'sd', 'uc']:
        qqr = qq[::-1]
        if sectors is None or 2*qqr in sectors or 2*qq in sectors:
            d.update(_Bern_to_JMS_I(_FlavioI_to_Bern_I(C, qq), qqr))

    # Class II
    for l in lflav.keys():
        for lp in lflav.keys():
            for qq in ['cb', 'ub', 'us', 'cs', 'cd', 'ud']:
                if sectors is None or qq+l+'nu' in sectors:
                    d.update(_Bern_to_JMS_II(_FlavioII_to_BernII(C,
                                             qq+'l_'+l+'nu_'+lp, p),
                                             qq+'l_'+l+'nu_'+lp))

    # Class V semileptonic
    for l in lflav.keys():
        for lp in lflav.keys():
            # ddll
            if sectors is None or ('sb' in sectors and l == lp) or ('sb'+ l + lp in sectors):
                d.update(Fierz_to_JMS_lep(Flavio_to_Fierz_lep(C,
                                            'sb'+'l_'+l+'nu_'+lp, p),
                                            'sb'+'l_'+l+'nu_'+lp))
            if sectors is None or ('db' in sectors and l == lp) or ('db'+ l + lp in sectors):
                d.update(Fierz_to_JMS_lep(Flavio_to_Fierz_lep(C,
                                            'db'+'l_'+l+'nu_'+lp, p),
                                            'db'+'l_'+l+'nu_'+lp))
            # not how both sd<->ds and l,lp<->lp,l are interchanged!
            if sectors is None or ('sd' in sectors and l == lp) or ('sd'+ lp + l in sectors):
                d.update(Fierz_to_JMS_lep(Flavio_to_Fierz_lep(C,
                                            'ds'+'l_'+l+'nu_'+lp, p),
                                            'ds'+'l_'+l+'nu_'+lp))
            # ddnunu
            if sectors is None or 'sbnunu' in sectors:
                d.update(Fierz_to_JMS_nunu(Flavio_to_Fierz_nunu(C,
                                            'sb'+'l_'+l+'nu_'+lp, p),
                                            'sb'+'l_'+l+'nu_'+lp))
            if sectors is None or 'dbnunu' in sectors:
                d.update(Fierz_to_JMS_nunu(Flavio_to_Fierz_nunu(C,
                                            'db'+'l_'+l+'nu_'+lp, p),
                                            'db'+'l_'+l+'nu_'+lp))
            if sectors is None or 'sdnunu' in sectors:
                d.update(Fierz_to_JMS_nunu(Flavio_to_Fierz_nunu(C,
                                            'ds'+'l_'+l+'nu_'+lp, p),
                                            'ds'+'l_'+l+'nu_'+lp))
            # uull
            if sectors is None or ('cu' in sectors and l == lp):
                if  l == lp:
                    d.update(Fierz_to_JMS_lep(Flavio_to_Fierz_lep(C,
                                                'uc'+'l_'+l+'nu_'+lp, p),
                                                'uc'+'l_'+l+'nu_'+lp))


    # Class V non-leptonic
    for qq1 in ['ds', 'sb', 'db', 'uc']:
        if sectors is None or qq1 in sectors or (qq1 == 'ds' and 'sd' in sectors) or (qq1 == 'uc' and 'cu' in sectors):
            for qq2 in ['uu', 'dd', 'ss', 'cc', 'bb']:
                qqqq = qq1 + qq2
                d.update(_Fierz_to_JMS_III_IV_V(_Flavio_to_Fierz_V(C, qqqq, p),
                                                qqqq))

    # Class V chromomagnetic
    if sectors is None or 'sb' in sectors:
        d.update(Fierz_to_JMS_chrom(Flavio_to_Fierz_chrom(C, 'sb', p), 'sb'))
    if sectors is None or 'db' in sectors:
        d.update(Fierz_to_JMS_chrom(Flavio_to_Fierz_chrom(C, 'db', p), 'db'))
    if sectors is None or 'sd' in sectors:
        d.update(Fierz_to_JMS_chrom(Flavio_to_Fierz_chrom(C, 'ds', p), 'ds'))
    if sectors is None or 'cu' in sectors:
        d.update(Fierz_to_JMS_chrom(Flavio_to_Fierz_chrom(C, 'uc', p), 'uc'))

    # Class VII
    if sectors is None or 'dF=0' in sectors or 'ffnunu' in sectors:
        d.update(_Flavio_to_JMS_VII(C, p))

    dlep = {}
    # LFV & ddll
    if sectors is None or bool(set(sectors) & {'nunumue', 'nunumutau', 'nunutaue'}):
        dlep.update(json.loads(pkgutil.get_data('wilson', 'data/flavio_jms_nunull.json').decode('utf8')))
    if sectors is None or bool(set(sectors) & {'mutau', 'mue', 'taue', 'tauetaue', 'taumutaumu', 'muemue', 'muemutau', 'etauemu', 'tauetaumu'}):
        dlep.update(json.loads(pkgutil.get_data('wilson', 'data/flavio_jms_lfv.json').decode('utf8')))
    for jkey, fkey in dlep.items():
        if fkey in C:
            d[jkey] = C[fkey]

    return d


def FlavorKit_to_JMS(C, scale, parameters=None, sectors=None):
    p = get_parameters(scale, f=5, input_parameters=parameters)
    d = json.loads(pkgutil.get_data('wilson', 'data/flavorkit_jms.json').decode('utf8'))
    d_conj = json.loads(pkgutil.get_data('wilson', 'data/flavorkit_jms_conj.json').decode('utf8'))
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


def JMS_to_FlavorKit(C, scale, parameters=None, sectors=None):
    p = get_parameters(scale, f=5, input_parameters=parameters)
    d = json.loads(pkgutil.get_data('wilson', 'data/flavorkit_jms.json').decode('utf8'))
    d = {v: k for k, v in d.items()}  # revert dict
    d_conj = json.loads(pkgutil.get_data('wilson', 'data/flavorkit_jms_conj.json').decode('utf8'))
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
