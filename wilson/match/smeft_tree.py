from math import sqrt, log, exp, pi
from numpy import einsum
import numpy as np


Nc = 3
kd = np.eye(3)
kdkd = einsum("pr,st",kd,kd)


def _match_all_array(C, p):

    # AUXILIARY FUNCTIONS

    mW = p['m_W']
    GF = p['GF']
    GFx = GF - sqrt(2)/4 * ( -C['ll'][1,0,0,1] - C['ll'][0,1,1,0] + 2*C['phil3'][1,1] + 2*C['phil3'][0,0] )

    vT = sqrt(1/sqrt(2)/abs(GFx))
    eps = C["phiWB"] * vT**2
    g2b = 2*p["m_W"]/vT

    alpha_e = p['alpha_e']
    eb = sqrt(4*pi*alpha_e)
    g1b = eb*g2b/sqrt(g2b**2-eb**2) + eb**2*g2b/(g2b**2-eb**2) * eps

    mZ = sqrt(vT**2 / 4 * (1 + vT**2 / 2 * C["phiD"]) * (g2b**2 + g1b**2) + vT**2 / 2 * eps * g1b * g2b)

    g1bar = g1b

    # DEFINE DAGGERED WCs
    for _f in ['ude']:
        for _b in ['B', 'G', 'phi', 'W' ]:
            if _f + _b in C:
                C[_f + _b + 'Dag'] = C[_f + _b].conjugate().T
    C['phiudDag'] = C['phiud'].conjugate().T
    C['ledqDag'] = einsum('rpts', C['ledq']).conjugate()
    C['lequ1Dag'] = einsum('rpts', C['lequ1']).conjugate()
    C['lequ3Dag'] = einsum('rpts', C['lequ3']).conjugate()
    C['quqd1Dag'] = einsum('rpts', C['quqd1']).conjugate()
    C['quqd8Dag'] = einsum('rpts', C['quqd8']).conjugate()

    # MATCHING CONDITIONS
    c = {}
        
    c['dgamma'] = (-(vT*(-2*mW*C["dB"] + g1bar*vT*C["dW"]))/(2.*sqrt(2)*mZ))
    c['dG'] = ((vT*C["dG"])/sqrt(2))
    c['egamma'] = (-(vT*(-2*mW*C["eB"] + g1bar*vT*C["eW"]))/(2.*sqrt(2)*mZ))
    # c['nugamma'] = (0)
    c['ugamma'] = ((vT*(2*mW*C["uB"] + g1bar*vT*C["uW"]))/(2.*sqrt(2)*mZ))
    c['uG'] = ((vT*C["uG"])/sqrt(2))
    c['Gtilde'] = (C["Gtilde"])
    c['G'] = (C["G"])
    # c['SdddLL'] = (0)
    # c['SdddLR'] = (0)
    # c['SdduLR'] = (0)
    # c['SuddLR'] = (0)
    # c['SdddRL'] = (0)
    # c['SdddRR'] = (0)
    # c['SuddRR'] = (0)
    c['SduuLL'] = (einsum("rpst",C["qqql"]) - einsum("rspt",C["qqql"]) + einsum("srpt",C["qqql"]))
    c['SuddLL'] = (einsum("rpst",C["qqql"]) - einsum("rspt",C["qqql"]) + einsum("srpt",C["qqql"]))
    c['SduuLR'] = (-C["qque"] - einsum("rpst",C["qque"]))
    c['SuudLR'] = np.zeros((3,3,3,3))
    c['SdduRL'] = np.zeros((3,3,3,3))
    c['SdudRL'] = (-C["duql"])
    c['SduuRL'] = (C["duql"])
    c['SuudRL'] = np.zeros((3,3,3,3))
    c['SduuRR'] = (C["duue"])
    # c['SnudLL'] = (0)
    # c['SnueduLL'] = (0)
    # c['SnueLL'] = (0)
    # c['SnuuLL'] = (0)
    # c['SnudLR'] = (0)
    # c['SnueduLR'] = (0)
    # c['SnueLR'] = (0)
    # c['SnuuLR'] = (0)
    # c['TnudLL'] = (0)
    # c['TnueduLL'] = (0)
    # c['TnueLL'] = (0)
    # c['TnuuLL'] = (0)
    # c['VnueduRL'] = (0)
    # c['VnueduRR'] = (0)
    # c['SnunuLL'] = (0)
    c['V1udLL'] = (C["qq1"] - C["qq3"] + einsum("stpr",C["qq1"]) - einsum("stpr",C["qq3"]) + ((16*mW**4 + 4*mW**2*mZ**2 - 2*mZ**4 + vT**2*((8*mW**4 + mZ**4)*C["phiD"] + 2*g1bar*mW*(8*mW**2 + mZ**2)*vT*C["phiWB"]))*kdkd)/(18.*mZ**4*vT**2) + (2*(-einsum("pt,sr",kd,kd) + vT**2*(einsum("ptsr",C["qq3"]) + einsum("srpt",C["qq3"]) - einsum("pt,sr",C["phiq3"],kd) - einsum("sr,pt",C["phiq3"],kd))))/(Nc*vT**2) + (-((2*mW**2 + mZ**2)*einsum("pr,st",C["phiq1"],kd)) + (2*mW**2 + mZ**2)*einsum("pr,st",C["phiq3"],kd) + (4*mW**2 - mZ**2)*(einsum("st,pr",C["phiq1"],kd) + einsum("st,pr",C["phiq3"],kd)))/(3.*mZ**2))
    c['V8udLL'] = (4*(einsum("ptsr",C["qq3"]) + einsum("srpt",C["qq3"]) - einsum("pt,sr",kd,kd)/vT**2 - einsum("pt,sr",C["phiq3"],kd) - einsum("sr,pt",C["phiq3"],kd)))
    c['VddLL'] = ((18*C["qq1"] + 18*C["qq3"] + 18*einsum("stpr",C["qq1"]) + 18*einsum("stpr",C["qq3"]) - ((2*mW**2 + mZ**2)*((2*(2*mW**2 + mZ**2) + vT**2*(2*mW**2*C["phiD"] - mZ**2*C["phiD"] + 4*g1bar*mW*vT*C["phiWB"]))*kdkd + 6*mZ**2*vT**2*(einsum("pr,st",C["phiq1"],kd) + einsum("pr,st",C["phiq3"],kd) + einsum("st,pr",C["phiq1"],kd) + einsum("st,pr",C["phiq3"],kd))))/(mZ**4*vT**2))/36.)
    c['VedLL'] = ((6*C["lq1"] + 6*C["lq3"] - ((8*mW**4 - 2*mZ**4 + (4*mW**4 + mZ**4)*vT**2*C["phiD"] + 8*g1bar*mW**3*vT**3*C["phiWB"])*kdkd)/(mZ**4*vT**2) - 2*(einsum("pr,st",C["phil1"],kd) + einsum("pr,st",C["phil3"],kd) - 3*(einsum("st,pr",C["phiq1"],kd) + einsum("st,pr",C["phiq3"],kd))) - (4*mW**2*(einsum("pr,st",C["phil1"],kd) + einsum("pr,st",C["phil3"],kd) + 3*(einsum("st,pr",C["phiq1"],kd) + einsum("st,pr",C["phiq3"],kd))))/mZ**2)/6.)
    c['VeeLL'] = ((2*C["ll"] + 2*einsum("ptsr",C["ll"]) + 2*einsum("srpt",C["ll"]) + 2*einsum("stpr",C["ll"]) + ((-2*mW**2 + mZ**2)*((4*mW**2 - 2*mZ**2 + vT**2*((2*mW**2 + mZ**2)*C["phiD"] + 4*g1bar*mW*vT*C["phiWB"]))*kdkd + 2*mZ**2*vT**2*einsum("pr,st",C["phil1"],kd) + 2*mZ**2*vT**2*einsum("pr,st",C["phil3"],kd) + 4*mW**2*einsum("pt,sr",kd,kd) - 2*mZ**2*einsum("pt,sr",kd,kd) + 2*mW**2*vT**2*C["phiD"]*einsum("pt,sr",kd,kd) + mZ**2*vT**2*C["phiD"]*einsum("pt,sr",kd,kd) + 4*g1bar*mW*vT**3*C["phiWB"]*einsum("pt,sr",kd,kd) + 2*mZ**2*vT**2*einsum("pt,sr",C["phil1"],kd) + 2*mZ**2*vT**2*einsum("pt,sr",C["phil3"],kd) + 2*mZ**2*vT**2*einsum("sr,pt",C["phil1"],kd) + 2*mZ**2*vT**2*einsum("sr,pt",C["phil3"],kd) + 2*mZ**2*vT**2*einsum("st,pr",C["phil1"],kd) + 2*mZ**2*vT**2*einsum("st,pr",C["phil3"],kd)))/(mZ**4*vT**2))/8.)
    c['VeuLL'] = ((6*C["lq1"] - 6*C["lq3"] + (((8*mW**4 - mZ**4)*vT**2*C["phiD"] + 2*(8*mW**4 - 6*mW**2*mZ**2 + mZ**4 + g1bar*mW*(8*mW**2 - 3*mZ**2)*vT**3*C["phiWB"]))*kdkd)/(mZ**4*vT**2) - 2*(einsum("pr,st",C["phil1"],kd) + einsum("pr,st",C["phil3"],kd) - 3*einsum("st,pr",C["phiq1"],kd) + 3*einsum("st,pr",C["phiq3"],kd)) + (4*mW**2*(2*einsum("pr,st",C["phil1"],kd) + 2*einsum("pr,st",C["phil3"],kd) - 3*einsum("st,pr",C["phiq1"],kd) + 3*einsum("st,pr",C["phiq3"],kd)))/mZ**2)/6.)
    c['VnudLL'] = (C["lq1"] - C["lq3"] - (C["phiD"]*kdkd)/6. + ((2*mW**2 + mZ**2 + g1bar*mW*vT**3*C["phiWB"])*kdkd)/(3.*mZ**2*vT**2) - ((2*mW**2 + mZ**2)*einsum("pr,st",C["phil1"],kd))/(3.*mZ**2) + ((2*mW**2 + mZ**2)*einsum("pr,st",C["phil3"],kd))/(3.*mZ**2) + einsum("st,pr",C["phiq1"],kd) + einsum("st,pr",C["phiq3"],kd))
    c['VnueduLL'] = (2*C["lq3"] - (2*kdkd)/vT**2 - 2*(einsum("pr,st",C["phil3"],kd) + einsum("st,pr",C["phiq3"],kd)))
    c['VnueLL'] = (C["ll"] + einsum("stpr",C["ll"]) + (C["phiD"]/2. + (2*mW**2 - mZ**2 + g1bar*mW*vT**3*C["phiWB"])/(mZ**2*vT**2))*kdkd + (1 - (2*mW**2)/mZ**2)*einsum("pr,st",C["phil1"],kd) - einsum("pr,st",C["phil3"],kd) + (2*mW**2*einsum("pr,st",C["phil3"],kd))/mZ**2 - (2*einsum("pt,sr",kd,kd))/vT**2 - 2*einsum("pt,sr",C["phil3"],kd) - 2*einsum("sr,pt",C["phil3"],kd) + einsum("st,pr",C["phil1"],kd) + einsum("st,pr",C["phil3"],kd))
    c['VnunuLL'] = ((2*C["ll"] + 2*einsum("ptsr",C["ll"]) + 2*einsum("srpt",C["ll"]) + 2*einsum("stpr",C["ll"]) + ((-2 + vT**2*C["phiD"])*(kdkd + einsum("pt,sr",kd,kd)))/vT**2 + 2*(einsum("pr,st",C["phil1"],kd) - einsum("pr,st",C["phil3"],kd) + einsum("pt,sr",C["phil1"],kd) - einsum("pt,sr",C["phil3"],kd) + einsum("sr,pt",C["phil1"],kd) - einsum("sr,pt",C["phil3"],kd) + einsum("st,pr",C["phil1"],kd) - einsum("st,pr",C["phil3"],kd)))/8.)
    c['VnuuLL'] = (C["lq1"] + C["lq3"] - ((8*mW**2 - 2*mZ**2 + mZ**2*vT**2*C["phiD"] + 4*g1bar*mW*vT**3*C["phiWB"])*kdkd)/(6.*mZ**2*vT**2) - ((-4*mW**2 + mZ**2)*einsum("pr,st",C["phil1"],kd))/(3.*mZ**2) + ((-4*mW**2 + mZ**2)*einsum("pr,st",C["phil3"],kd))/(3.*mZ**2) + einsum("st,pr",C["phiq1"],kd) - einsum("st,pr",C["phiq3"],kd))
    c['VuuLL'] = ((18*C["qq1"] + 18*C["qq3"] + 18*einsum("stpr",C["qq1"]) + 18*einsum("stpr",C["qq3"]) + ((-4*mW**2 + mZ**2)*((8*mW**2 - 2*mZ**2 + vT**2*((4*mW**2 + mZ**2)*C["phiD"] + 8*g1bar*mW*vT*C["phiWB"]))*kdkd + 6*mZ**2*vT**2*(-einsum("pr,st",C["phiq1"],kd) + einsum("pr,st",C["phiq3"],kd) - einsum("st,pr",C["phiq1"],kd) + einsum("st,pr",C["phiq3"],kd))))/(mZ**4*vT**2))/36.)
    c['V1ddLR'] = (C["qd1"] + (((2*(-2*mW**4 + mW**2*mZ**2 + mZ**4) - (2*mW**4 + mZ**4)*vT**2*C["phiD"] + g1bar*mW*(-4*mW**2 + mZ**2)*vT**3*C["phiWB"])*kdkd)/vT**2 + 3*mZ**2*(2*(-mW**2 + mZ**2)*(einsum("pr,st",C["phiq1"],kd) + einsum("pr,st",C["phiq3"],kd)) - (2*mW**2 + mZ**2)*einsum("st,pr",C["phid"],kd)))/(9.*mZ**4))
    c['V1duLR'] = (C["qu1"] + ((2*(2*(mW - mZ)*(mW + mZ)*(2*mW**2 + mZ**2) + vT**2*((2*mW**4 + mZ**4)*C["phiD"] + g1bar*mW*(4*mW**2 - mZ**2)*vT*C["phiWB"]))*kdkd)/vT**2 + 3*mZ**2*(4*(mW - mZ)*(mW + mZ)*(einsum("pr,st",C["phiq1"],kd) + einsum("pr,st",C["phiq3"],kd)) - (2*mW**2 + mZ**2)*einsum("st,pr",C["phiu"],kd)))/(9.*mZ**4))
    c['V1udduLR'] = (-einsum("st,pr",C["phiudDag"],kd))
    c['V1udLR'] = (C["qd1"] + (((2*(4*mW**4 - 5*mW**2*mZ**2 + mZ**4) + vT**2*((4*mW**4 - mZ**4)*C["phiD"] + g1bar*mW*(8*mW**2 - 5*mZ**2)*vT*C["phiWB"]))*kdkd)/vT**2 + 3*mZ**2*(2*(-mW**2 + mZ**2)*(einsum("pr,st",C["phiq1"],kd) - einsum("pr,st",C["phiq3"],kd)) + (4*mW**2 - mZ**2)*einsum("st,pr",C["phid"],kd)))/(9.*mZ**4))
    c['V1uuLR'] = (C["qu1"] + (2*(-2*(4*mW**4 - 5*mW**2*mZ**2 + mZ**4) + (-4*mW**4 + mZ**4)*vT**2*C["phiD"] + g1bar*mW*(-8*mW**2 + 5*mZ**2)*vT**3*C["phiWB"])*kdkd)/(9.*mZ**4*vT**2) + (4*(mW - mZ)*(mW + mZ)*(einsum("pr,st",C["phiq1"],kd) - einsum("pr,st",C["phiq3"],kd)) + (4*mW**2 - mZ**2)*einsum("st,pr",C["phiu"],kd))/(3.*mZ**2))
    c['V8ddLR'] = (C["qd8"])
    c['V8duLR'] = (C["qu8"])
    c['V8udduLR'] = np.zeros((3,3,3,3))
    c['V8udLR'] = (C["qd8"])
    c['V8uuLR'] = (C["qu8"])
    c['VdeLR'] = (C["qe"] + ((2*(-2*mW**4 + mW**2*mZ**2 + mZ**4) - (2*mW**4 + mZ**4)*vT**2*C["phiD"] + g1bar*mW*(-4*mW**2 + mZ**2)*vT**3*C["phiWB"])*kdkd)/(3.*mZ**4*vT**2) + (6*(-mW**2 + mZ**2)*(einsum("pr,st",C["phiq1"],kd) + einsum("pr,st",C["phiq3"],kd)) - (2*mW**2 + mZ**2)*einsum("st,pr",C["phie"],kd))/(3.*mZ**2))
    c['VedLR'] = (C["ld"] + ((-2*(2*mW**4 - 3*mW**2*mZ**2 + mZ**4) + (-2*mW**4 + mZ**4)*vT**2*C["phiD"] + g1bar*mW*(-4*mW**2 + 3*mZ**2)*vT**3*C["phiWB"])*kdkd)/(3.*mZ**4*vT**2) + (2*(-mW**2 + mZ**2)*(einsum("pr,st",C["phil1"],kd) + einsum("pr,st",C["phil3"],kd)) + 3*(-2*mW**2 + mZ**2)*einsum("st,pr",C["phid"],kd))/(3.*mZ**2))
    c['VeeLR'] = (C["le"] + ((-2*(2*mW**4 - 3*mW**2*mZ**2 + mZ**4) + (-2*mW**4 + mZ**4)*vT**2*C["phiD"] + g1bar*mW*(-4*mW**2 + 3*mZ**2)*vT**3*C["phiWB"])*kdkd)/(mZ**4*vT**2) + 2*(einsum("pr,st",C["phil1"],kd) + einsum("pr,st",C["phil3"],kd)) + einsum("st,pr",C["phie"],kd) - (2*mW**2*(einsum("pr,st",C["phil1"],kd) + einsum("pr,st",C["phil3"],kd) + einsum("st,pr",C["phie"],kd)))/mZ**2)
    c['VeuLR'] = (C["lu"] + ((2*(2*(2*mW**4 - 3*mW**2*mZ**2 + mZ**4) + vT**2*((2*mW**4 - mZ**4)*C["phiD"] + g1bar*mW*(4*mW**2 - 3*mZ**2)*vT*C["phiWB"]))*kdkd)/vT**2 + mZ**2*(4*(mW - mZ)*(mW + mZ)*(einsum("pr,st",C["phil1"],kd) + einsum("pr,st",C["phil3"],kd)) + 3*(-2*mW**2 + mZ**2)*einsum("st,pr",C["phiu"],kd)))/(3.*mZ**4))
    c['VnudLR'] = (C["ld"] + ((2*(mW - mZ)*(mW + mZ) + mZ**2*vT**2*C["phiD"] + g1bar*mW*vT**3*C["phiWB"])*kdkd)/(3.*mZ**2*vT**2) - (2*(mW - mZ)*(mW + mZ)*(einsum("pr,st",C["phil1"],kd) - einsum("pr,st",C["phil3"],kd)))/(3.*mZ**2) + einsum("st,pr",C["phid"],kd))
    c['VnueduLR'] = (-einsum("st,pr",C["phiudDag"],kd))
    c['VnueLR'] = (C["le"] + ((2*(mW - mZ)*(mW + mZ) + mZ**2*vT**2*C["phiD"] + g1bar*mW*vT**3*C["phiWB"])*kdkd)/(mZ**2*vT**2) - (2*(mW - mZ)*(mW + mZ)*(einsum("pr,st",C["phil1"],kd) - einsum("pr,st",C["phil3"],kd)))/mZ**2 + einsum("st,pr",C["phie"],kd))
    c['VnuuLR'] = (C["lu"] - (2*(2*(mW - mZ)*(mW + mZ) + mZ**2*vT**2*C["phiD"] + g1bar*mW*vT**3*C["phiWB"])*kdkd)/(3.*mZ**2*vT**2) + (4*(mW - mZ)*(mW + mZ)*(einsum("pr,st",C["phil1"],kd) - einsum("pr,st",C["phil3"],kd)))/(3.*mZ**2) + einsum("st,pr",C["phiu"],kd))
    c['VueLR'] = (C["qe"] + ((2*(4*mW**4 - 5*mW**2*mZ**2 + mZ**4) + (4*mW**4 - mZ**4)*vT**2*C["phiD"] + g1bar*mW*(8*mW**2 - 5*mZ**2)*vT**3*C["phiWB"])*kdkd)/(3.*mZ**4*vT**2) + (6*(-mW**2 + mZ**2)*(einsum("pr,st",C["phiq1"],kd) - einsum("pr,st",C["phiq3"],kd)) + (4*mW**2 - mZ**2)*einsum("st,pr",C["phie"],kd))/(3.*mZ**2))
    c['S1ddRR'] = np.zeros((3,3,3,3))
    c['S1udduRR'] = (-einsum("stpr",C["quqd1"]))
    c['S1udRR'] = (C["quqd1"])
    c['S1uuRR'] = np.zeros((3,3,3,3))
    c['S8ddRR'] = np.zeros((3,3,3,3))
    c['S8udduRR'] = (-einsum("stpr",C["quqd8"]))
    c['S8udRR'] = (C["quqd8"])
    c['S8uuRR'] = np.zeros((3,3,3,3))
    c['SedRR'] = np.zeros((3,3,3,3))
    c['SeeRR'] = np.zeros((3,3,3,3))
    c['SeuRR'] = (-C["lequ1"])
    c['SnueduRR'] = (C["lequ1"])
    c['TedRR'] = np.zeros((3,3,3,3))
    c['TeuRR'] = (-C["lequ3"])
    c['TnueduRR'] = (C["lequ3"])
    c['SedRL'] = (C["ledq"])
    c['SeuRL'] = np.zeros((3,3,3,3))
    c['SnueduRL'] = (C["ledq"])
    c['V1udRR'] = (C["ud1"] + (2*(mW - mZ)*(mW + mZ)*(2*(2*(mW - mZ)*(mW + mZ) + vT**2*((mW**2 + mZ**2)*C["phiD"] + 2*g1bar*mW*vT*C["phiWB"]))*kdkd - 3*mZ**2*vT**2*(einsum("pr,st",C["phiu"],kd) - 2*einsum("st,pr",C["phid"],kd))))/(9.*mZ**4*vT**2))
    c['V8udRR'] = (C["ud8"])
    c['VddRR'] = ((9*C["dd"] + 9*einsum("stpr",C["dd"]) - (2*(mW - mZ)*(mW + mZ)*((2*(mW - mZ)*(mW + mZ) + vT**2*((mW**2 + mZ**2)*C["phiD"] + 2*g1bar*mW*vT*C["phiWB"]))*kdkd + 3*mZ**2*vT**2*(einsum("pr,st",C["phid"],kd) + einsum("st,pr",C["phid"],kd))))/(mZ**4*vT**2))/18.)
    c['VedRR'] = (C["ed"] - (2*(mW - mZ)*(mW + mZ)*((2*(mW - mZ)*(mW + mZ) + vT**2*((mW**2 + mZ**2)*C["phiD"] + 2*g1bar*mW*vT*C["phiWB"]))*kdkd + mZ**2*vT**2*(einsum("pr,st",C["phie"],kd) + 3*einsum("st,pr",C["phid"],kd))))/(3.*mZ**4*vT**2))
    c['VeeRR'] = ((C["ee"] + einsum("ptsr",C["ee"]) + einsum("srpt",C["ee"]) + einsum("stpr",C["ee"]) - (2*(mW - mZ)*(mW + mZ)*((2*(mW - mZ)*(mW + mZ) + vT**2*((mW**2 + mZ**2)*C["phiD"] + 2*g1bar*mW*vT*C["phiWB"]))*kdkd + mZ**2*vT**2*einsum("pr,st",C["phie"],kd) + 2*mW**2*einsum("pt,sr",kd,kd) - 2*mZ**2*einsum("pt,sr",kd,kd) + mW**2*vT**2*C["phiD"]*einsum("pt,sr",kd,kd) + mZ**2*vT**2*C["phiD"]*einsum("pt,sr",kd,kd) + 2*g1bar*mW*vT**3*C["phiWB"]*einsum("pt,sr",kd,kd) + mZ**2*vT**2*einsum("pt,sr",C["phie"],kd) + mZ**2*vT**2*einsum("sr,pt",C["phie"],kd) + mZ**2*vT**2*einsum("st,pr",C["phie"],kd)))/(mZ**4*vT**2))/4.)
    c['VeuRR'] = (C["eu"] + (2*(mW - mZ)*(mW + mZ)*(2*(2*(mW - mZ)*(mW + mZ) + vT**2*((mW**2 + mZ**2)*C["phiD"] + 2*g1bar*mW*vT*C["phiWB"]))*kdkd + mZ**2*vT**2*(2*einsum("pr,st",C["phie"],kd) - 3*einsum("st,pr",C["phiu"],kd))))/(3.*mZ**4*vT**2))
    c['VuuRR'] = ((9*C["uu"] + 9*einsum("stpr",C["uu"]) - (4*(mW - mZ)*(mW + mZ)*(2*(2*(mW - mZ)*(mW + mZ) + vT**2*((mW**2 + mZ**2)*C["phiD"] + 2*g1bar*mW*vT*C["phiWB"]))*kdkd - 3*mZ**2*vT**2*(einsum("pr,st",C["phiu"],kd) + einsum("st,pr",C["phiu"],kd))))/(mZ**4*vT**2))/18.)
    return c


def match_all_array(C_SMEFT, p):
    # generate a dictionary with 0 Wilson coefficients = Standard Model
    C_SMEFT_0 = {k: 0*v for k, v in C_SMEFT.items()}
    # compute the SMEFT matching contribution but subtract the SM part
    match_C = _match_all_array(C_SMEFT, p)
    match_C0 = _match_all_array(C_SMEFT_0, p)
    return {k: match_C[k] - match_C0[k] for k in match_C}
