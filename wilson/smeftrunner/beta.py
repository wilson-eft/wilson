"""SMEFT beta functions"""

import numpy as np
from collections import OrderedDict
from functools import reduce
import operator
from wilson.util.smeftutil import C_keys, C_keys_shape
from .definitions import I3
from functools import lru_cache


class HashableArray(np.ndarray):
    def __new__(cls, data, dtype=None):
        return np.array(data, dtype).view(cls)

    def __hash__(self):
        return hash(self.data.tobytes())
        # return int(sha1(self).hexdigest(), 16)

    def __eq__(self, other):
        return np.all(np.ndarray.__eq__(self, other))

    def __setitem__(self, key, value):
        raise Exception('HashableArray is read-only')


def my_einsum(indices, *args):
    hashargs = [HashableArray(arg) for arg in args]
    return _cached_einsum(indices, *hashargs)


@lru_cache(2048)
def _cached_einsum(indices, *args):
    return np.einsum(indices, *args)


def beta(C, HIGHSCALE, newphys=True):
    """Return the beta functions of all SM parameters and SMEFT Wilson
    coefficients."""

    g = C["g"]
    gp = C["gp"]
    gs = C["gs"]
    m2 = C["m2"]
    Lambda = C["Lambda"]
    Gu = C["Gu"]
    Gd = C["Gd"]
    Ge = C["Ge"]


    Eta1 = (3*np.trace(C["uphi"] @ Gu.conj().T) \
      + 3*np.trace(C["dphi"] @ Gd.conj().T) \
      + np.trace(C["ephi"] @ Ge.conj().T) \
      + 3*np.conj(np.trace(C["uphi"] @ Gu.conj().T)) \
      + 3*np.conj(np.trace(C["dphi"] @ Gd.conj().T)) \
      + np.conj(np.trace(C["ephi"] @ Ge.conj().T)))/2
    Eta2 = -6*np.trace(C["phiq3"] @ Gu @ Gu.conj().T) \
      - 6*np.trace(C["phiq3"] @ Gd @ Gd.conj().T) \
      - 2*np.trace(C["phil3"] @ Ge @ Ge.conj().T) \
      + 3*(np.trace(C["phiud"] @ Gd.conj().T @ Gu) \
      + np.conj(np.trace(C["phiud"] @ Gd.conj().T @ Gu)))
    Eta3 = 3*np.trace(C["phiq1"] @ Gd @ Gd.conj().T) \
      - 3*np.trace(C["phiq1"] @ Gu @ Gu.conj().T) \
      + 9*np.trace(C["phiq3"] @ Gd @ Gd.conj().T) \
      + 9*np.trace(C["phiq3"] @ Gu @ Gu.conj().T) \
      + 3*np.trace(C["phiu"] @ Gu.conj().T @ Gu) \
      - 3*np.trace(C["phid"] @ Gd.conj().T @ Gd) \
      - 3*(np.trace(C["phiud"] @ Gd.conj().T @ Gu) \
      + np.conj(np.trace(C["phiud"] @ Gd.conj().T @ Gu))) \
      + np.trace(C["phil1"] @ Ge @ Ge.conj().T) \
      + 3*np.trace(C["phil3"] @ Ge @ Ge.conj().T) \
      - np.trace(C["phie"] @ Ge.conj().T @ Ge)
    Eta4 = 12*np.trace(C["phiq1"] @ Gd @ Gd.conj().T) \
      - 12*np.trace(C["phiq1"] @ Gu @ Gu.conj().T) \
      + 12*np.trace(C["phiu"] @ Gu.conj().T @ Gu) \
      - 12*np.trace(C["phid"] @ Gd.conj().T @ Gd) \
      + 6*(np.trace(C["phiud"] @ Gd.conj().T @ Gu) \
      + np.conj(np.trace(C["phiud"] @ Gd.conj().T @ Gu))) \
      + 4*np.trace(C["phil1"] @ Ge @ Ge.conj().T) \
      - 4*np.trace(C["phie"] @ Ge.conj().T @ Ge)
    Eta5 = 1j*3/2*(np.trace(Gd @ C["dphi"].conj().T) \
      - np.conj(np.trace(Gd @ C["dphi"].conj().T))) \
      - 1j*3/2*(np.trace(Gu @ C["uphi"].conj().T) \
      - np.conj(np.trace(Gu @ C["uphi"].conj().T))) \
      + 1j*1/2*(np.trace(Ge @ C["ephi"].conj().T) \
      - np.conj(np.trace(Ge @ C["ephi"].conj().T)))

    GammaH = np.trace(3*Gu @ Gu.conj().T + 3*Gd @ Gd.conj().T + Ge @ Ge.conj().T)
    Gammaq = 1/2*(Gu @ Gu.conj().T + Gd @ Gd.conj().T)
    Gammau = Gu.conj().T @ Gu
    Gammad = Gd.conj().T @ Gd
    Gammal = 1/2*Ge @ Ge.conj().T
    Gammae = Ge.conj().T @ Ge

    Beta = OrderedDict()

    Beta["g"] = -19/6*g**3 - 8*g*m2/HIGHSCALE**2*C["phiW"]

    Beta["gp"] = 41/6*gp**3  - 8*gp*m2/HIGHSCALE**2*C["phiB"]

    Beta["gs"] = -7*gs**3  - 8*gs*m2/HIGHSCALE**2*C["phiG"]

    Beta["Lambda"] = 12*Lambda**2 \
      + 3/4*gp**4 + 3/2*g**2*gp**2 + 9/4*g**4 - 3*(gp**2 + 3*g**2)*Lambda \
      + 4*Lambda*GammaH \
      - 4*(3*np.trace(Gd @ Gd.conj().T @ Gd @ Gd.conj().T) \
      + 3*np.trace(Gu @ Gu.conj().T @ Gu @ Gu.conj().T) \
      + np.trace(Ge @ Ge.conj().T @ Ge @ Ge.conj().T)) \
      + 4*m2/HIGHSCALE**2*(12*C["phi"] \
      + (-16*Lambda + 10/3*g**2)*C["phiBox"] \
      + (6*Lambda + 3/2*(gp**2 - g**2))*C["phiD"] \
      + 2*(Eta1 + Eta2) \
      + 9*g**2*C["phiW"] \
      + 3*gp**2*C["phiB"] \
      + 3*g*gp*C["phiWB"] \
      + 4/3*g**2*(np.trace(C["phil3"]) \
      + 3*np.trace(C["phiq3"])))

    Beta["m2"] = m2*(6*Lambda - 9/2*g**2 - 3/2*gp**2 \
      + 2*GammaH + 4*m2/HIGHSCALE**2*(C["phiD"] \
      - 2*C["phiBox"]))

    Beta["Gu"] = 3/2*(Gu @ Gu.conj().T @ Gu - Gd @ Gd.conj().T @ Gu) \
      + (GammaH - 9/4*g**2 - 17/12*gp**2 - 8*gs**2)*Gu \
      + 2*m2/HIGHSCALE**2*(3*C["uphi"] \
      + 1/2*(C["phiD"] - 2*C["phiBox"])*Gu \
      - C["phiq1"].conj().T @ Gu \
      + 3*C["phiq3"].conj().T @ Gu \
      + Gu @ C["phiu"].conj().T \
      - Gd @ C["phiud"].conj().T \
      - 2*(my_einsum("rpts,pt", C["qu1"], Gu) \
      + 4/3*my_einsum("rpts,pt", C["qu8"], Gu)) \
      - my_einsum("ptrs,pt", C["lequ1"], np.conj(Ge)) \
      + 3*my_einsum("rspt,pt", C["quqd1"], np.conj(Gd)) \
      + 1/2*(my_einsum("psrt,pt", C["quqd1"], np.conj(Gd)) \
      + 4/3*my_einsum("psrt,pt", C["quqd8"], np.conj(Gd))))

    Beta["Gd"] = 3/2*(Gd @ Gd.conj().T @ Gd - Gu @ Gu.conj().T @ Gd) \
      + (GammaH - 9/4*g**2 - 5/12*gp**2 - 8*gs**2)*Gd \
      + 2*m2/HIGHSCALE**2*(3*C["dphi"] + 1/2*(C["phiD"] \
      - 2*C["phiBox"])*Gd \
      + C["phiq1"].conj().T @ Gd \
      + 3*C["phiq3"].conj().T @ Gd \
      - Gd @ C["phid"].conj().T \
      - Gu @ C["phiud"] \
      - 2*(my_einsum("rpts,pt", C["qd1"], Gd) \
      + 4/3*my_einsum("rpts,pt", C["qd8"], Gd)) \
      + my_einsum("ptsr,pt", np.conj(C["ledq"]), Ge) \
      + 3*my_einsum("ptrs,pt", C["quqd1"], np.conj(Gu)) \
      + 1/2*(my_einsum("rpts,tp", C["quqd1"], np.conj(Gu)) \
      + 4/3*my_einsum("rpts,tp", C["quqd8"], np.conj(Gu))))

    Beta["Ge"] = 3/2*Ge @ Ge.conj().T @ Ge + (GammaH \
      - 3/4*(3*g**2 + 5*gp**2))*Ge + 2*m2/HIGHSCALE**2*(3*C["ephi"] \
      + 1/2*(C["phiD"] - 2*C["phiBox"])*Ge \
      + C["phil1"].conj().T @ Ge \
      + 3*C["phil3"].conj().T @ Ge \
      - Ge @ C["phie"].conj().T \
      - 2*my_einsum("rpts,pt", C["le"], Ge) \
      + 3*my_einsum("rspt,tp", C["ledq"], Gd) \
      - 3*my_einsum("rspt,pt", C["lequ1"], np.conj(Gu)))

    Beta["Theta"] = -128*np.pi**2/g**2*m2/HIGHSCALE**2*C["phiWtilde"]

    Beta["Thetap"] = -128*np.pi**2/gp**2*m2/HIGHSCALE**2*C["phiBtilde"]

    Beta["Thetas"] = -128*np.pi**2/gs**2*m2/HIGHSCALE**2*C["phiGtilde"]

    if not newphys:
        # if there is no new physics, generate a dictionary with zero
        # Wilson coefficients (i.e. zero beta functions)
        BetaSM = C_array2dict(np.zeros(5000))
        BetaSM.update(Beta)
        return BetaSM


    XiB = 2/3*(C["phiBox"] + C["phiD"]) \
      + 8/3*( - np.trace(C["phil1"]) + np.trace(C["phiq1"]) \
      - np.trace(C["phie"]) \
      + 2*np.trace(C["phiu"]) - np.trace(C["phid"]))
    Xie = 2*my_einsum("prst,rs", C["le"], Ge) \
      - 3*my_einsum("ptsr,rs", C["ledq"], Gd) \
      + 3*my_einsum("ptsr,sr", C["lequ1"], np.conj(Gu))
    Xid = 2*(my_einsum("prst,rs", C["qd1"], Gd) \
      + 4/3*my_einsum("prst,rs", C["qd8"], Gd)) \
      - (3*my_einsum("srpt,sr", C["quqd1"], np.conj(Gu)) \
      + 1/2*(my_einsum("prst,sr", C["quqd1"], np.conj(Gu)) \
      + 4/3*my_einsum("prst,sr", C["quqd8"], np.conj(Gu)))) \
      - my_einsum("srtp,sr", np.conj(C["ledq"]), Ge)
    Xiu = 2*(my_einsum("prst,rs", C["qu1"], Gu) \
      + 4/3*my_einsum("prst,rs", C["qu8"], Gu)) \
      - (3*my_einsum("ptsr,sr", C["quqd1"], np.conj(Gd)) \
      + 1/2*(my_einsum("stpr,sr", C["quqd1"], np.conj(Gd)) \
      + 4/3*my_einsum("stpr,sr", C["quqd8"], np.conj(Gd)))) \
      + my_einsum("srpt,sr", C["lequ1"], np.conj(Ge))

    Beta["G"] = 15*gs**2*C["G"]

    Beta["Gtilde"] = 15*gs**2*C["Gtilde"]

    Beta["W"] = 29/2*g**2*C["W"]

    Beta["Wtilde"] = 29/2*g**2*C["Wtilde"]

    #c.c.
    Beta["phi"] = -9/2*(3*g**2 \
      + gp**2)*C["phi"] \
      + Lambda*(20/3*g**2*C["phiBox"] \
      + 3*(gp**2 \
      - g**2)*C["phiD"]) \
      - 3/4*(g**2 \
      + gp**2)**2*C["phiD"] \
      + 6*Lambda*(3*g**2*C["phiW"] \
      + gp**2*C["phiB"] \
      + g*gp*C["phiWB"]) \
      - 3*(g**2*gp**2 \
      + 3*g**4)*C["phiW"] \
      - 3*(gp**4 \
      + g**2*gp**2)*C["phiB"] \
      - 3*(g*gp**3 \
      + g**3*gp)*C["phiWB"] \
      + 8/3*Lambda*g**2*(np.trace(C["phil3"]) \
      + 3*np.trace(C["phiq3"])) \
      + 54*Lambda*C["phi"] \
      - 40*Lambda**2*C["phiBox"] \
      + 12*Lambda**2*C["phiD"] \
      + 4*Lambda*(Eta1 \
      + Eta2) \
      - 4*(3*np.trace(C["uphi"] @ Gu.conj().T @ Gu @ Gu.conj().T) \
      + 3*np.trace(C["dphi"] @ Gd.conj().T @ Gd @ Gd.conj().T) \
      + np.trace(C["ephi"] @ Ge.conj().T @ Ge @ Ge.conj().T) \
      + 3*np.conj(np.trace(C["uphi"] @ Gu.conj().T @ Gu @ Gu.conj().T)) \
      + 3*np.conj(np.trace(C["dphi"] @ Gd.conj().T @ Gd @ Gd.conj().T)) \
      + np.conj(np.trace(C["ephi"] @ Ge.conj().T @ Ge @ Ge.conj().T))) \
      + 6*GammaH*C["phi"]

    Beta["phiBox"] = -(4*g**2 \
      + 4/3*gp**2)*C["phiBox"] \
      + 5/3*gp**2*C["phiD"] \
      + 2*g**2*(np.trace(C["phil3"]) \
      + 3*np.trace(C["phiq3"])) \
      + 2/3*gp**2*(2*np.trace(C["phiu"]) \
      - np.trace(C["phid"]) \
      - np.trace(C["phie"]) \
      + np.trace(C["phiq1"]) \
      - np.trace(C["phil1"])) \
      + 12*Lambda*C["phiBox"] \
      - 2*Eta3 \
      + 4*GammaH*C["phiBox"]

    Beta["phiD"] = 20/3*gp**2*C["phiBox"] \
      + (9/2*g**2 \
      - 5/6*gp**2)*C["phiD"] \
      + 8/3*gp**2*(2*np.trace(C["phiu"]) \
      - np.trace(C["phid"]) \
      - np.trace(C["phie"]) \
      + np.trace(C["phiq1"]) \
      - np.trace(C["phil1"])) \
      + 6*Lambda*C["phiD"] \
      - 2*Eta4 \
      + 4*GammaH*C["phiD"]

    #c.c.
    Beta["phiG"] = (-3/2*gp**2 \
      - 9/2*g**2 \
      - 14*gs**2)*C["phiG"] \
      + 6*Lambda*C["phiG"] \
      - 2*gs*(np.trace(C["uG"] @ Gu.conj().T) \
      + np.trace(C["dG"] @ Gd.conj().T) \
      + np.conj(np.trace(C["uG"] @ Gu.conj().T)) \
      + np.conj(np.trace(C["dG"] @ Gd.conj().T))) \
      + 2*GammaH*C["phiG"]

    #c.c.
    Beta["phiB"] = (85/6*gp**2 \
      - 9/2*g**2)*C["phiB"] \
      + 3*g*gp*C["phiWB"] \
      + 6*Lambda*C["phiB"] \
      + gp*( \
      - 5*np.trace(C["uB"] @ Gu.conj().T) \
      + np.trace(C["dB"] @ Gd.conj().T) \
      + 3*np.trace(C["eB"] @ Ge.conj().T) \
      - 5*np.conj(np.trace(C["uB"] @ Gu.conj().T)) \
      + np.conj(np.trace(C["dB"] @ Gd.conj().T)) \
      + 3*np.conj(np.trace(C["eB"] @ Ge.conj().T))) \
      + 2*GammaH*C["phiB"]

    #c.c.
    Beta["phiW"] = (-3/2*gp**2 \
      - 53/6*g**2)*C["phiW"] \
      + g*gp*C["phiWB"] \
      - 15*g**3*C["W"] \
      + 6*Lambda*C["phiW"] \
      - g*(3*np.trace(C["uW"] @ Gu.conj().T) \
      + 3*np.trace(C["dW"] @ Gd.conj().T) \
      + np.trace(C["eW"] @ Ge.conj().T) \
      + 3*np.conj(np.trace(C["uW"] @ Gu.conj().T)) \
      + 3*np.conj(np.trace(C["dW"] @ Gd.conj().T)) \
      + np.conj(np.trace(C["eW"] @ Ge.conj().T))) \
      + 2*GammaH*C["phiW"]

    #c.c.
    Beta["phiWB"] = (19/3*gp**2 \
      + 4/3*g**2)*C["phiWB"] \
      + 2*g*gp*(C["phiB"] \
      + C["phiW"]) \
      + 3*g**2*gp*C["W"] \
      + 2*Lambda*C["phiWB"] \
      + g*(3*np.trace(C["uB"] @ Gu.conj().T) \
      - 3*np.trace(C["dB"] @ Gd.conj().T) \
      - np.trace(C["eB"] @ Ge.conj().T) \
      + 3*np.conj(np.trace(C["uB"] @ Gu.conj().T)) \
      - 3*np.conj(np.trace(C["dB"] @ Gd.conj().T)) \
      - np.conj(np.trace(C["eB"] @ Ge.conj().T))) \
      + gp*(5*np.trace(C["uW"] @ Gu.conj().T) \
      + np.trace(C["dW"] @ Gd.conj().T) \
      + 3*np.trace(C["eW"] @ Ge.conj().T) \
      + 5*np.conj(np.trace(C["uW"] @ Gu.conj().T)) \
      + np.conj(np.trace(C["dW"] @ Gd.conj().T)) \
      + 3*np.conj(np.trace(C["eW"] @ Ge.conj().T))) \
      + 2*GammaH*C["phiWB"]

    #problem with i as I*iCPV
    Beta["phiGtilde"] = (-3/2*gp**2 \
      - 9/2*g**2 \
      - 14*gs**2)*C["phiGtilde"] \
      + 6*Lambda*C["phiGtilde"] \
      + 2j*gs*(np.trace(C["uG"] @ Gu.conj().T) \
      + np.trace(C["dG"] @ Gd.conj().T) \
      - np.conj(np.trace(C["uG"] @ Gu.conj().T)) \
      - np.conj(np.trace(C["dG"] @ Gd.conj().T))) \
      + 2*GammaH*C["phiGtilde"]

    #i
    Beta["phiBtilde"] = (85/6*gp**2 \
      - 9/2*g**2)*C["phiBtilde"] \
      + 3*g*gp*C["phiWtildeB"] \
      + 6*Lambda*C["phiBtilde"] \
      - 1j*gp*( \
      - 5*np.trace(C["uB"] @ Gu.conj().T) \
      + np.trace(C["dB"] @ Gd.conj().T) \
      + 3*np.trace(C["eB"] @ Ge.conj().T) \
      + 5*np.conj(np.trace(C["uB"] @ Gu.conj().T)) \
      - np.conj(np.trace(C["dB"] @ Gd.conj().T)) \
      - 3*np.conj(np.trace(C["eB"] @ Ge.conj().T))) \
      + 2*GammaH*C["phiBtilde"]

    #i
    Beta["phiWtilde"] = (-3/2*gp**2 \
      - 53/6*g**2)*C["phiWtilde"] \
      + g*gp*C["phiWtildeB"] \
      - 15*g**3*C["Wtilde"] \
      + 6*Lambda*C["phiWtilde"] \
      + 1j*g*(3*np.trace(C["uW"] @ Gu.conj().T) \
      + 3*np.trace(C["dW"] @ Gd.conj().T) \
      + np.trace(C["eW"] @ Ge.conj().T) \
      - 3*np.conj(np.trace(C["uW"] @ Gu.conj().T)) \
      - 3*np.conj(np.trace(C["dW"] @ Gd.conj().T)) \
      - np.conj(np.trace(C["eW"] @ Ge.conj().T))) \
      + 2*GammaH*C["phiWtilde"]

    #i
    Beta["phiWtildeB"] = (19/3*gp**2 \
      + 4/3*g**2)*C["phiWtildeB"] \
      + 2*g*gp*(C["phiBtilde"] \
      + C["phiWtilde"]) \
      + 3*g**2*gp*C["Wtilde"] \
      + 2*Lambda*C["phiWtildeB"] \
      - 1j*g*(3*np.trace(C["uB"] @ Gu.conj().T) \
      - 3*np.trace(C["dB"] @ Gd.conj().T) \
      - np.trace(C["eB"] @ Ge.conj().T) \
      - 3*np.conj(np.trace(C["uB"] @ Gu.conj().T)) \
      + 3*np.conj(np.trace(C["dB"] @ Gd.conj().T)) \
      + np.conj(np.trace(C["eB"] @ Ge.conj().T))) \
      - 1j*gp*(5*np.trace(C["uW"] @ Gu.conj().T) \
      + np.trace(C["dW"] @ Gd.conj().T) \
      + 3*np.trace(C["eW"] @ Ge.conj().T) \
      - 5*np.conj(np.trace(C["uW"] @ Gu.conj().T)) \
      - np.conj(np.trace(C["dW"] @ Gd.conj().T)) \
      - 3*np.conj(np.trace(C["eW"] @ Ge.conj().T))) \
      + 2*GammaH*C["phiWtildeB"]

    """(3,3)"""
    #i  #the coefficients of Eta5 is not equal
    Beta["uphi"] = (10/3*g**2*C["phiBox"] \
      + 3/2*(gp**2 \
      - g**2)*C["phiD"] \
      + 32*gs**2*(C["phiG"] \
      + 1j*C["phiGtilde"]) \
      + 9*g**2*(C["phiW"] \
      + 1j*C["phiWtilde"]) \
      + 17/3*gp**2*(C["phiB"] \
      + 1j*C["phiBtilde"]) \
      - g*gp*(C["phiWB"] \
      + 1j*C["phiWtildeB"]) \
      + 4/3*g**2*(np.trace(C["phil3"]) \
      + 3*np.trace(C["phiq3"])))*Gu \
      - (35/12*gp**2 \
      + 27/4*g**2 \
      + 8*gs**2)*C["uphi"] \
      - gp*(5*gp**2 \
      - 3*g**2)*C["uB"] \
      + g*(5*gp**2 \
      - 9*g**2)*C["uW"] \
      - (3*g**2 \
      - gp**2)*Gu @ C["phiu"] \
      + 3*g**2*Gd @ C["phiud"].conj().T \
      + 4*gp**2*C["phiq1"] @ Gu \
      - 4*gp**2*C["phiq3"] @ Gu \
      - 5*gp*(C["uB"] @ Gu.conj().T @ Gu \
      + Gu @ Gu.conj().T @ C["uB"]) \
      - 3*g*(C["uW"] @ Gu.conj().T @ Gu \
      - Gu @ Gu.conj().T @ C["uW"]) \
      - 16*gs*(C["uG"] @ Gu.conj().T @ Gu \
      + Gu @ Gu.conj().T @ C["uG"]) \
      - 12*g*Gd @ Gd.conj().T @ C["uW"] \
      - 6*g*C["dW"] @ Gd.conj().T @ Gu \
      + Lambda*(12*C["uphi"] \
      - 2*C["phiq1"] @ Gu \
      + 6*C["phiq3"] @ Gu \
      + 2*Gu @ C["phiu"] \
      - 2*Gd @ C["phiud"].conj().T \
      - 2*C["phiBox"]*Gu \
      + C["phiD"]*Gu \
      - 4*my_einsum("rpts,pt", C["qu1"], Gu) \
      - 16/3*my_einsum("rpts,pt", C["qu8"], Gu) \
      - 2*my_einsum("ptrs,pt", C["lequ1"], np.conj(Ge)) \
      + 6*my_einsum("rspt,pt", C["quqd1"], np.conj(Gd)) \
      + my_einsum("psrt,pt", C["quqd1"], np.conj(Gd)) \
      + 4/3*my_einsum("psrt,pt", C["quqd8"], np.conj(Gd))) \
      + 2*(Eta1 \
      + Eta2 \
      - 1j*Eta5)*Gu \
      + (C["phiD"] \
      - 6*C["phiBox"])*Gu @ Gu.conj().T @ Gu \
      - 2*C["phiq1"] @ Gu @ Gu.conj().T @ Gu \
      + 6*C["phiq3"] @ Gd @ Gd.conj().T @ Gu \
      + 2*Gu @ Gu.conj().T @ Gu @ C["phiu"] \
      - 2*Gd @ Gd.conj().T @ Gd @ C["phiud"].conj().T \
      + 8*(my_einsum("rpts,pt", C["qu1"], Gu @ Gu.conj().T @ Gu) \
      + 4/3*my_einsum("rpts,pt", C["qu8"], Gu @ Gu.conj().T @ Gu)) \
      - 2*(my_einsum("tsrp,pt", C["quqd1"], Gd.conj().T @ Gd @ Gd.conj().T) \
      + 4/3*my_einsum("tsrp,pt", C["quqd8"], Gd.conj().T @ Gd @ Gd.conj().T)) \
      - 12*my_einsum("rstp,pt", C["quqd1"], Gd.conj().T @ Gd @ Gd.conj().T) \
      + 4*my_einsum("tprs,pt", C["lequ1"], Ge.conj().T @ Ge @ Ge.conj().T) \
      + 4*C["uphi"] @ Gu.conj().T @ Gu \
      + 5*Gu @ Gu.conj().T @ C["uphi"] \
      - 2*Gd @ C["dphi"].conj().T @ Gu \
      - C["dphi"] @ Gd.conj().T @ Gu \
      - 2*Gd @ Gd.conj().T @ C["uphi"] \
      + 3*GammaH*C["uphi"] \
      + Gammaq @ C["uphi"] \
      + C["uphi"] @ Gammau

    #i  #Eta5
    Beta["dphi"] = (10/3*g**2*C["phiBox"] \
      + 3/2*(gp**2 \
      - g**2)*C["phiD"] \
      + 32*gs**2*(C["phiG"] \
      + 1j*C["phiGtilde"]) \
      + 9*g**2*(C["phiW"] \
      + 1j*C["phiWtilde"]) \
      + 5/3*gp**2*(C["phiB"] \
      + 1j*C["phiBtilde"]) \
      + g*gp*(C["phiWB"] \
      + 1j*C["phiWtildeB"]) \
      + 4/3*g**2*(np.trace(C["phil3"]) \
      + 3*np.trace(C["phiq3"])))*Gd \
      - (23/12*gp**2 \
      + 27/4*g**2 \
      + 8*gs**2)*C["dphi"] \
      - gp*(3*g**2 \
      - gp**2)*C["dB"] \
      - g*(9*g**2 \
      - gp**2)*C["dW"] \
      + (3*g**2 \
      + gp**2)*Gd @ C["phid"] \
      + 3*g**2*Gu @ C["phiud"] \
      - 2*gp**2*C["phiq1"] @ Gd \
      - 2*gp**2*C["phiq3"] @ Gd \
      + gp*(C["dB"] @ Gd.conj().T @ Gd \
      + Gd @ Gd.conj().T @ C["dB"]) \
      - 3*g*(C["dW"] @ Gd.conj().T @ Gd \
      - Gd @ Gd.conj().T @ C["dW"]) \
      - 16*gs*(C["dG"] @ Gd.conj().T @ Gd \
      + Gd @ Gd.conj().T @ C["dG"]) \
      - 12*g*Gu @ Gu.conj().T @ C["dW"] \
      - 6*g*C["uW"] @ Gu.conj().T @ Gd \
      + Lambda*(12*C["dphi"] \
      + 2*C["phiq1"] @ Gd \
      + 6*C["phiq3"] @ Gd \
      - 2*Gd @ C["phid"] \
      - 2*Gu @ C["phiud"] \
      - 2*C["phiBox"]*Gd \
      + C["phiD"]*Gd \
      - 4*my_einsum("rpts,pt", C["qd1"], Gd) \
      - 16/3*my_einsum("rpts,pt", C["qd8"], Gd) \
      + 2*my_einsum("ptsr,pt", np.conj(C["ledq"]), Ge) \
      + 6*my_einsum("ptrs,pt", C["quqd1"], np.conj(Gu)) \
      + my_einsum("rtps,pt", C["quqd1"], np.conj(Gu)) \
      + 4/3*my_einsum("rtps,pt", C["quqd8"], np.conj(Gu))) \
      + 2*(Eta1 \
      + Eta2 \
      + 1j*Eta5)*Gd \
      + (C["phiD"] \
      - 6*C["phiBox"])*Gd @ Gd.conj().T @ Gd \
      + 2*C["phiq1"] @ Gd @ Gd.conj().T @ Gd \
      + 6*C["phiq3"] @ Gu @ Gu.conj().T @ Gd \
      - 2*Gd @ Gd.conj().T @ Gd @ C["phid"] \
      - 2*Gu @ Gu.conj().T @ Gu @ C["phiud"] \
      + 8*(my_einsum("rpts,pt", C["qd1"], Gd @ Gd.conj().T @ Gd) \
      + 4/3*my_einsum("rpts,pt", C["qd8"], Gd @ Gd.conj().T @ Gd)) \
      - 2*(my_einsum("rpts,pt", C["quqd1"], Gu.conj().T @ Gu @ Gu.conj().T) \
      + 4/3*my_einsum("rpts,pt", C["quqd8"], Gu.conj().T @ Gu @ Gu.conj().T)) \
      - 12*my_einsum("tprs,pt", C["quqd1"], Gu @ Gu.conj().T @ Gu) \
      - 4*my_einsum("ptsr,pt", np.conj(C["ledq"]), Ge @ Ge.conj().T @ Ge) \
      + 4*C["dphi"] @ Gd.conj().T @ Gd \
      + 5*Gd @ Gd.conj().T @ C["dphi"] \
      - 2*Gu @ C["uphi"].conj().T @ Gd \
      - C["uphi"] @ Gu.conj().T @ Gd \
      - 2*Gu @ Gu.conj().T @ C["dphi"] \
      + 3*GammaH*C["dphi"] \
      + Gammaq @ C["dphi"] \
      + C["dphi"] @ Gammad

    #i
    Beta["ephi"] = (10/3*g**2*C["phiBox"] \
      + 3/2*(gp**2 \
      - g**2)*C["phiD"] \
      + 9*g**2*(C["phiW"] \
      + 1j*C["phiWtilde"]) \
      + 15*gp**2*(C["phiB"] \
      + 1j*C["phiBtilde"]) \
      - 3*g*gp*(C["phiWB"] \
      + 1j*C["phiWtildeB"]) \
      + 4/3*g**2*(np.trace(C["phil3"]) \
      + 3*np.trace(C["phiq3"])))*Ge \
      - 3/4*(7*gp**2 \
      + 9*g**2)*C["ephi"] \
      - 3*gp*(g**2 \
      - 3*gp**2)*C["eB"] \
      - 9*g*(g**2 \
      - gp**2)*C["eW"] \
      + 3*(g**2 \
      - gp**2)*Ge @ C["phie"] \
      - 6*gp**2*C["phil1"] @ Ge \
      - 6*gp**2*C["phil3"] @ Ge \
      + 9*gp*(C["eB"] @ Ge.conj().T @ Ge \
      + Ge @ Ge.conj().T @ C["eB"]) \
      - 3*g*(C["eW"] @ Ge.conj().T @ Ge \
      - Ge @ Ge.conj().T @ C["eW"]) \
      + Lambda*(12*C["ephi"] \
      + 2*C["phil1"] @ Ge \
      + 6*C["phil3"] @ Ge \
      - 2*Ge @ C["phie"] \
      - 2*C["phiBox"]*Ge \
      + C["phiD"]*Ge \
      - 4*my_einsum("rpts,pt", C["le"], Ge) \
      + 6*my_einsum("rspt,tp", C["ledq"], Gd) \
      - 6*my_einsum("rspt,pt", C["lequ1"], np.conj(Gu))) \
      + 2*(Eta1 \
      + Eta2 \
      + 1j*Eta5)*Ge \
      + (C["phiD"] \
      - 6*C["phiBox"])*Ge @ Ge.conj().T @ Ge \
      + 2*C["phil1"] @ Ge @ Ge.conj().T @ Ge \
      - 2*Ge @ Ge.conj().T @ Ge @ C["phie"] \
      + 8*my_einsum("rpts,pt", C["le"], Ge @ Ge.conj().T @ Ge) \
      - 12*my_einsum("rspt,tp", C["ledq"], Gd @ Gd.conj().T @ Gd) \
      + 12*my_einsum("rstp,pt", C["lequ1"], Gu.conj().T @ Gu @ Gu.conj().T) \
      + 4*C["ephi"] @ Ge.conj().T @ Ge \
      + 5*Ge @ Ge.conj().T @ C["ephi"] \
      + 3*GammaH*C["ephi"] \
      + Gammal @ C["ephi"] \
      + C["ephi"] @ Gammae

    #i
    Beta["eW"] = 1/12*(3*gp**2 \
      - 11*g**2)*C["eW"] \
      - 1/2*g*gp*C["eB"] \
      - (g*(C["phiW"] \
      + 1j*C["phiWtilde"]) \
      - 3/2*gp*(C["phiWB"] \
      + 1j*C["phiWtildeB"]))*Ge \
      - 6*g*my_einsum("rspt,pt", C["lequ3"], np.conj(Gu)) \
      + C["eW"] @ Ge.conj().T @ Ge \
      + GammaH*C["eW"] \
      + Gammal @ C["eW"] \
      + C["eW"] @ Gammae

    #i
    Beta["eB"] = 1/4*(151/3*gp**2 \
      - 9*g**2)*C["eB"] \
      - 3/2*g*gp*C["eW"] \
      - (3/2*g*(C["phiWB"] \
      + 1j*C["phiWtildeB"]) \
      - 3*gp*(C["phiB"] \
      + 1j*C["phiBtilde"]))*Ge \
      + 10*gp*my_einsum("rspt,pt", C["lequ3"], np.conj(Gu)) \
      + C["eB"] @ Ge.conj().T @ Ge \
      + 2*Ge @ Ge.conj().T @ C["eB"] \
      + GammaH*C["eB"] \
      + Gammal @ C["eB"] \
      + C["eB"] @ Gammae

    #i
    Beta["uG"] = -1/36*(81*g**2 \
      + 19*gp**2 \
      + 204*gs**2)*C["uG"] \
      + 6*g*gs*C["uW"] \
      + 10/3*gp*gs*C["uB"] \
      - gs*(4*(C["phiG"] \
      + 1j*C["phiGtilde"]) \
      - 9*gs*(C["G"] \
      + 1j*C["Gtilde"]))*Gu \
      - gs*(my_einsum("psrt,pt", C["quqd1"], np.conj(Gd)) \
      - 1/6*my_einsum("psrt,pt", C["quqd8"], np.conj(Gd))) \
      + 2*Gu @ Gu.conj().T @ C["uG"] \
      - 2*Gd @ Gd.conj().T @ C["uG"] \
      - C["dG"] @ Gd.conj().T @ Gu \
      + C["uG"] @ Gu.conj().T @ Gu \
      + GammaH*C["uG"] \
      + Gammaq @ C["uG"] \
      + C["uG"] @ Gammau

    #i
    Beta["uW"] = -1/36*(33*g**2 \
      + 19*gp**2 \
      - 96*gs**2)*C["uW"] \
      + 8/3*g*gs*C["uG"] \
      - 1/6*g*gp*C["uB"] \
      - (g*(C["phiW"] \
      + 1j*C["phiWtilde"]) \
      - 5/6*gp*(C["phiWB"] \
      + 1j*C["phiWtildeB"]))*Gu \
      + g/4*(my_einsum("psrt,pt", C["quqd1"], np.conj(Gd)) \
      + 4/3*my_einsum("psrt,pt", C["quqd8"], np.conj(Gd))) \
      - 2*g*my_einsum("ptrs,pt", C["lequ3"], np.conj(Ge)) \
      + 2*Gd @ Gd.conj().T @ C["uW"] \
      - C["dW"] @ Gd.conj().T @ Gu \
      + C["uW"] @ Gu.conj().T @ Gu \
      + GammaH*C["uW"] \
      + Gammaq @ C["uW"] \
      + C["uW"] @ Gammau

    #i
    Beta["uB"] = -1/36*(81*g**2 \
      - 313*gp**2 \
      - 96*gs**2)*C["uB"] \
      + 40/9*gp*gs*C["uG"] \
      - 1/2*g*gp*C["uW"] \
      - (-3/2*g*(C["phiWB"] \
      + 1j*C["phiWtildeB"]) \
      + 5/3*gp*(C["phiB"] \
      + 1j*C["phiBtilde"]))*Gu \
      + gp/12*(my_einsum("psrt,pt", C["quqd1"], np.conj(Gd)) \
      + 4/3*my_einsum("psrt,pt", C["quqd8"], np.conj(Gd))) \
      - 6*gp*my_einsum("ptrs,pt", C["lequ3"], np.conj(Ge)) \
      + 2*Gu @ Gu.conj().T @ C["uB"] \
      - 2*Gd @ Gd.conj().T @ C["uB"] \
      - C["dB"] @ Gd.conj().T @ Gu \
      + C["uB"] @ Gu.conj().T @ Gu \
      + GammaH*C["uB"] \
      + Gammaq @ C["uB"] \
      + C["uB"] @ Gammau

    #i
    Beta["dG"] = -1/36*(81*g**2 \
      + 31*gp**2 \
      + 204*gs**2)*C["dG"] \
      + 6*g*gs*C["dW"] \
      - 2/3*gp*gs*C["dB"] \
      - gs*(4*(C["phiG"] \
      + 1j*C["phiGtilde"]) \
      - 9*gs*(C["G"] \
      + 1j*C["Gtilde"]))*Gd \
      - gs*(my_einsum("rtps,pt", C["quqd1"], np.conj(Gu)) \
      - 1/6*my_einsum("rtps,pt", C["quqd8"], np.conj(Gu))) \
      - 2*Gu @ Gu.conj().T @ C["dG"] \
      + 2*Gd @ Gd.conj().T @ C["dG"] \
      - C["uG"] @ Gu.conj().T @ Gd \
      + C["dG"] @ Gd.conj().T @ Gd \
      + GammaH*C["dG"] \
      + Gammaq @ C["dG"] \
      + C["dG"] @ Gammad

    #i
    Beta["dW"] = -1/36*(33*g**2 \
      + 31*gp**2 \
      - 96*gs**2)*C["dW"] \
      + 8/3*g*gs*C["dG"] \
      + 5/6*g*gp*C["dB"] \
      - (g*(C["phiW"] \
      + 1j*C["phiWtilde"]) \
      - gp/6*(C["phiWB"] \
      + 1j*C["phiWtildeB"]))*Gd \
      + g/4*(my_einsum("rtps,pt", C["quqd1"], np.conj(Gu)) \
      + 4/3*my_einsum("rtps,pt", C["quqd8"], np.conj(Gu))) \
      + 2*Gu @ Gu.conj().T @ C["dW"] \
      - C["uW"] @ Gu.conj().T @ Gd \
      + C["dW"] @ Gd.conj().T @ Gd \
      + GammaH*C["dW"] \
      + Gammaq @ C["dW"] \
      + C["dW"] @ Gammad

    #i
    Beta["dB"] = -1/36*(81*g**2 \
      - 253*gp**2 \
      - 96*gs**2)*C["dB"] \
      - 8/9*gp*gs*C["dG"] \
      + 5/2*g*gp*C["dW"] \
      - (3/2*g*(C["phiWB"] \
      + 1j*C["phiWtildeB"]) \
      - gp/3*(C["phiB"] \
      + 1j*C["phiBtilde"]))*Gd \
      - 5/12*gp*(my_einsum("rtps,pt", C["quqd1"], np.conj(Gu)) \
      + 4/3*my_einsum("rtps,pt", C["quqd8"], np.conj(Gu))) \
      - 2*Gu @ Gu.conj().T @ C["dB"] \
      + 2*Gd @ Gd.conj().T @ C["dB"] \
      - C["uB"] @ Gu.conj().T @ Gd \
      + C["dB"] @ Gd.conj().T @ Gd \
      + GammaH*C["dB"] \
      + Gammaq @ C["dB"] \
      + C["dB"] @ Gammad

    #I3 #coefficient not equal with manual!!!!!!
    Beta["phil1"] = -1/4*XiB*gp**2*I3 \
      + 1/3*gp**2*C["phil1"] \
      - 2/3*gp**2*(my_einsum("rstt", C["ld"]) \
      + my_einsum("rstt", C["le"]) \
      + 2*my_einsum("rstt", C["ll"]) \
      + my_einsum("rtts", C["ll"]) \
      - my_einsum("rstt", C["lq1"]) \
      - 2*my_einsum("rstt", C["lu"])) \
      - 1/2*(C["phiBox"] \
      + C["phiD"])*Ge @ Ge.conj().T \
      - Ge @ C["phie"] @ Ge.conj().T \
      + 3/2*(Ge @ Ge.conj().T @ C["phil1"] \
      + C["phil1"] @ Ge @ Ge.conj().T \
      + 3*Ge @ Ge.conj().T @ C["phil3"] \
      + 3*C["phil3"] @ Ge @ Ge.conj().T) \
      + 2*my_einsum("rspt,tp", C["le"], Ge.conj().T @ Ge) \
      - 2*(2*my_einsum("rspt,tp", C["ll"], Ge @ Ge.conj().T) \
      + my_einsum("rtps,tp", C["ll"], Ge @ Ge.conj().T)) \
      - 6*my_einsum("rspt,tp", C["lq1"], Gd @ Gd.conj().T) \
      + 6*my_einsum("rspt,tp", C["lq1"], Gu @ Gu.conj().T) \
      - 6*my_einsum("rspt,tp", C["lu"], Gu.conj().T @ Gu) \
      + 6*my_einsum("rspt,tp", C["ld"], Gd.conj().T @ Gd) \
      + 2*GammaH*C["phil1"] \
      + Gammal @ C["phil1"] \
      + C["phil1"] @ Gammal

    #I3 #coefficient
    Beta["phil3"] = 2/3*g**2*(1/4*C["phiBox"] \
      + np.trace(C["phil3"]) \
      + 3*np.trace(C["phiq3"]))*I3 \
      - 17/3*g**2*C["phil3"] \
      + 2/3*g**2*my_einsum("rtts", C["ll"]) \
      + 2*g**2*my_einsum("rstt", C["lq3"]) \
      - 1/2*C["phiBox"]*Ge @ Ge.conj().T \
      + 1/2*(3*Ge @ Ge.conj().T @ C["phil1"] \
      + 3*C["phil1"] @ Ge @ Ge.conj().T \
      + Ge @ Ge.conj().T @ C["phil3"] \
      + C["phil3"] @ Ge @ Ge.conj().T) \
      - 2*(my_einsum("rtps,tp", C["ll"], Ge @ Ge.conj().T)) \
      - 6*my_einsum("rspt,tp", C["lq3"], Gd @ Gd.conj().T) \
      - 6*my_einsum("rspt,tp", C["lq3"], Gu @ Gu.conj().T) \
      + 2*GammaH*C["phil3"] \
      + Gammal @ C["phil3"] \
      + C["phil3"] @ Gammal

    #I3  #coefficient even terms not equal...
    Beta["phie"] = -1/2*XiB*gp**2*I3 \
      + 1/3*gp**2*C["phie"] \
      - 2/3*gp**2*(my_einsum("rstt", C["ed"]) \
      + 4*my_einsum("rstt", C["ee"]) \
      - 2*my_einsum("rstt", C["eu"]) \
      + my_einsum("ttrs", C["le"]) \
      - my_einsum("ttrs", C["qe"])) \
      + (C["phiBox"] \
      + C["phiD"])*Ge.conj().T @ Ge \
      - 2*Ge.conj().T @ C["phil1"] @ Ge \
      + 3*(Ge.conj().T @ Ge @ C["phie"] \
      + C["phie"] @ Ge.conj().T @ Ge) \
      - 2*my_einsum("ptrs,tp", C["le"], Ge @ Ge.conj().T) \
      + 8*my_einsum("rspt,tp", C["ee"], Ge.conj().T @ Ge) \
      - 6*my_einsum("rspt,tp", C["eu"], Gu.conj().T @ Gu) \
      + 6*my_einsum("rspt,tp", C["ed"], Gd.conj().T @ Gd) \
      - 6*my_einsum("ptrs,tp", C["qe"], Gd @ Gd.conj().T) \
      + 6*my_einsum("ptrs,tp", C["qe"], Gu @ Gu.conj().T) \
      + 2*GammaH*C["phie"] \
      + Gammae @ C["phie"] \
      + C["phie"] @ Gammae

    #I3  #coefficient???
    Beta["phiq1"] = 1/12*XiB*gp**2*I3 \
      + 1/3*gp**2*C["phiq1"] \
      - 2/3*gp**2*(my_einsum("ttrs", C["lq1"]) \
      + my_einsum("rstt", C["qd1"]) \
      - 2*my_einsum("rstt", C["qu1"]) \
      + my_einsum("rstt", C["qe"]) \
      - 2*my_einsum("rstt", C["qq1"]) \
      - 1/3*my_einsum("rtts", C["qq1"]) \
      - my_einsum("rtts", C["qq3"])) \
      + 1/2*(C["phiBox"] \
      + C["phiD"])*(Gu @ Gu.conj().T \
      - Gd @ Gd.conj().T) \
      - Gu @ C["phiu"] @ Gu.conj().T \
      - Gd @ C["phid"] @ Gd.conj().T \
      + 2*my_einsum("rspt,tp", C["qe"], Ge.conj().T @ Ge) \
      - 2*my_einsum("ptrs,tp", C["lq1"], Ge @ Ge.conj().T) \
      + 3/2*(Gd @ Gd.conj().T @ C["phiq1"] \
      + Gu @ Gu.conj().T @ C["phiq1"] \
      + C["phiq1"] @ Gd @ Gd.conj().T \
      + C["phiq1"] @ Gu @ Gu.conj().T \
      + 3*Gd @ Gd.conj().T @ C["phiq3"] \
      - 3*Gu @ Gu.conj().T @ C["phiq3"] \
      + 3*C["phiq3"] @ Gd @ Gd.conj().T \
      - 3*C["phiq3"] @ Gu @ Gu.conj().T) \
      - 2*(6*my_einsum("ptrs,tp", C["qq1"], Gd @ Gd.conj().T) \
      + my_einsum("psrt,tp", C["qq1"], Gd @ Gd.conj().T) \
      + 3*my_einsum("psrt,tp", C["qq3"], Gd @ Gd.conj().T) \
      - 6*my_einsum("ptrs,tp", C["qq1"], Gu @ Gu.conj().T) \
      - my_einsum("psrt,tp", C["qq1"], Gu @ Gu.conj().T) \
      - 3*my_einsum("psrt,tp", C["qq3"], Gu @ Gu.conj().T)) \
      - 6*my_einsum("rspt,tp", C["qu1"], Gu.conj().T @ Gu) \
      + 6*my_einsum("rspt,tp", C["qd1"], Gd.conj().T @ Gd) \
      + 2*GammaH*C["phiq1"] \
      + Gammaq @ C["phiq1"] \
      + C["phiq1"] @ Gammaq

    #I3 #co
    Beta["phiq3"] = 2/3*g**2*(1/4*C["phiBox"] \
      + np.trace(C["phil3"]) \
      + 3*np.trace(C["phiq3"]))*I3 \
      - 17/3*g**2*C["phiq3"] \
      + 2/3*g**2*(my_einsum("ttrs", C["lq3"]) \
      + my_einsum("rtts", C["qq1"]) \
      + 6*my_einsum("rstt", C["qq3"]) \
      - my_einsum("rtts", C["qq3"])) \
      - 1/2*C["phiBox"]*(Gu @ Gu.conj().T \
      + Gd @ Gd.conj().T) \
      + 1/2*(3*Gd @ Gd.conj().T @ C["phiq1"] \
      - 3*Gu @ Gu.conj().T @ C["phiq1"] \
      + 3*C["phiq1"] @ Gd @ Gd.conj().T \
      - 3*C["phiq1"] @ Gu @ Gu.conj().T \
      + Gd @ Gd.conj().T @ C["phiq3"] \
      + Gu @ Gu.conj().T @ C["phiq3"] \
      + C["phiq3"] @ Gd @ Gd.conj().T \
      + C["phiq3"] @ Gu @ Gu.conj().T) \
      - 2*(6*my_einsum("rspt,tp", C["qq3"], Gd @ Gd.conj().T) \
      + my_einsum("rtps,tp", C["qq1"], Gd @ Gd.conj().T) \
      - my_einsum("rtps,tp", C["qq3"], Gd @ Gd.conj().T) \
      + 6*my_einsum("rspt,tp", C["qq3"], Gu @ Gu.conj().T) \
      + my_einsum("rtps,tp", C["qq1"], Gu @ Gu.conj().T) \
      - my_einsum("rtps,tp", C["qq3"], Gu @ Gu.conj().T)) \
      - 2*my_einsum("ptrs,tp", C["lq3"], Ge @ Ge.conj().T) \
      + 2*GammaH*C["phiq3"] \
      + Gammaq @ C["phiq3"] \
      + C["phiq3"] @ Gammaq

    #I3 #co
    Beta["phiu"] = 1/3*XiB*gp**2*I3 \
      + 1/3*gp**2*C["phiu"] \
      - 2/3*gp**2*(my_einsum("ttrs", C["eu"]) \
      + my_einsum("ttrs", C["lu"]) \
      - my_einsum("ttrs", C["qu1"]) \
      + my_einsum("rstt", C["ud1"]) \
      - 4*my_einsum("rstt", C["uu"]) \
      - 4/3*my_einsum("rtts", C["uu"])) \
      - (C["phiBox"] \
      + C["phiD"])*Gu.conj().T @ Gu \
      - 2*Gu.conj().T @ C["phiq1"] @ Gu \
      + 3*(Gu.conj().T @ Gu @ C["phiu"] \
      + C["phiu"] @ Gu.conj().T @ Gu) \
      + Gu.conj().T @ Gd @ C["phiud"].conj().T \
      + C["phiud"] @ Gd.conj().T @ Gu \
      - 4*(3*my_einsum("rspt,tp", C["uu"], Gu.conj().T @ Gu) \
      + my_einsum("rtps,tp", C["uu"], Gu.conj().T @ Gu)) \
      + 2*my_einsum("ptrs,tp", C["eu"], Ge.conj().T @ Ge) \
      - 2*my_einsum("ptrs,tp", C["lu"], Ge @ Ge.conj().T) \
      + 6*my_einsum("rspt,tp", C["ud1"], Gd.conj().T @ Gd) \
      - 6*my_einsum("ptrs,tp", C["qu1"], Gd @ Gd.conj().T) \
      + 6*my_einsum("ptrs,tp", C["qu1"], Gu @ Gu.conj().T) \
      + 2*GammaH*C["phiu"] \
      + Gammau @ C["phiu"] \
      + C["phiu"] @ Gammau

    #I3 #co
    Beta["phid"] = -1/6*XiB*gp**2*I3 \
      + 1/3*gp**2*C["phid"] \
      - 2/3*gp**2*(2*my_einsum("rstt", C["dd"]) \
      + 2/3*my_einsum("rtts", C["dd"]) \
      + my_einsum("ttrs", C["ed"]) \
      + my_einsum("ttrs", C["ld"]) \
      - my_einsum("ttrs", C["qd1"]) \
      - 2*my_einsum("ttrs", C["ud1"])) \
      + (C["phiBox"] \
      + C["phiD"])*Gd.conj().T @ Gd \
      - 2*Gd.conj().T @ C["phiq1"] @ Gd \
      + 3*(Gd.conj().T @ Gd @ C["phid"] \
      + C["phid"] @ Gd.conj().T @ Gd) \
      - Gd.conj().T @ Gu @ C["phiud"] \
      - C["phiud"].conj().T @ Gu.conj().T @ Gd \
      + 4*(3*my_einsum("rspt,tp", C["dd"], Gd.conj().T @ Gd) \
      + my_einsum("rtps,tp", C["dd"], Gd.conj().T @ Gd)) \
      + 2*my_einsum("ptrs,tp", C["ed"], Ge.conj().T @ Ge) \
      - 2*my_einsum("ptrs,tp", C["ld"], Ge @ Ge.conj().T) \
      - 6*my_einsum("ptrs,tp", C["ud1"], Gu.conj().T @ Gu) \
      - 6*my_einsum("ptrs,tp", C["qd1"], Gd @ Gd.conj().T) \
      + 6*my_einsum("ptrs,tp", C["qd1"], Gu @ Gu.conj().T) \
      + 2*GammaH*C["phid"] \
      + Gammad @ C["phid"] \
      + C["phid"] @ Gammad

        #co
    Beta["phiud"] = -3*gp**2*C["phiud"] \
      + (2*C["phiBox"] \
      - C["phiD"])*Gu.conj().T @ Gd \
      - 2*Gu.conj().T @ Gd @ C["phid"] \
      + 2*C["phiu"] @ Gu.conj().T @ Gd \
      + 4*(my_einsum("rtps,tp", C["ud1"], Gu.conj().T @ Gd) \
      + 4/3*my_einsum("rtps,tp", C["ud8"], Gu.conj().T @ Gd)) \
      + 2*Gu.conj().T @ Gu @ C["phiud"] \
      + 2*C["phiud"] @ Gd.conj().T @ Gd \
      + 2*GammaH*C["phiud"] \
      + Gammau @ C["phiud"] \
      + C["phiud"] @ Gammad

    """Dimension-5"""
    Beta["llphiphi"] = (2*Lambda \
      - 3*g**2 \
      + 2*GammaH)*C["llphiphi"]-3/2*(C["llphiphi"] @ Ge @ Ge.conj().T \
      + Ge.conj() @ Ge.T @ C["llphiphi"])

    """(3,3,3,3)"""
    # the einsum function is strong
    Beta["ll"] = -1/6*gp**2*my_einsum("st,pr", C["phil1"], I3) \
      - 1/6*g**2*(my_einsum("st,pr", C["phil3"], I3) \
      - 2*my_einsum("sr,pt", C["phil3"], I3)) \
      + 1/3*gp**2*(2*my_einsum("prww,st", C["ll"], I3) \
      + my_einsum("pwwr,st", C["ll"], I3)) \
      - 1/3*g**2*my_einsum("pwwr,st", C["ll"], I3) \
      + 2/3*g**2*my_einsum("swwr,pt", C["ll"], I3) \
      - 1/3*gp**2*my_einsum("prww,st", C["lq1"], I3) \
      - g**2*my_einsum("prww,st", C["lq3"], I3) \
      + 2*g**2*my_einsum("ptww,rs", C["lq3"], I3) \
      + 1/3*gp**2*( \
      - 2*my_einsum("prww,st", C["lu"], I3) \
      + my_einsum("prww,st", C["ld"], I3) \
      + my_einsum("prww,st", C["le"], I3)) \
      - 1/2*(my_einsum("pr,st", Ge @ Ge.conj().T, C["phil1"]) \
      - my_einsum("pr,st", Ge @ Ge.conj().T, C["phil3"])) \
      - my_einsum("pt,sr", Ge @ Ge.conj().T, C["phil3"]) \
      - 1/2*my_einsum("sv,tw,prvw", Ge, np.conj(Ge), C["le"]) \
      + my_einsum("pv,vrst", Gammal, C["ll"]) \
      + my_einsum("pvst,vr", C["ll"], Gammal) \
      - 1/6*gp**2*my_einsum("pr,st", C["phil1"], I3) \
      - 1/6*g**2*(my_einsum("pr,st", C["phil3"], I3) \
      - 2*my_einsum("pt,sr", C["phil3"], I3)) \
      + 1/3*gp**2*(2*my_einsum("stww,pr", C["ll"], I3) \
      + my_einsum("swwt,pr", C["ll"], I3)) \
      - 1/3*g**2*my_einsum("swwt,pr", C["ll"], I3) \
      + 2/3*g**2*my_einsum("pwwt,sr", C["ll"], I3) \
      - 1/3*gp**2*my_einsum("stww,pr", C["lq1"], I3) \
      - g**2*my_einsum("stww,pr", C["lq3"], I3) \
      + 2*g**2*my_einsum("srww,tp", C["lq3"], I3) \
      + 1/3*gp**2*( \
      - 2*my_einsum("stww,pr", C["lu"], I3) \
      + my_einsum("stww,pr", C["ld"], I3) \
      + my_einsum("stww,pr", C["le"], I3)) \
      - 1/2*(my_einsum("st,pr", Ge @ Ge.conj().T, C["phil1"]) \
      - my_einsum("st,pr", Ge @ Ge.conj().T, C["phil3"])) \
      - my_einsum("sr,pt", Ge @ Ge.conj().T, C["phil3"]) \
      - 1/2*my_einsum("pv,rw,stvw", Ge, np.conj(Ge), C["le"]) \
      + my_einsum("sv,vtpr", Gammal, C["ll"]) \
      + my_einsum("svpr,vt", C["ll"], Gammal) \
      + 6*g**2*my_einsum("ptsr", C["ll"]) \
      + 3*(gp**2 \
      - g**2)*my_einsum("prst", C["ll"])

    Beta["qq1"] = 1/18*gp**2*my_einsum("st,pr", C["phiq1"], I3) \
      - 1/9*gp**2*my_einsum("wwst,pr", C["lq1"], I3) \
      + 1/9*gp**2*(2*my_einsum("prww,st", C["qq1"], I3) \
      + 1/3*(my_einsum("pwwr,st", C["qq1"], I3) \
      + 3*my_einsum("pwwr,st", C["qq3"], I3))) \
      + 1/3*gs**2*(my_einsum("swwr,pt", C["qq1"], I3) \
      + 3*my_einsum("swwr,pt", C["qq3"], I3)) \
      - 2/9*gs**2*(my_einsum("pwwr,st", C["qq1"], I3) \
      + 3*my_einsum("pwwr,st", C["qq3"], I3)) \
      + 2/9*gp**2*my_einsum("prww,st", C["qu1"], I3) \
      - 1/9*gp**2*my_einsum("prww,st", C["qd1"], I3) \
      + 1/12*gs**2*(my_einsum("srww,pt", C["qu8"], I3) \
      + my_einsum("srww,pt", C["qd8"], I3)) \
      - 1/18*gs**2*(my_einsum("prww,st", C["qu8"], I3) \
      + my_einsum("prww,st", C["qd8"], I3)) \
      - 1/9*gp**2*my_einsum("prww,st", C["qe"], I3) \
      + 1/2*(my_einsum("pr,st", Gu @ Gu.conj().T, C["phiq1"]) \
      - my_einsum("pr,st", Gd @ Gd.conj().T, C["phiq1"])) \
      - 1/2*(my_einsum("pv,rw,stvw", Gu, np.conj(Gu), C["qu1"]) \
      - 1/6*my_einsum("pv,rw,stvw", Gu, np.conj(Gu), C["qu8"])) \
      - 1/2*(my_einsum("pv,rw,stvw", Gd, np.conj(Gd), C["qd1"]) \
      - 1/6*my_einsum("pv,rw,stvw", Gd, np.conj(Gd), C["qd8"])) \
      - 1/8*(my_einsum("pv,tw,srvw", Gu, np.conj(Gu), C["qu8"]) \
      + my_einsum("pv,tw,srvw", Gd, np.conj(Gd), C["qd8"])) \
      - 1/8*(my_einsum("tw,rv,pvsw", np.conj(Gd), np.conj(Gu), C["quqd1"]) \
      - 1/6*my_einsum("tw,rv,pvsw", np.conj(Gd), np.conj(Gu), C["quqd8"])) \
      - 1/8*(my_einsum("sw,pv,rvtw", Gd, Gu, np.conj(C["quqd1"])) \
      - 1/6*my_einsum("sw,pv,rvtw", Gd, Gu, np.conj(C["quqd8"]))) \
      + 1/16*(my_einsum("tw,rv,svpw", np.conj(Gd), np.conj(Gu), C["quqd8"]) \
      + my_einsum("sw,pv,tvrw", Gd, Gu, np.conj(C["quqd8"]))) \
      + my_einsum("pv,vrst", Gammaq, C["qq1"]) \
      + my_einsum("pvst,vr", C["qq1"], Gammaq) \
      + 1/18*gp**2*my_einsum("pr,st", C["phiq1"], I3) \
      - 1/9*gp**2*my_einsum("wwpr,st", C["lq1"], I3) \
      + 1/9*gp**2*(2*my_einsum("stww,pr", C["qq1"], I3) \
      + 1/3*(my_einsum("swwt,pr", C["qq1"], I3) \
      + 3*my_einsum("swwt,pr", C["qq3"], I3))) \
      + 1/3*gs**2*(my_einsum("pwwt,sr", C["qq1"], I3) \
      + 3*my_einsum("pwwt,sr", C["qq3"], I3)) \
      - 2/9*gs**2*(my_einsum("swwt,pr", C["qq1"], I3) \
      + 3*my_einsum("swwt,pr", C["qq3"], I3)) \
      + 2/9*gp**2*my_einsum("stww,pr", C["qu1"], I3) \
      - 1/9*gp**2*my_einsum("stww,pr", C["qd1"], I3) \
      + 1/12*gs**2*(my_einsum("ptww,sr", C["qu8"], I3) \
      + my_einsum("ptww,sr", C["qd8"], I3)) \
      - 1/18*gs**2*(my_einsum("stww,pr", C["qu8"], I3) \
      + my_einsum("stww,pr", C["qd8"], I3)) \
      - 1/9*gp**2*my_einsum("stww,pr", C["qe"], I3) \
      + 1/2*(my_einsum("st,pr", Gu @ Gu.conj().T, C["phiq1"]) \
      - my_einsum("st,pr", Gd @ Gd.conj().T, C["phiq1"])) \
      - 1/2*(my_einsum("sv,tw,prvw", Gu, np.conj(Gu), C["qu1"]) \
      - 1/6*my_einsum("sv,tw,prvw", Gu, np.conj(Gu), C["qu8"])) \
      - 1/2*(my_einsum("sv,tw,prvw", Gd, np.conj(Gd), C["qd1"]) \
      - 1/6*my_einsum("sv,tw,prvw", Gd, np.conj(Gd), C["qd8"])) \
      - 1/8*(my_einsum("sv,rw,ptvw", Gu, np.conj(Gu), C["qu8"]) \
      + my_einsum("sv,rw,ptvw", Gd, np.conj(Gd), C["qd8"])) \
      - 1/8*(my_einsum("rw,tv,svpw", np.conj(Gd), np.conj(Gu), C["quqd1"]) \
      - 1/6*my_einsum("rw,tv,svpw", np.conj(Gd), np.conj(Gu), C["quqd8"])) \
      - 1/8*(my_einsum("pw,sv,tvrw", Gd, Gu, np.conj(C["quqd1"])) \
      - 1/6*my_einsum("pw,sv,tvrw", Gd, Gu, np.conj(C["quqd8"]))) \
      + 1/16*(my_einsum("rw,tv,pvsw", np.conj(Gd), np.conj(Gu), C["quqd8"]) \
      + my_einsum("pw,sv,rvtw", Gd, Gu, np.conj(C["quqd8"]))) \
      + my_einsum("sv,vtpr", Gammaq, C["qq1"]) \
      + my_einsum("svpr,vt", C["qq1"], Gammaq) \
      + 9*g**2*my_einsum("prst", C["qq3"]) \
      - 2*(gs**2 \
      - 1/6*gp**2)*my_einsum("prst", C["qq1"]) \
      + 3*gs**2*(my_einsum("ptsr", C["qq1"]) \
      + 3*my_einsum("ptsr", C["qq3"]))

    Beta["qq3"] = 1/6*g**2*my_einsum("st,pr", C["phiq3"], I3) \
      + 1/3*g**2*my_einsum("wwst,pr", C["lq3"], I3) \
      + 1/3*g**2*(my_einsum("pwwr,st", C["qq1"], I3) \
      - my_einsum("pwwr,st", C["qq3"], I3)) \
      + 2*g**2*my_einsum("prww,st", C["qq3"], I3) \
      + 1/3*gs**2*(my_einsum("swwr,pt", C["qq1"], I3) \
      + 3*my_einsum("swwr,pt", C["qq3"], I3)) \
      + 1/12*gs**2*(my_einsum("srww,pt", C["qu8"], I3) \
      + my_einsum("srww,pt", C["qd8"], I3)) \
      - 1/2*(my_einsum("pr,st", Gu @ Gu.conj().T, C["phiq3"]) \
      + my_einsum("pr,st", Gd @ Gd.conj().T, C["phiq3"])) \
      - 1/8*(my_einsum("pv,tw,srvw", Gu, np.conj(Gu), C["qu8"]) \
      + my_einsum("pv,tw,srvw", Gd, np.conj(Gd), C["qd8"])) \
      + 1/8*(my_einsum("tw,rv,pvsw", np.conj(Gd), np.conj(Gu), C["quqd1"]) \
      - 1/6*my_einsum("tw,rv,pvsw", np.conj(Gd), np.conj(Gu), C["quqd8"])) \
      + 1/8*(my_einsum("sw,pv,rvtw", Gd, Gu, np.conj(C["quqd1"])) \
      - 1/6*my_einsum("sw,pv,rvtw", Gd, Gu, np.conj(C["quqd8"]))) \
      - 1/16*(my_einsum("tw,rv,svpw", np.conj(Gd), np.conj(Gu), C["quqd8"]) \
      + my_einsum("sw,pv,tvrw", Gd, Gu, np.conj(C["quqd8"]))) \
      + my_einsum("pv,vrst", Gammaq, C["qq3"]) \
      + my_einsum("pvst,vr", C["qq3"], Gammaq) \
      + 1/6*g**2*my_einsum("pr,st", C["phiq3"], I3) \
      + 1/3*g**2*my_einsum("wwpr,st", C["lq3"], I3) \
      + 1/3*g**2*(my_einsum("swwt,pr", C["qq1"], I3) \
      - my_einsum("swwt,pr", C["qq3"], I3)) \
      + 2*g**2*my_einsum("stww,pr", C["qq3"], I3) \
      + 1/3*gs**2*(my_einsum("pwwt,sr", C["qq1"], I3) \
      + 3*my_einsum("pwwt,sr", C["qq3"], I3)) \
      + 1/12*gs**2*(my_einsum("ptww,sr", C["qu8"], I3) \
      + my_einsum("ptww,sr", C["qd8"], I3)) \
      - 1/2*(my_einsum("st,pr", Gu @ Gu.conj().T, C["phiq3"]) \
      + my_einsum("st,pr", Gd @ Gd.conj().T, C["phiq3"])) \
      - 1/8*(my_einsum("sv,rw,ptvw", Gu, np.conj(Gu), C["qu8"]) \
      + my_einsum("sv,rw,ptvw", Gd, np.conj(Gd), C["qd8"])) \
      + 1/8*(my_einsum("rw,tv,svpw", np.conj(Gd), np.conj(Gu), C["quqd1"]) \
      - 1/6*my_einsum("rw,tv,svpw", np.conj(Gd), np.conj(Gu), C["quqd8"])) \
      + 1/8*(my_einsum("pw,sv,tvrw", Gd, Gu, np.conj(C["quqd1"])) \
      - 1/6*my_einsum("pw,sv,tvrw", Gd, Gu, np.conj(C["quqd8"]))) \
      - 1/16*(my_einsum("rw,tv,pvsw", np.conj(Gd), np.conj(Gu), C["quqd8"]) \
      + my_einsum("pw,sv,rvtw", Gd, Gu, np.conj(C["quqd8"]))) \
      + my_einsum("sv,vtpr", Gammaq, C["qq3"]) \
      + my_einsum("svpr,vt", C["qq3"], Gammaq) \
      + 3*gs**2*(my_einsum("ptsr", C["qq1"]) \
      - my_einsum("ptsr", C["qq3"])) \
      - 2*(gs**2 \
      + 3*g**2 \
      - 1/6*gp**2)*my_einsum("prst", C["qq3"]) \
      + 3*g**2*my_einsum("prst", C["qq1"])

    #the terms are equal, but the order is not. No wonder if you check some differences inside
    Beta["lq1"] = -1/3*gp**2*my_einsum("st,pr", C["phiq1"], I3) \
      + 1/9*gp**2*my_einsum("pr,st", C["phil1"], I3) \
      - 2/9*gp**2*(2*my_einsum("prww,st", C["ll"], I3) \
      + my_einsum("pwwr,st", C["ll"], I3)) \
      + 2/9*gp**2*my_einsum("prww,st", C["lq1"], I3) \
      + 2/3*gp**2*my_einsum("wwst,pr", C["lq1"], I3) \
      - 2/9*gp**2*(6*my_einsum("stww,pr", C["qq1"], I3) \
      + my_einsum("swwt,pr", C["qq1"], I3) \
      + 3*my_einsum("swwt,pr", C["qq3"], I3)) \
      - 2/3*gp**2*(2*my_einsum("stww,pr", C["qu1"], I3) \
      - my_einsum("stww,pr", C["qd1"], I3) \
      - my_einsum("stww,pr", C["qe"], I3)) \
      + 2/9*gp**2*(2*my_einsum("prww,st", C["lu"], I3) \
      - my_einsum("prww,st", C["ld"], I3) \
      - my_einsum("prww,st", C["le"], I3)) \
      - gp**2*my_einsum("prst", C["lq1"]) \
      + 9*g**2*my_einsum("prst", C["lq3"]) \
      - my_einsum("pr,st", Ge @ Ge.conj().T, C["phiq1"]) \
      + my_einsum("st,pr", Gu @ Gu.conj().T, C["phil1"]) \
      - my_einsum("st,pr", Gd @ Gd.conj().T, C["phil1"]) \
      + 1/4*(my_einsum("tw,rv,pvsw", np.conj(Gu), np.conj(Ge), C["lequ1"]) \
      - 12*my_einsum("tw,rv,pvsw", np.conj(Gu), np.conj(Ge), C["lequ3"]) \
      + my_einsum("sw,pv,rvtw", Gu, Ge, np.conj(C["lequ1"])) \
      - 12*my_einsum("sw,pv,rvtw", Gu, Ge, np.conj(C["lequ3"]))) \
      - my_einsum("sv,tw,prvw", Gu, np.conj(Gu), C["lu"]) \
      - my_einsum("sv,tw,prvw", Gd, np.conj(Gd), C["ld"]) \
      - my_einsum("pv,rw,stvw", Ge, np.conj(Ge), C["qe"]) \
      + 1/4*(my_einsum("sw,rv,pvwt", Gd, np.conj(Ge), C["ledq"]) \
      + my_einsum("pv,tw,rvws", Ge, np.conj(Gd), np.conj(C["ledq"]))) \
      + my_einsum("pv,vrst", Gammal, C["lq1"]) \
      + my_einsum("sv,prvt", Gammaq, C["lq1"]) \
      + my_einsum("pvst,vr", C["lq1"], Gammal) \
      + my_einsum("prsv,vt", C["lq1"], Gammaq)

    Beta["lq3"] = 1/3*g**2*(my_einsum("st,pr", C["phiq3"], I3) \
      + my_einsum("pr,st", C["phil3"], I3)) \
      + 2/3*g**2*(3*my_einsum("prww,st", C["lq3"], I3) \
      + my_einsum("wwst,pr", C["lq3"], I3)) \
      + 2/3*g**2*(6*my_einsum("stww,pr", C["qq3"], I3) \
      + my_einsum("swwt,pr", C["qq1"], I3) \
      - my_einsum("swwt,pr", C["qq3"], I3)) \
      + 2/3*g**2*my_einsum("pwwr,st", C["ll"], I3) \
      + 3*g**2*my_einsum("prst", C["lq1"]) \
      - (6*g**2 \
      + gp**2)*my_einsum("prst", C["lq3"]) \
      - my_einsum("pr,st", Ge @ Ge.conj().T, C["phiq3"]) \
      - my_einsum("st,pr", Gu @ Gu.conj().T, C["phil3"]) \
      - my_einsum("st,pr", Gd @ Gd.conj().T, C["phil3"]) \
      - 1/4*(my_einsum("tw,rv,pvsw", np.conj(Gu), np.conj(Ge), C["lequ1"]) \
      - 12*my_einsum("tw,rv,pvsw", np.conj(Gu), np.conj(Ge), C["lequ3"]) \
      + my_einsum("sw,pv,rvtw", Gu, Ge, np.conj(C["lequ1"])) \
      - 12*my_einsum("sw,pv,rvtw", Gu, Ge, np.conj(C["lequ3"]))) \
      + 1/4*(my_einsum("sw,rv,pvwt", Gd, np.conj(Ge), C["ledq"]) \
      + my_einsum("pv,tw,rvws", Ge, np.conj(Gd), np.conj(C["ledq"]))) \
      + my_einsum("pv,vrst", Gammal, C["lq3"]) \
      + my_einsum("sv,prvt", Gammaq, C["lq3"]) \
      + my_einsum("pvst,vr", C["lq3"], Gammal) \
      + my_einsum("prsv,vt", C["lq3"], Gammaq)

    #order
    Beta["ee"] = -1/3*gp**2*my_einsum("st,pr", C["phie"], I3) \
      + 2/3*gp**2*(my_einsum("wwpr,st", C["le"], I3) \
      - my_einsum("wwpr,st", C["qe"], I3) \
      - 2*my_einsum("prww,st", C["eu"], I3) \
      + my_einsum("prww,st", C["ed"], I3) \
      + 4*my_einsum("prww,st", C["ee"], I3)) \
      + my_einsum("pr,st", Ge.conj().T @ Ge, C["phie"]) \
      - my_einsum("wr,vp,vwst", Ge, np.conj(Ge), C["le"]) \
      + my_einsum("pv,vrst", Gammae, C["ee"]) \
      + my_einsum("pvst,vr", C["ee"], Gammae) \
      - 1/3*gp**2*my_einsum("pr,st", C["phie"], I3) \
      + 2/3*gp**2*(my_einsum("wwst,pr", C["le"], I3) \
      - my_einsum("wwst,pr", C["qe"], I3) \
      - 2*my_einsum("stww,pr", C["eu"], I3) \
      + my_einsum("stww,pr", C["ed"], I3) \
      + 4*my_einsum("wwst,pr", C["ee"], I3)) \
      + my_einsum("st,pr", Ge.conj().T @ Ge, C["phie"]) \
      - my_einsum("wt,vs,vwpr", Ge, np.conj(Ge), C["le"]) \
      + my_einsum("sv,vtpr", Gammae, C["ee"]) \
      + my_einsum("svpr,vt", C["ee"], Gammae) \
      + 12*gp**2*my_einsum("prst", C["ee"])

    #order
    Beta["uu"] = 2/9*gp**2*my_einsum("st,pr", C["phiu"], I3) \
      - 4/9*gp**2*(my_einsum("wwst,pr", C["eu"], I3) \
      + my_einsum("wwst,pr", C["lu"], I3) \
      - my_einsum("wwst,pr", C["qu1"], I3) \
      - 4*my_einsum("wwst,pr", C["uu"], I3) \
      - 4/3*my_einsum("swwt,pr", C["uu"], I3)) \
      - 1/9*gs**2*(my_einsum("wwst,pr", C["qu8"], I3) \
      - 3*my_einsum("wwsr,pt", C["qu8"], I3)) \
      + 2/3*gs**2*my_einsum("pwwt,rs", C["uu"], I3) \
      - 2/9*gs**2*my_einsum("swwt,pr", C["uu"], I3) \
      - 4/9*gp**2*my_einsum("stww,pr", C["ud1"], I3) \
      - 1/18*gs**2*(my_einsum("stww,pr", C["ud8"], I3) \
      - 3*my_einsum("srww,pt", C["ud8"], I3)) \
      - my_einsum("pr,st", Gu.conj().T @ Gu, C["phiu"]) \
      - (my_einsum("wr,vp,vwst", Gu, np.conj(Gu), C["qu1"]) \
      - 1/6*my_einsum("wr,vp,vwst", Gu, np.conj(Gu), C["qu8"])) \
      - 1/2*my_einsum("wr,vs,vwpt", Gu, np.conj(Gu), C["qu8"]) \
      + my_einsum("pv,vrst", Gammau, C["uu"]) \
      + my_einsum("pvst,vr", C["uu"], Gammau) \
      + 2/9*gp**2*my_einsum("pr,st", C["phiu"], I3) \
      - 4/9*gp**2*(my_einsum("wwpr,st", C["eu"], I3) \
      + my_einsum("wwpr,st", C["lu"], I3) \
      - my_einsum("wwpr,st", C["qu1"], I3) \
      - 4*my_einsum("wwpr,st", C["uu"], I3) \
      - 4/3*my_einsum("pwwr,st", C["uu"], I3)) \
      - 1/9*gs**2*(my_einsum("wwpr,st", C["qu8"], I3) \
      - 3*my_einsum("wwpt,sr", C["qu8"], I3)) \
      + 2/3*gs**2*my_einsum("swwr,tp", C["uu"], I3) \
      - 2/9*gs**2*my_einsum("pwwr,st", C["uu"], I3) \
      - 4/9*gp**2*my_einsum("prww,st", C["ud1"], I3) \
      - 1/18*gs**2*(my_einsum("prww,st", C["ud8"], I3) \
      - 3*my_einsum("ptww,sr", C["ud8"], I3)) \
      - my_einsum("st,pr", Gu.conj().T @ Gu, C["phiu"]) \
      - (my_einsum("wt,vs,vwpr", Gu, np.conj(Gu), C["qu1"]) \
      - 1/6*my_einsum("wt,vs,vwpr", Gu, np.conj(Gu), C["qu8"])) \
      - 1/2*my_einsum("wt,vp,vwsr", Gu, np.conj(Gu), C["qu8"]) \
      + my_einsum("sv,vtpr", Gammau, C["uu"]) \
      + my_einsum("svpr,vt", C["uu"], Gammau) \
      + 2*(8/3*gp**2 \
      - gs**2)*my_einsum("prst", C["uu"]) \
      + 6*gs**2*my_einsum("ptsr", C["uu"])

    #order
    Beta["dd"] = -1/9*gp**2*my_einsum("st,pr", C["phid"], I3) \
      + 2/9*gp**2*(my_einsum("wwst,pr", C["ed"], I3) \
      + my_einsum("wwst,pr", C["ld"], I3) \
      - my_einsum("wwst,pr", C["qd1"], I3) \
      + 2*my_einsum("wwst,pr", C["dd"], I3) \
      + 2/3*my_einsum("swwt,pr", C["dd"], I3)) \
      - 1/9*gs**2*(my_einsum("wwst,pr", C["qd8"], I3) \
      - 3*my_einsum("wwsr,pt", C["qd8"], I3)) \
      + 2/3*gs**2*my_einsum("pwwt,rs", C["dd"], I3) \
      - 2/9*gs**2*my_einsum("swwt,pr", C["dd"], I3) \
      - 4/9*gp**2*my_einsum("wwst,pr", C["ud1"], I3) \
      - 1/18*gs**2*(my_einsum("wwst,pr", C["ud8"], I3) \
      - 3*my_einsum("wwsr,pt", C["ud8"], I3)) \
      + my_einsum("pr,st", Gd.conj().T @ Gd, C["phid"]) \
      - (my_einsum("wr,vp,vwst", Gd, np.conj(Gd), C["qd1"]) \
      - 1/6*my_einsum("wr,vp,vwst", Gd, np.conj(Gd), C["qd8"])) \
      - 1/2*my_einsum("wr,vs,vwpt", Gd, np.conj(Gd), C["qd8"]) \
      + my_einsum("pv,vrst", Gammad, C["dd"]) \
      + my_einsum("pvst,vr", C["dd"], Gammad) \
      - 1/9*gp**2*my_einsum("pr,st", C["phid"], I3) \
      + 2/9*gp**2*(my_einsum("wwpr,st", C["ed"], I3) \
      + my_einsum("wwpr,st", C["ld"], I3) \
      - my_einsum("wwpr,st", C["qd1"], I3) \
      + 2*my_einsum("wwpr,st", C["dd"], I3) \
      + 2/3*my_einsum("pwwr,st", C["dd"], I3)) \
      - 1/9*gs**2*(my_einsum("wwpr,st", C["qd8"], I3) \
      - 3*my_einsum("wwpt,sr", C["qd8"], I3)) \
      + 2/3*gs**2*my_einsum("swwr,tp", C["dd"], I3) \
      - 2/9*gs**2*my_einsum("pwwr,st", C["dd"], I3) \
      - 4/9*gp**2*my_einsum("wwpr,st", C["ud1"], I3) \
      - 1/18*gs**2*(my_einsum("wwpr,st", C["ud8"], I3) \
      - 3*my_einsum("wwpt,sr", C["ud8"], I3)) \
      + my_einsum("st,pr", Gd.conj().T @ Gd, C["phid"]) \
      - (my_einsum("wt,vs,vwpr", Gd, np.conj(Gd), C["qd1"]) \
      - 1/6*my_einsum("wt,vs,vwpr", Gd, np.conj(Gd), C["qd8"])) \
      - 1/2*my_einsum("wt,vp,vwsr", Gd, np.conj(Gd), C["qd8"]) \
      + my_einsum("sv,vtpr", Gammad, C["dd"]) \
      + my_einsum("svpr,vt", C["dd"], Gammad) \
      + 2*(2/3*gp**2 \
      - gs**2)*my_einsum("prst", C["dd"]) \
      + 6*gs**2*my_einsum("ptsr", C["dd"])

    Beta["eu"] = -2/3*gp**2*(my_einsum("st,pr", C["phiu"], I3) \
      + 2*(my_einsum("wwst,pr", C["qu1"], I3) \
      - my_einsum("wwst,pr", C["lu"], I3) \
      + 4*my_einsum("wwst,pr", C["uu"], I3) \
      - my_einsum("wwst,pr", C["eu"], I3) \
      - my_einsum("stww,pr", C["ud1"], I3)) \
      + 8/3*my_einsum("swwt,pr", C["uu"], I3)) \
      + 4/9*gp**2*(my_einsum("pr,st", C["phie"], I3) \
      + 2*(my_einsum("wwpr,st", C["qe"], I3) \
      - my_einsum("wwpr,st", C["le"], I3) \
      - 4*my_einsum("prww,st", C["ee"], I3) \
      + 2*my_einsum("prww,st", C["eu"], I3) \
      - my_einsum("prww,st", C["ed"], I3))) \
      - 8*gp**2*my_einsum("prst", C["eu"]) \
      + 2*my_einsum("pr,st", Ge.conj().T @ Ge, C["phiu"]) \
      - 2*my_einsum("st,pr", Gu.conj().T @ Gu, C["phie"]) \
      + my_einsum("vp,ws,vrwt", np.conj(Ge), np.conj(Gu), C["lequ1"]) \
      - 12*my_einsum("vp,ws,vrwt", np.conj(Ge), np.conj(Gu), C["lequ3"]) \
      + my_einsum("vr,wt,vpws", Ge, Gu, np.conj(C["lequ1"])) \
      - 12*my_einsum("vr,wt,vpws", Ge, Gu, np.conj(C["lequ3"])) \
      - 2*my_einsum("vp,wr,vwst", np.conj(Ge), Ge, C["lu"]) \
      - 2*my_einsum("vs,wt,vwpr", np.conj(Gu), Gu, C["qe"]) \
      + my_einsum("pv,vrst", Gammae, C["eu"]) \
      + my_einsum("sv,prvt", Gammau, C["eu"]) \
      + my_einsum("pvst,vr", C["eu"], Gammae) \
      + my_einsum("prsv,vt", C["eu"], Gammau)

    Beta["ed"] = -2/3*gp**2*(my_einsum("st,pr", C["phid"], I3) \
      + 2*(my_einsum("wwst,pr", C["qd1"], I3) \
      - my_einsum("wwst,pr", C["ld"], I3) \
      - 2*my_einsum("wwst,pr", C["dd"], I3) \
      - my_einsum("wwst,pr", C["ed"], I3) \
      + 2*my_einsum("wwst,pr", C["ud1"], I3)) \
      - 4/3*my_einsum("swwt,pr", C["dd"], I3)) \
      - 2/9*gp**2*(my_einsum("pr,st", C["phie"], I3) \
      + 2*(my_einsum("wwpr,st", C["qe"], I3) \
      - my_einsum("wwpr,st", C["le"], I3) \
      - 4*my_einsum("prww,st", C["ee"], I3) \
      - my_einsum("prww,st", C["ed"], I3) \
      + 2*my_einsum("prww,st", C["eu"], I3))) \
      + 4*gp**2*my_einsum("prst", C["ed"]) \
      + 2*my_einsum("pr,st", Ge.conj().T @ Ge, C["phid"]) \
      + 2*my_einsum("st,pr", Gd.conj().T @ Gd, C["phie"]) \
      - 2*my_einsum("vp,wr,vwst", np.conj(Ge), Ge, C["ld"]) \
      - 2*my_einsum("vs,wt,vwpr", np.conj(Gd), Gd, C["qe"]) \
      + my_einsum("vp,wt,vrsw", np.conj(Ge), Gd, C["ledq"]) \
      + my_einsum("vr,ws,vptw", Ge, np.conj(Gd), np.conj(C["ledq"])) \
      + my_einsum("pv,vrst", Gammae, C["ed"]) \
      + my_einsum("sv,prvt", Gammad, C["ed"]) \
      + my_einsum("pvst,vr", C["ed"], Gammae) \
      + my_einsum("prsv,vt", C["ed"], Gammad)

    #order
    Beta["ud1"] = 4/9*gp**2*(my_einsum("st,pr", C["phid"], I3) \
      + 2*(my_einsum("wwst,pr", C["qd1"], I3) \
      - my_einsum("wwst,pr", C["ld"], I3) \
      - 2*my_einsum("wwst,pr", C["dd"], I3) \
      + 2*my_einsum("wwst,pr", C["ud1"], I3) \
      - my_einsum("wwst,pr", C["ed"], I3)) \
      - 4/3*my_einsum("swwt,pr", C["dd"], I3)) \
      - 2/9*gp**2*(my_einsum("pr,st", C["phiu"], I3) \
      + 2*(my_einsum("wwpr,st", C["qu1"], I3) \
      - my_einsum("wwpr,st", C["lu"], I3) \
      + 4*my_einsum("wwpr,st", C["uu"], I3) \
      - my_einsum("prww,st", C["ud1"], I3) \
      - my_einsum("wwpr,st", C["eu"], I3)) \
      + 8/3*my_einsum("pwwr,st", C["uu"], I3)) \
      - 8/3*(gp**2*my_einsum("prst", C["ud1"]) \
      - gs**2*my_einsum("prst", C["ud8"])) \
      - 2*my_einsum("pr,st", Gu.conj().T @ Gu, C["phid"]) \
      + 2*my_einsum("st,pr", Gd.conj().T @ Gd, C["phiu"]) \
      + 2/3*my_einsum("sr,pt", Gd.conj().T @ Gu, C["phiud"]) \
      + 2/3*my_einsum("pt,rs", Gu.conj().T @ Gd, np.conj(C["phiud"])) \
      + 1/3*(my_einsum("vs,wp,vrwt", np.conj(Gd), np.conj(Gu), C["quqd1"]) \
      + 4/3*my_einsum("vs,wp,vrwt", np.conj(Gd), np.conj(Gu), C["quqd8"]) \
      + my_einsum("vt,wr,vpws", Gd, Gu, np.conj(C["quqd1"])) \
      + 4/3*my_einsum("vt,wr,vpws", Gd, Gu, np.conj(C["quqd8"]))) \
      - my_einsum("ws,vp,vrwt", np.conj(Gd), np.conj(Gu), C["quqd1"]) \
      - my_einsum("wt,vr,vpws", Gd, Gu, np.conj(C["quqd1"])) \
      - 2*my_einsum("vp,wr,vwst", np.conj(Gu), Gu, C["qd1"]) \
      - 2*my_einsum("vs,wt,vwpr", np.conj(Gd), Gd, C["qu1"]) \
      + my_einsum("pv,vrst", Gammau, C["ud1"]) \
      + my_einsum("sv,prvt", Gammad, C["ud1"]) \
      + my_einsum("pvst,vr", C["ud1"], Gammau) \
      + my_einsum("prsv,vt", C["ud1"], Gammad)

    #order
    Beta["ud8"] = 8/3*gs**2*my_einsum("pwwr,st", C["uu"], I3) \
      + 8/3*gs**2*my_einsum("swwt,pr", C["dd"], I3) \
      + 4/3*gs**2*my_einsum("wwpr,st", C["qu8"], I3) \
      + 4/3*gs**2*my_einsum("wwst,pr", C["qd8"], I3) \
      + 2/3*gs**2*my_einsum("prww,st", C["ud8"], I3) \
      + 2/3*gs**2*my_einsum("wwst,pr", C["ud8"], I3) \
      - 4*(2/3*gp**2 \
      + gs**2)*my_einsum("prst", C["ud8"]) \
      + 12*gs**2*my_einsum("prst", C["ud1"]) \
      + 4*my_einsum("sr,pt", Gd.conj().T @ Gu, C["phiud"]) \
      + 4*my_einsum("pt,rs", Gu.conj().T @ Gd, np.conj(C["phiud"])) \
      + 2*(my_einsum("vs,wp,vrwt", np.conj(Gd), np.conj(Gu), C["quqd1"]) \
      - 1/6*my_einsum("vs,wp,vrwt", np.conj(Gd), np.conj(Gu), C["quqd8"]) \
      + my_einsum("vt,wr,vpws", Gd, Gu, np.conj(C["quqd1"])) \
      - 1/6*my_einsum("vt,wr,vpws", Gd, Gu, np.conj(C["quqd8"]))) \
      - 2*my_einsum("vp,wr,vwst", np.conj(Gu), Gu, C["qd8"]) \
      - 2*my_einsum("vs,wt,vwpr", np.conj(Gd), Gd, C["qu8"]) \
      - (my_einsum("ws,vp,vrwt", np.conj(Gd), np.conj(Gu), C["quqd8"]) \
      + my_einsum("wt,vr,vpws", Gd, Gu, np.conj(C["quqd8"]))) \
      + my_einsum("pv,vrst", Gammau, C["ud8"]) \
      + my_einsum("sv,prvt", Gammad, C["ud8"]) \
      + my_einsum("pvst,vr", C["ud8"], Gammau) \
      + my_einsum("prsv,vt", C["ud8"], Gammad)

    Beta["le"] = -1/3*gp**2*my_einsum("st,pr", C["phie"], I3) \
      - 2/3*gp**2*my_einsum("pr,st", C["phil1"], I3) \
      + 8/3*gp**2*my_einsum("prww,st", C["ll"], I3) \
      + 4/3*gp**2*my_einsum("pwwr,st", C["ll"], I3) \
      - 4/3*gp**2*my_einsum("prww,st", C["lq1"], I3) \
      - 2/3*gp**2*my_einsum("wwst,pr", C["qe"], I3) \
      + 4/3*gp**2*my_einsum("prww,st", C["le"], I3) \
      + 2/3*gp**2*my_einsum("wwst,pr", C["le"], I3) \
      - 8/3*gp**2*my_einsum("prww,st", C["lu"], I3) \
      + 4/3*gp**2*my_einsum("prww,st", C["ld"], I3) \
      - 4/3*gp**2*my_einsum("stww,pr", C["eu"], I3) \
      + 2/3*gp**2*my_einsum("stww,pr", C["ed"], I3) \
      + 8/3*gp**2*my_einsum("wwst,pr", C["ee"], I3) \
      - 6*gp**2*my_einsum("prst", C["le"]) \
      + my_einsum("rs,pt", np.conj(Ge), Xie) \
      + my_einsum("pt,rs", Ge, np.conj(Xie)) \
      - my_einsum("pr,st", Ge @ Ge.conj().T, C["phie"]) \
      + 2*my_einsum("st,pr", Ge.conj().T @ Ge, C["phil1"]) \
      - 4*my_einsum("pv,rw,vtsw", Ge, np.conj(Ge), C["ee"]) \
      + my_einsum("pw,vs,vrwt", Ge, np.conj(Ge), C["le"]) \
      - 2*my_einsum("wt,vs,pwvr", Ge, np.conj(Ge), C["ll"]) \
      - 4*my_einsum("wt,vs,prvw", Ge, np.conj(Ge), C["ll"]) \
      + my_einsum("vt,rw,pvsw", Ge, np.conj(Ge), C["le"]) \
      + my_einsum("pv,vrst", Gammal, C["le"]) \
      + my_einsum("sv,prvt", Gammae, C["le"]) \
      + my_einsum("pvst,vr", C["le"], Gammal) \
      + my_einsum("prsv,vt", C["le"], Gammae)

    #order
    Beta["lu"] = -1/3*gp**2*my_einsum("st,pr", C["phiu"], I3) \
      + 4/9*gp**2*my_einsum("pr,st", C["phil1"], I3) \
      - 16/9*gp**2*my_einsum("prww,st", C["ll"], I3) \
      - 8/9*gp**2*my_einsum("pwwr,st", C["ll"], I3) \
      + 8/9*gp**2*my_einsum("prww,st", C["lq1"], I3) \
      - 2/3*gp**2*my_einsum("wwst,pr", C["qu1"], I3) \
      + 16/9*gp**2*my_einsum("prww,st", C["lu"], I3) \
      + 2/3*gp**2*my_einsum("wwst,pr", C["lu"], I3) \
      - 8/9*gp**2*my_einsum("prww,st", C["ld"], I3) \
      - 8/9*gp**2*my_einsum("prww,st", C["le"], I3) \
      + 2/3*gp**2*my_einsum("stww,pr", C["ud1"], I3) \
      + 2/3*gp**2*my_einsum("wwst,pr", C["eu"], I3) \
      - 8/3*gp**2*my_einsum("stww,pr", C["uu"], I3) \
      - 8/9*gp**2*my_einsum("swwt,pr", C["uu"], I3) \
      + 4*gp**2*my_einsum("prst", C["lu"]) \
      - my_einsum("pr,st", Ge @ Ge.conj().T, C["phiu"]) \
      - 2*my_einsum("st,pr", Gu.conj().T @ Gu, C["phil1"]) \
      - 1/2*(my_einsum("rv,ws,pvwt", np.conj(Ge), np.conj(Gu), C["lequ1"]) \
      + 12*my_einsum("rv,ws,pvwt", np.conj(Ge), np.conj(Gu), C["lequ3"])) \
      - 1/2*(my_einsum("pv,wt,rvws", Ge, Gu, np.conj(C["lequ1"])) \
      + 12*my_einsum("pv,wt,rvws", Ge, Gu, np.conj(C["lequ3"]))) \
      - 2*my_einsum("vs,wt,prvw", np.conj(Gu), Gu, C["lq1"]) \
      - my_einsum("rw,pv,vwst", np.conj(Ge), Ge, C["eu"]) \
      + my_einsum("pv,vrst", Gammal, C["lu"]) \
      + my_einsum("sv,prvt", Gammau, C["lu"]) \
      + my_einsum("pvst,vr", C["lu"], Gammal) \
      + my_einsum("prsv,vt", C["lu"], Gammau)

    Beta["ld"] = -1/3*gp**2*my_einsum("st,pr", C["phid"], I3) \
      - 2/9*gp**2*my_einsum("pr,st", C["phil1"], I3) \
      + 8/9*gp**2*my_einsum("prww,st", C["ll"], I3) \
      + 4/9*gp**2*my_einsum("pwwr,st", C["ll"], I3) \
      - 4/9*gp**2*my_einsum("prww,st", C["lq1"], I3) \
      - 2/3*gp**2*my_einsum("wwst,pr", C["qd1"], I3) \
      + 4/9*gp**2*my_einsum("prww,st", C["ld"], I3) \
      + 2/3*gp**2*my_einsum("wwst,pr", C["ld"], I3) \
      - 8/9*gp**2*my_einsum("prww,st", C["lu"], I3) \
      + 4/9*gp**2*my_einsum("prww,st", C["le"], I3) \
      - 4/3*gp**2*my_einsum("wwst,pr", C["ud1"], I3) \
      + 2/3*gp**2*my_einsum("wwst,pr", C["ed"], I3) \
      + 4/3*gp**2*my_einsum("stww,pr", C["dd"], I3) \
      + 4/9*gp**2*my_einsum("swwt,pr", C["dd"], I3) \
      - 2*gp**2*my_einsum("prst", C["ld"]) \
      - my_einsum("pr,st", Ge @ Ge.conj().T, C["phid"]) \
      + 2*my_einsum("st,pr", Gd.conj().T @ Gd, C["phil1"]) \
      - 1/2*my_einsum("rv,wt,pvsw", np.conj(Ge), Gd, C["ledq"]) \
      - 1/2*my_einsum("pv,ws,rvtw", Ge, np.conj(Gd), np.conj(C["ledq"])) \
      - 2*my_einsum("vs,wt,prvw", np.conj(Gd), Gd, C["lq1"]) \
      - my_einsum("rw,pv,vwst", np.conj(Ge), Ge, C["ed"]) \
      + my_einsum("pv,vrst", Gammal, C["ld"]) \
      + my_einsum("sv,prvt", Gammad, C["ld"]) \
      + my_einsum("pvst,vr", C["ld"], Gammal) \
      + my_einsum("prsv,vt", C["ld"], Gammad)

    Beta["qe"] = 1/9*gp**2*my_einsum("st,pr", C["phie"], I3) \
      - 2/3*gp**2*my_einsum("pr,st", C["phiq1"], I3) \
      - 8/3*gp**2*my_einsum("prww,st", C["qq1"], I3) \
      - 4/9*gp**2*(my_einsum("pwwr,st", C["qq1"], I3) \
      + 3*my_einsum("pwwr,st", C["qq3"], I3)) \
      + 4/3*gp**2*my_einsum("wwpr,st", C["lq1"], I3) \
      - 2/9*gp**2*my_einsum("wwst,pr", C["le"], I3) \
      + 4/3*gp**2*my_einsum("prww,st", C["qe"], I3) \
      + 2/9*gp**2*my_einsum("wwst,pr", C["qe"], I3) \
      - 8/3*gp**2*my_einsum("prww,st", C["qu1"], I3) \
      + 4/3*gp**2*my_einsum("prww,st", C["qd1"], I3) \
      + 4/9*gp**2*my_einsum("stww,pr", C["eu"], I3) \
      - 2/9*gp**2*my_einsum("stww,pr", C["ed"], I3) \
      - 8/9*gp**2*my_einsum("wwst,pr", C["ee"], I3) \
      + 2*gp**2*my_einsum("prst", C["qe"]) \
      + my_einsum("pr,st", Gu @ Gu.conj().T, C["phie"]) \
      - my_einsum("pr,st", Gd @ Gd.conj().T, C["phie"]) \
      + 2*my_einsum("st,pr", Ge.conj().T @ Ge, C["phiq1"]) \
      - 1/2*my_einsum("pw,vs,vtwr", Gd, np.conj(Ge), C["ledq"]) \
      - 1/2*my_einsum("vt,rw,vswp", Ge, np.conj(Gd), np.conj(C["ledq"])) \
      - 2*my_einsum("vs,wt,vwpr", np.conj(Ge), Ge, C["lq1"]) \
      - 1/2*(my_einsum("rw,vs,vtpw", np.conj(Gu), np.conj(Ge), C["lequ1"]) \
      + 12*my_einsum("rw,vs,vtpw", np.conj(Gu), np.conj(Ge), C["lequ3"])) \
      - 1/2*(my_einsum("pw,vt,vsrw", Gu, Ge, np.conj(C["lequ1"])) \
      + 12*my_einsum("pw,vt,vsrw", Gu, Ge, np.conj(C["lequ3"]))) \
      - my_einsum("rw,pv,stvw", np.conj(Gd), Gd, C["ed"]) \
      - my_einsum("rw,pv,stvw", np.conj(Gu), Gu, C["eu"]) \
      + my_einsum("pv,vrst", Gammaq, C["qe"]) \
      + my_einsum("sv,prvt", Gammae, C["qe"]) \
      + my_einsum("pvst,vr", C["qe"], Gammaq) \
      + my_einsum("prsv,vt", C["qe"], Gammae)

    Beta["qu1"] = 1/9*gp**2*my_einsum("st,pr", C["phiu"], I3) \
      + 4/9*gp**2*my_einsum("pr,st", C["phiq1"], I3) \
      + 16/9*gp**2*my_einsum("prww,st", C["qq1"], I3) \
      + 8/27*gp**2*(my_einsum("pwwr,st", C["qq1"], I3) \
      + 3*my_einsum("pwwr,st", C["qq3"], I3)) \
      - 8/9*gp**2*my_einsum("wwpr,st", C["lq1"], I3) \
      - 8/9*gp**2*my_einsum("prww,st", C["qe"], I3) \
      - 8/9*gp**2*my_einsum("prww,st", C["qd1"], I3) \
      + 16/9*gp**2*my_einsum("prww,st", C["qu1"], I3) \
      + 2/9*gp**2*my_einsum("wwst,pr", C["qu1"], I3) \
      - 2/9*gp**2*my_einsum("wwst,pr", C["lu"], I3) \
      - 2/9*gp**2*my_einsum("wwst,pr", C["eu"], I3) \
      - 2/9*gp**2*my_einsum("stww,pr", C["ud1"], I3) \
      + 8/9*gp**2*my_einsum("stww,pr", C["uu"], I3) \
      + 8/27*gp**2*my_einsum("swwt,pr", C["uu"], I3) \
      - 4/3*gp**2*my_einsum("prst", C["qu1"]) \
      - 8/3*gs**2*my_einsum("prst", C["qu8"]) \
      + 1/3*my_einsum("rs,pt", np.conj(Gu), Xiu) \
      + 1/3*my_einsum("pt,rs", Gu, np.conj(Xiu)) \
      + my_einsum("pr,st", Gu @ Gu.conj().T, C["phiu"]) \
      - my_einsum("pr,st", Gd @ Gd.conj().T, C["phiu"]) \
      - 2*my_einsum("st,pr", Gu.conj().T @ Gu, C["phiq1"]) \
      + 1/3*(my_einsum("pw,vs,vrwt", Gu, np.conj(Gu), C["qu1"]) \
      + 4/3*my_einsum("pw,vs,vrwt", Gu, np.conj(Gu), C["qu8"])) \
      + 1/3*(my_einsum("vt,rw,pvsw", Gu, np.conj(Gu), C["qu1"]) \
      + 4/3*my_einsum("vt,rw,pvsw", Gu, np.conj(Gu), C["qu8"])) \
      + 1/3*(my_einsum("rw,vs,ptvw", np.conj(Gd), np.conj(Gu), C["quqd1"]) \
      + 4/3*my_einsum("rw,vs,ptvw", np.conj(Gd), np.conj(Gu), C["quqd8"])) \
      + 1/3*(my_einsum("pw,vt,rsvw", Gd, Gu, np.conj(C["quqd1"])) \
      + 4/3*my_einsum("pw,vt,rsvw", Gd, Gu, np.conj(C["quqd8"]))) \
      + 1/2*my_einsum("rw,vs,vtpw", np.conj(Gd), np.conj(Gu), C["quqd1"]) \
      + 1/2*my_einsum("pw,vt,vsrw", Gd, Gu, np.conj(C["quqd1"])) \
      - 2/3*(my_einsum("vt,ws,pvwr", Gu, np.conj(Gu), C["qq1"]) \
      + 3*my_einsum("vt,ws,pvwr", Gu, np.conj(Gu), C["qq3"])) \
      - 4*my_einsum("wt,vs,prvw", Gu, np.conj(Gu), C["qq1"]) \
      - 2/3*my_einsum("pv,rw,vtsw", Gu, np.conj(Gu), C["uu"]) \
      - 2*my_einsum("pv,rw,vwst", Gu, np.conj(Gu), C["uu"]) \
      - my_einsum("pv,rw,stvw", Gd, np.conj(Gd), C["ud1"]) \
      + my_einsum("pv,vrst", Gammaq, C["qu1"]) \
      + my_einsum("sv,prvt", Gammau, C["qu1"]) \
      + my_einsum("pvst,vr", C["qu1"], Gammaq) \
      + my_einsum("prsv,vt", C["qu1"], Gammau)

    Beta["qd1"] = 1/9*gp**2*my_einsum("st,pr", C["phid"], I3) \
      - 2/9*gp**2*my_einsum("pr,st", C["phiq1"], I3) \
      - 8/9*gp**2*my_einsum("prww,st", C["qq1"], I3) \
      - 4/27*gp**2*(my_einsum("pwwr,st", C["qq1"], I3) \
      + 3*my_einsum("pwwr,st", C["qq3"], I3)) \
      + 4/9*gp**2*my_einsum("wwpr,st", C["lq1"], I3) \
      + 4/9*gp**2*my_einsum("prww,st", C["qe"], I3) \
      - 8/9*gp**2*my_einsum("prww,st", C["qu1"], I3) \
      + 4/9*gp**2*my_einsum("prww,st", C["qd1"], I3) \
      + 2/9*gp**2*my_einsum("wwst,pr", C["qd1"], I3) \
      - 2/9*gp**2*my_einsum("wwst,pr", C["ld"], I3) \
      - 2/9*gp**2*my_einsum("wwst,pr", C["ed"], I3) \
      + 4/9*gp**2*my_einsum("wwst,pr", C["ud1"], I3) \
      - 4/9*gp**2*my_einsum("stww,pr", C["dd"], I3) \
      - 4/27*gp**2*my_einsum("swwt,pr", C["dd"], I3) \
      + 2/3*gp**2*my_einsum("prst", C["qd1"]) \
      - 8/3*gs**2*my_einsum("prst", C["qd8"]) \
      + 1/3*my_einsum("rs,pt", np.conj(Gd), Xid) \
      + 1/3*my_einsum("pt,rs", Gd, np.conj(Xid)) \
      + my_einsum("pr,st", Gu @ Gu.conj().T, C["phid"]) \
      - my_einsum("pr,st", Gd @ Gd.conj().T, C["phid"]) \
      + 2*my_einsum("st,pr", Gd.conj().T @ Gd, C["phiq1"]) \
      + 1/3*(my_einsum("pw,vs,vrwt", Gd, np.conj(Gd), C["qd1"]) \
      + 4/3*my_einsum("pw,vs,vrwt", Gd, np.conj(Gd), C["qd8"])) \
      + 1/3*(my_einsum("vt,rw,pvsw", Gd, np.conj(Gd), C["qd1"]) \
      + 4/3*my_einsum("vt,rw,pvsw", Gd, np.conj(Gd), C["qd8"])) \
      + 1/3*(my_einsum("rw,vs,vwpt", np.conj(Gu), np.conj(Gd), C["quqd1"]) \
      + 4/3*my_einsum("rw,vs,vwpt", np.conj(Gu), np.conj(Gd), C["quqd8"])) \
      + 1/3*(my_einsum("pw,vt,vwrs", Gu, Gd, np.conj(C["quqd1"])) \
      + 4/3*my_einsum("pw,vt,vwrs", Gu, Gd, np.conj(C["quqd8"]))) \
      + 1/2*my_einsum("ws,rv,pvwt", np.conj(Gd), np.conj(Gu), C["quqd1"]) \
      + 1/2*my_einsum("pv,wt,rvws", Gu, Gd, np.conj(C["quqd1"])) \
      - 2/3*(my_einsum("vt,ws,pvwr", Gd, np.conj(Gd), C["qq1"]) \
      + 3*my_einsum("vt,ws,pvwr", Gd, np.conj(Gd), C["qq3"])) \
      - 4*my_einsum("wt,vs,prvw", Gd, np.conj(Gd), C["qq1"]) \
      - 2/3*my_einsum("pv,rw,vtsw", Gd, np.conj(Gd), C["dd"]) \
      - 2*my_einsum("pv,rw,vwst", Gd, np.conj(Gd), C["dd"]) \
      - my_einsum("pv,rw,vwst", Gu, np.conj(Gu), C["ud1"]) \
      + my_einsum("pv,vrst", Gammaq, C["qd1"]) \
      + my_einsum("sv,prvt", Gammad, C["qd1"]) \
      + my_einsum("pvst,vr", C["qd1"], Gammaq) \
      + my_einsum("prsv,vt", C["qd1"], Gammad)

    Beta["qu8"] = 8/3*gs**2*(my_einsum("pwwr,st", C["qq1"], I3) \
      + 3*my_einsum("pwwr,st", C["qq3"], I3)) \
      + 2/3*gs**2*my_einsum("prww,st", C["qu8"], I3) \
      + 2/3*gs**2*my_einsum("prww,st", C["qd8"], I3) \
      + 4/3*gs**2*my_einsum("wwst,pr", C["qu8"], I3) \
      + 2/3*gs**2*my_einsum("stww,pr", C["ud8"], I3) \
      + 8/3*gs**2*my_einsum("swwt,pr", C["uu"], I3) \
      - (4/3*gp**2 \
      + 14*gs**2)*my_einsum("prst", C["qu8"]) \
      - 12*gs**2*my_einsum("prst", C["qu1"]) \
      + 2*my_einsum("rs,pt", np.conj(Gu), Xiu) \
      + 2*my_einsum("pt,rs", Gu, np.conj(Xiu)) \
      + 2*(my_einsum("pw,vs,vrwt", Gu, np.conj(Gu), C["qu1"]) \
      - 1/6*my_einsum("pw,vs,vrwt", Gu, np.conj(Gu), C["qu8"])) \
      + 2*(my_einsum("vt,rw,pvsw", Gu, np.conj(Gu), C["qu1"]) \
      - 1/6*my_einsum("vt,rw,pvsw", Gu, np.conj(Gu), C["qu8"])) \
      + 2*(my_einsum("rw,vs,ptvw", np.conj(Gd), np.conj(Gu), C["quqd1"]) \
      - 1/6*my_einsum("rw,vs,ptvw", np.conj(Gd), np.conj(Gu), C["quqd8"])) \
      + 2*(my_einsum("pw,vt,rsvw", Gd, Gu, np.conj(C["quqd1"])) \
      - 1/6*my_einsum("pw,vt,rsvw", Gd, Gu, np.conj(C["quqd8"]))) \
      + 1/2*my_einsum("vs,rw,vtpw", np.conj(Gu), np.conj(Gd), C["quqd8"]) \
      + 1/2*my_einsum("vt,pw,vsrw", Gu, Gd, np.conj(C["quqd8"])) \
      - 4*(my_einsum("vt,ws,pvwr", Gu, np.conj(Gu), C["qq1"]) \
      + 3*my_einsum("vt,ws,pvwr", Gu, np.conj(Gu), C["qq3"])) \
      - 4*my_einsum("pv,rw,vtsw", Gu, np.conj(Gu), C["uu"]) \
      - my_einsum("pv,rw,stvw", Gd, np.conj(Gd), C["ud8"]) \
      + my_einsum("pv,vrst", Gammaq, C["qu8"]) \
      + my_einsum("sv,prvt", Gammau, C["qu8"]) \
      + my_einsum("pvst,vr", C["qu8"], Gammaq) \
      + my_einsum("prsv,vt", C["qu8"], Gammau)

    Beta["qd8"] = 8/3*gs**2*(my_einsum("pwwr,st", C["qq1"], I3) \
      + 3*my_einsum("pwwr,st", C["qq3"], I3)) \
      + 2/3*gs**2*my_einsum("prww,st", C["qu8"], I3) \
      + 2/3*gs**2*my_einsum("prww,st", C["qd8"], I3) \
      + 4/3*gs**2*my_einsum("wwst,pr", C["qd8"], I3) \
      + 2/3*gs**2*my_einsum("wwst,pr", C["ud8"], I3) \
      + 8/3*gs**2*my_einsum("swwt,pr", C["dd"], I3) \
      - (-2/3*gp**2 \
      + 14*gs**2)*my_einsum("prst", C["qd8"]) \
      - 12*gs**2*my_einsum("prst", C["qd1"]) \
      + 2*my_einsum("rs,pt", np.conj(Gd), Xid) \
      + 2*my_einsum("pt,rs", Gd, np.conj(Xid)) \
      + 2*(my_einsum("pw,vs,vrwt", Gd, np.conj(Gd), C["qd1"]) \
      - 1/6*my_einsum("pw,vs,vrwt", Gd, np.conj(Gd), C["qd8"])) \
      + 2*(my_einsum("vt,rw,pvsw", Gd, np.conj(Gd), C["qd1"]) \
      - 1/6*my_einsum("vt,rw,pvsw", Gd, np.conj(Gd), C["qd8"])) \
      + 2*(my_einsum("rw,vs,vwpt", np.conj(Gu), np.conj(Gd), C["quqd1"]) \
      - 1/6*my_einsum("rw,vs,vwpt", np.conj(Gu), np.conj(Gd), C["quqd8"])) \
      + 2*(my_einsum("pw,vt,vwrs", Gu, Gd, np.conj(C["quqd1"])) \
      - 1/6*my_einsum("pw,vt,vwrs", Gu, Gd, np.conj(C["quqd8"]))) \
      + 1/2*my_einsum("vs,rw,pwvt", np.conj(Gd), np.conj(Gu), C["quqd8"]) \
      + 1/2*my_einsum("vt,pw,rwvs", Gd, Gu, np.conj(C["quqd8"])) \
      - 4*(my_einsum("vt,ws,pvwr", Gd, np.conj(Gd), C["qq1"]) \
      + 3*my_einsum("vt,ws,pvwr", Gd, np.conj(Gd), C["qq3"])) \
      - 4*my_einsum("pv,rw,vtsw", Gd, np.conj(Gd), C["dd"]) \
      - my_einsum("pv,rw,vwst", Gu, np.conj(Gu), C["ud8"]) \
      + my_einsum("pv,vrst", Gammaq, C["qd8"]) \
      + my_einsum("sv,prvt", Gammad, C["qd8"]) \
      + my_einsum("pvst,vr", C["qd8"], Gammaq) \
      + my_einsum("prsv,vt", C["qd8"], Gammad)

    Beta["ledq"] = -(8/3*gp**2 \
      + 8*gs**2)*my_einsum("prst", C["ledq"]) \
      - 2*my_einsum("ts,pr", np.conj(Gd), Xie) \
      - 2*my_einsum("pr,ts", Ge, np.conj(Xid)) \
      + 2*my_einsum("pv,tw,vrsw", Ge, np.conj(Gd), C["ed"]) \
      - 2*my_einsum("vr,tw,pvsw", Ge, np.conj(Gd), C["ld"]) \
      + 2*my_einsum("vr,ws,pvwt", Ge, np.conj(Gd), C["lq1"]) \
      + 6*my_einsum("vr,ws,pvwt", Ge, np.conj(Gd), C["lq3"]) \
      - 2*my_einsum("pw,vs,vtwr", Ge, np.conj(Gd), C["qe"]) \
      + 2*my_einsum("vs,tw,prvw", np.conj(Gd), np.conj(Gu), C["lequ1"]) \
      + my_einsum("pv,vrst", Gammal, C["ledq"]) \
      + my_einsum("sv,prvt", Gammad, C["ledq"]) \
      + my_einsum("pvst,vr", C["ledq"], Gammae) \
      + my_einsum("prsv,vt", C["ledq"], Gammaq)

    Beta["quqd1"] = 10/3*gp*my_einsum("st,pr", C["dB"], Gu) \
      - 6*g*my_einsum("st,pr", C["dW"], Gu) \
      - 20/9*gp*my_einsum("pt,sr", C["dB"], Gu) \
      + 4*g*my_einsum("pt,sr", C["dW"], Gu) \
      - 64/9*gs*my_einsum("pt,sr", C["dG"], Gu) \
      - 2/3*gp*my_einsum("pr,st", C["uB"], Gd) \
      - 6*g*my_einsum("pr,st", C["uW"], Gd) \
      + 4/9*gp*my_einsum("sr,pt", C["uB"], Gd) \
      + 4*g*my_einsum("sr,pt", C["uW"], Gd) \
      - 64/9*gs*my_einsum("sr,pt", C["uG"], Gd) \
      - 1/2*(11/9*gp**2 + 3*g**2 + 32*gs**2)*my_einsum("prst", C["quqd1"]) \
      - 1/3*( - 5/9*gp**2 - 3*g**2 + 64/3*gs**2)*my_einsum("srpt", C["quqd1"]) \
      - 4/9*( - 5/9*gp**2 - 3*g**2 + 28/3*gs**2)*my_einsum("srpt", C["quqd8"]) \
      + 16/9*gs**2*my_einsum("prst", C["quqd8"]) \
      - 2*my_einsum("pr,st", Gu, Xid) \
      - 2*my_einsum("st,pr", Gd, Xiu) \
      + 4/3*(my_einsum("vr,pw,svwt", Gu, Gd, C["qd1"]) \
      + 4/3*my_einsum("vr,pw,svwt", Gu, Gd, C["qd8"]) \
      + my_einsum("vt,sw,pvwr", Gd, Gu, C["qu1"]) \
      + 4/3*my_einsum("vt,sw,pvwr", Gd, Gu, C["qu8"]) \
      + my_einsum("pw,sv,vrwt", Gd, Gu, C["ud1"]) \
      + 4/3*my_einsum("pw,sv,vrwt", Gd, Gu, C["ud8"])) \
      + 8/3*(my_einsum("wt,vr,svpw", Gd, Gu, C["qq1"]) \
      - 3*my_einsum("wt,vr,svpw", Gd, Gu, C["qq3"]) \
      - 3*my_einsum("wt,vr,swpv", Gd, Gu, C["qq1"]) \
      + 9*my_einsum("wt,vr,swpv", Gd, Gu, C["qq3"])) \
      - 4*my_einsum("sw,pv,vrwt", Gd, Gu, C["ud1"]) \
      + my_einsum("pv,vrst", Gammaq, C["quqd1"]) \
      + my_einsum("sv,prvt", Gammaq, C["quqd1"]) \
      + my_einsum("pvst,vr", C["quqd1"], Gammau) \
      + my_einsum("prsv,vt", C["quqd1"], Gammad)

    Beta["quqd8"] = 8*gs*my_einsum("st,pr", C["dG"], Gu) \
      - 40/3*gp*my_einsum("pt,sr", C["dB"], Gu) \
      + 24*g*my_einsum("pt,sr", C["dW"], Gu) \
      + 16/3*gs*my_einsum("pt,sr", C["dG"], Gu) \
      + 8*gs*my_einsum("pr,st", C["uG"], Gd) \
      + 8/3*gp*my_einsum("sr,pt", C["uB"], Gd) \
      + 24*g*my_einsum("sr,pt", C["uW"], Gd) \
      + 16/3*gs*my_einsum("sr,pt", C["uG"], Gd) \
      + 8*gs**2*my_einsum("prst", C["quqd1"]) \
      + (10/9*gp**2 + 6*g**2 + 16/3*gs**2)*my_einsum("srpt", C["quqd1"]) \
      + (-11/18*gp**2 - 3/2*g**2 + 16/3*gs**2)*my_einsum("prst", C["quqd8"]) \
      - 1/3*(5/9*gp**2 + 3*g**2 \
      + 44/3*gs**2)*my_einsum("srpt", C["quqd8"]) \
      + 8*(my_einsum("vr,pw,svwt", Gu, Gd, C["qd1"]) \
      - 1/6*my_einsum("vr,pw,svwt", Gu, Gd, C["qd8"]) \
      + my_einsum("vt,sw,pvwr", Gd, Gu, C["qu1"]) \
      - 1/6*my_einsum("vt,sw,pvwr", Gd, Gu, C["qu8"]) \
      + my_einsum("pw,sv,vrwt", Gd, Gu, C["ud1"]) \
      - 1/6*my_einsum("pw,sv,vrwt", Gd, Gu, C["ud8"])) \
      + 16*(my_einsum("wt,vr,svpw", Gd, Gu, C["qq1"]) \
      - 3*my_einsum("wt,vr,svpw", Gd, Gu, C["qq3"])) \
      - 4*my_einsum("sw,pv,vrwt", Gd, Gu, C["ud8"]) \
      + my_einsum("pv,vrst", Gammaq, C["quqd8"]) \
      + my_einsum("sv,prvt", Gammaq, C["quqd8"]) \
      + my_einsum("pvst,vr", C["quqd8"], Gammau) \
      + my_einsum("prsv,vt", C["quqd8"], Gammad)

    Beta["lequ1"] = -(11/3*gp**2 + 8*gs**2)*my_einsum("prst", C["lequ1"]) \
      + (30*gp**2 + 18*g**2)*my_einsum("prst", C["lequ3"]) \
      + 2*my_einsum("st,pr", Gu, Xie) \
      + 2*my_einsum("pr,st", Ge, Xiu) \
      + 2*my_einsum("sv,wt,prvw", Gd, Gu, C["ledq"]) \
      + 2*my_einsum("pv,sw,vrwt", Ge, Gu, C["eu"]) \
      + 2*my_einsum("vr,wt,pvsw", Ge, Gu, C["lq1"]) \
      - 6*my_einsum("vr,wt,pvsw", Ge, Gu, C["lq3"]) \
      - 2*my_einsum("vr,sw,pvwt", Ge, Gu, C["lu"]) \
      - 2*my_einsum("pw,vt,svwr", Ge, Gu, C["qe"]) \
      + my_einsum("pv,vrst", Gammal, C["lequ1"]) \
      + my_einsum("sv,prvt", Gammaq, C["lequ1"]) \
      + my_einsum("pvst,vr", C["lequ1"], Gammae) \
      + my_einsum("prsv,vt", C["lequ1"], Gammau)

    Beta["lequ3"] = 5/6*gp*my_einsum("pr,st", C["eB"], Gu) \
      - 3/2*g*my_einsum("st,pr", C["uW"], Ge) \
      - 3/2*gp*my_einsum("st,pr", C["uB"], Ge) \
      - 3/2*g*my_einsum("pr,st", C["eW"], Gu) \
      + (2/9*gp**2 - 3*g**2 + 8/3*gs**2)*my_einsum("prst", C["lequ3"]) \
      + 1/8*(5*gp**2 + 3*g**2)*my_einsum("prst", C["lequ1"]) \
      - 1/2*my_einsum("sw,pv,vrwt", Gu, Ge, C["eu"]) \
      - 1/2*my_einsum("vr,wt,pvsw", Ge, Gu, C["lq1"]) \
      + 3/2*my_einsum("vr,wt,pvsw", Ge, Gu, C["lq3"]) \
      - 1/2*my_einsum("vr,sw,pvwt", Ge, Gu, C["lu"]) \
      - 1/2*my_einsum("pw,vt,svwr", Ge, Gu, C["qe"]) \
      + my_einsum("pv,vrst", Gammal, C["lequ3"]) \
      + my_einsum("sv,prvt", Gammaq, C["lequ3"]) \
      + my_einsum("pvst,vr", C["lequ3"], Gammae) \
      + my_einsum("prsv,vt", C["lequ3"], Gammau)

    Beta["duql"] = -(9/2*g**2 \
      + 11/6*gp**2 \
      + 4*gs**2)*my_einsum("prst", C["duql"]) \
      - my_einsum("sv,wp,vrwt", np.conj(Gd), Gd, C["duql"]) \
      - my_einsum("sv,wr,pvwt", np.conj(Gu), Gu, C["duql"]) \
      + 2*my_einsum("tv,sw,prwv", np.conj(Ge), np.conj(Gu), C["duue"]) \
      + my_einsum("tv,sw,pwrv", np.conj(Ge), np.conj(Gu), C["duue"]) \
      + 4*my_einsum("vp,wr,vwst", Gd, Gu, C["qqql"]) \
      + 4*my_einsum("vp,wr,wvst", Gd, Gu, C["qqql"]) \
      - my_einsum("vp,wr,vswt", Gd, Gu, C["qqql"]) \
      - my_einsum("vp,wr,wsvt", Gd, Gu, C["qqql"]) \
      + 2*my_einsum("wp,tv,wsrv", Gd, np.conj(Ge), C["qque"]) \
      + my_einsum("vp,vrst", Gd.conj().T @ Gd, C["duql"]) \
      + my_einsum("vr,pvst", Gu.conj().T @ Gu, C["duql"]) \
      + 1/2*(my_einsum("vs,prvt", Gu @ Gu.conj().T, C["duql"]) \
      + my_einsum("vs,prvt", Gd @ Gd.conj().T, C["duql"])) \
      + 1/2*my_einsum("vt,prsv", Ge @ Ge.conj().T, C["duql"])

    Beta["qque"] = -(9/2*g**2 \
      + 23/6*gp**2 + 4*gs**2)*my_einsum("prst", C["qque"]) \
      - my_einsum("rv,ws,pwvt", np.conj(Gu), Gu, C["qque"]) \
      + 1/2*my_einsum("wt,rv,vspw", Ge, np.conj(Gd), C["duql"]) \
      - 1/2*(2*my_einsum("pv,rw,vwst", np.conj(Gd), np.conj(Gu), C["duue"]) \
      + my_einsum("pv,rw,vswt", np.conj(Gd), np.conj(Gu), C["duue"])) \
      + 1/2*( \
      - 2*my_einsum("ws,vt,prwv", Gu, Ge, C["qqql"]) \
      + my_einsum("ws,vt,pwrv", Gu, Ge, C["qqql"]) \
      - 2*my_einsum("ws,vt,wprv", Gu, Ge, C["qqql"])) \
      + 1/2*(my_einsum("vp,vrst", Gu @ Gu.conj().T, C["qque"]) \
      + my_einsum("vp,vrst", Gd @ Gd.conj().T, C["qque"])) \
      - my_einsum("pv,ws,rwvt", np.conj(Gu), Gu, C["qque"]) \
      + 1/2*my_einsum("wt,pv,vsrw", Ge, np.conj(Gd), C["duql"]) \
      - 1/2*(2*my_einsum("rv,pw,vwst", np.conj(Gd), np.conj(Gu), C["duue"]) \
      + my_einsum("rv,pw,vswt", np.conj(Gd), np.conj(Gu), C["duue"])) \
      + 1/2*( \
      - 2*my_einsum("ws,vt,rpwv", Gu, Ge, C["qqql"]) \
      + my_einsum("ws,vt,rwpv", Gu, Ge, C["qqql"]) \
      - 2*my_einsum("ws,vt,wrpv", Gu, Ge, C["qqql"])) \
      + 1/2*(my_einsum("vr,vpst", Gu @ Gu.conj().T, C["qque"]) \
      + my_einsum("vr,vpst", Gd @ Gd.conj().T, C["qque"])) \
      + my_einsum("vs,prvt", Gu.conj().T @ Gu, C["qque"]) \
      + my_einsum("vt,prsv", Ge.conj().T @ Ge, C["qque"])

    Beta["qqql"] = -(3*g**2 \
      + 1/3*gp**2 + 4*gs**2)*my_einsum("prst", C["qqql"]) \
      - 4*g**2*(my_einsum("rpst", C["qqql"]) \
      + my_einsum("srpt", C["qqql"]) \
      + my_einsum("psrt", C["qqql"])) \
      - 4*my_einsum("tv,sw,prwv", np.conj(Ge), np.conj(Gu), C["qque"]) \
      + 2*(my_einsum("pv,rw,vwst", np.conj(Gd), np.conj(Gu), C["duql"]) \
      + my_einsum("rv,pw,vwst", np.conj(Gd), np.conj(Gu), C["duql"])) \
      + 1/2*(my_einsum("vp,vrst", Gu @ Gu.conj().T, C["qqql"]) \
      + my_einsum("vp,vrst", Gd @ Gd.conj().T, C["qqql"])) \
      + 1/2*(my_einsum("vr,pvst", Gu @ Gu.conj().T, C["qqql"]) \
      + my_einsum("vr,pvst", Gd @ Gd.conj().T, C["qqql"])) \
      + 1/2*(my_einsum("vs,prvt", Gu @ Gu.conj().T, C["qqql"]) \
      + my_einsum("vs,prvt", Gd @ Gd.conj().T, C["qqql"])) \
      + 1/2*my_einsum("vt,prsv", Ge @ Ge.conj().T, C["qqql"])

    Beta["duue"] = -(2*gp**2 + 4*gs**2)*my_einsum("prst", C["duue"]) \
      - 20/3*gp**2*my_einsum("psrt", C["duue"]) \
      + 4*my_einsum("ws,vt,prwv", Gu, Ge, C["duql"]) \
      - 8*my_einsum("vp,wr,vwst", Gd, Gu, C["qque"]) \
      + my_einsum("vp,vrst", Gd.conj().T @ Gd, C["duue"]) \
      + my_einsum("vr,pvst", Gu.conj().T @ Gu, C["duue"]) \
      + my_einsum("vs,prvt", Gu.conj().T @ Gu, C["duue"]) \
      + my_einsum("vt,prsv", Ge.conj().T @ Ge, C["duue"])

    Beta["llphiphi"] = (2*Lambda \
      - 3*g**2 \
      + 2*GammaH)*C["llphiphi"]-3/2*(C["llphiphi"] @ Ge @ Ge.conj().T \
      + Ge.conj() @ Ge.T @ C["llphiphi"])

    return Beta

def beta_array(C, HIGHSCALE, *args, **kwargs):
    """Return the beta functions of all SM parameters and SMEFT Wilson
    coefficients as a 1D numpy array."""
    beta_odict = beta(C, HIGHSCALE, *args, **kwargs)
    return np.hstack([np.asarray(b).ravel() for b in beta_odict.values()])

def C_array2dict(C):
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

def C_dict2array(C):
    """Convert an OrderedDict containing C values to a 1D array."""
    return np.hstack([np.asarray(C[k]).ravel() for k in C_keys])
