"""Functions for translating to and from the SMEFT Higgs-like basis.

The code was adapted from the Rosetta package:
https://github.com/kenmimasu/Rosetta/tree/master/Rosetta
(c) Adam Falkowski, Benjamin Fuks, Kentarou Mawatari, 
Ken Mimasu, Veronica Sanz & Francesco Riva
"""

from math import sqrt, pi

import ckmutil
import numpy as np

from wilson import wcxf
from wilson.parameters import p as default_parameters


def higgslike_to_warsaw_up(C, parameters=None, sectors=None):
    """Translate from the Higgs-Warsaw basis to the Warsaw up
    basis."""

    basis = wcxf.Basis["SMEFT", "Higgs-Warsaw up"]
    H = {k: C.get(k, 0) for k in basis.all_wcs}

    p = default_parameters.copy()
    if parameters is not None:
        # if parameters are passed in, overwrite the default values
        p.update(parameters)
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["delta"])
    Vdag = V.T.conjugate()

    W = H.copy()

    # parameters
    # neglecting dim.-6 corrections here as these are higher-order
    # when multiplying dim.-6 WCs!
    v = 1 / sqrt(sqrt(2) * p["GF"])  # Higgs VEV
    lam = p["m_h"] ** 2 / (2 * v ** 2)  # Higgs self-coupling
    e = sqrt(4 * pi * p["alpha_e"])
    s2w = 1 / 2 * (1 - sqrt(1 - (4 * pi * p['alpha_e']) / (sqrt(2) * p['GF'] * p['m_Z']**2)))
    gs = sqrt(4 * pi * p["alpha_s"])
    sw = sqrt(s2w)
    g = gw = e / sw
    c2w = 1 - s2w
    cw = sqrt(c2w)
    gp = e / cw
    gw2 = g ** 2
    gp2 = gp ** 2
    gs2 = gs ** 2

    dM = (
        H["deltagWlL_11"] + H["deltagWlL_22"] - H["ll_1221"] / 2
    ) / 2  # W mass correction

    W["phi"] = (
        3 * lam * H["deltacz"]
        + 8 * lam * dM
        - H["deltalambda3"]
        + (
            H["czBox"] * gw2
            + H["czz"] * gp2
            - H["czgamma"] * gp2 * (c2w - s2w)
            - H["cgammagamma"] * gw2 * s2w ** 2
        )
        * 4
        * gw2
        * lam
        / (gw2 - gp2)
    )

    W["phiBox"] = (
        H["deltacz"]
        + 2 * dM
        + (
            H["czBox"] * (3 * gw2 - gp2)
            + 2 * H["czz"] * gp2
            - 2 * H["czgamma"] * gp2 * (c2w - s2w)
            - 2 * H["cgammagamma"] * gw2 * s2w ** 2
        )
        * gw2
        / (gw2 - gp2)
        / 2
    )

    W["phiD"] = (
        -(
            H["czBox"]
            + H["czz"]
            - H["czgamma"] * (c2w - s2w)
            - H["cgammagamma"] * s2w * c2w
        )
        * 2
        * gw2
        * gp2
        / (gw2 - gp2)
        - 4 * dM
    )

    W["phiG"] = H["cgg"] * gs2 / 4

    W["phiW"] = (
        (H["czz"] + H["czgamma"] * 2 * s2w + H["cgammagamma"] * s2w ** 2) * gw2 / 4
    )

    W["phiB"] = (
        (H["czz"] - H["czgamma"] * 2 * c2w + H["cgammagamma"] * c2w ** 2) * gp2 / 4
    )

    W["phiWB"] = (
        (H["czz"] - H["czgamma"] * (c2w - s2w) - H["cgammagamma"] * c2w * s2w)
        * gw
        * gp
        / 2
    )

    W["phiGtilde"] = H["cggtilde"] * gs2 / 4

    W["phiWtilde"] = (
        (H["czztilde"] + H["czgammatilde"] * 2 * s2w + H["cgammagammatilde"] * s2w ** 2)
        * gw2
        / 4
    )

    W["phiBtilde"] = (
        (H["czztilde"] - H["czgammatilde"] * 2 * c2w + H["cgammagammatilde"] * c2w ** 2)
        * gp2
        / 4
    )

    W["phiWtildeB"] = (
        (
            H["czztilde"]
            - H["czgammatilde"] * (c2w - s2w)
            - H["cgammagammatilde"] * c2w * s2w
        )
        * gw
        * gp
        / 2
    )

    C1 = (
        (
            H["czBox"] * gw2
            + H["czz"] * gp2
            - H["czgamma"] * gp2 * (c2w - s2w)
            - H["cgammagamma"] * gw2 * s2w ** 2
        )
        * gw2
        / (gw2 - gp2)
    )

    C2 = (
        (
            H["czBox"]
            + H["czz"]
            - H["czgamma"] * (c2w - s2w)
            - H["cgammagamma"] * s2w * c2w
        )
        * gw2
        * gp2
        / (gw2 - gp2)
    )

    # rotate deltagZdL to up basis
    deltagZdL = np.zeros((3, 3), complex)
    for i in range(3):
        for j in range(3):
            if j >= i:
                ind = "_{}{}".format(i + 1, j + 1)
                deltagZdL[i, j] = H["deltagZdL" + ind]
    # symmetrize
    deltagZdL = deltagZdL + deltagZdL.T.conjugate() - np.diag(np.diag(deltagZdL))
    deltagZdLrot = V @ deltagZdL @ Vdag

    facp = np.eye(3) * C1
    fac = np.eye(3) * C2

    for i in range(3):
        for j in range(3):
            ind = "_{}{}".format(i + 1, j + 1)
            if j >= i:
                W["phil3" + ind] = H["deltagWlL" + ind] + facp[i, j] / 2
                W["phiq3" + ind] = (
                    H["deltagZuL" + ind] - deltagZdLrot[i, j] + facp[i, j] / 2
                )
                W["phil1" + ind] = (
                    -2 * H["deltagZeL" + ind] - H["deltagWlL" + ind] + fac[i, j] / 2
                )
                W["phiq1" + ind] = (
                    -H["deltagZuL" + ind] - deltagZdLrot[i, j] - fac[i, j] / 6
                )
                W["phie" + ind] = -2 * H["deltagZeR" + ind] + fac[i, j]
                W["phiu" + ind] = -2 * H["deltagZuR" + ind] - 2 / 3 * fac[i, j]
                W["phid" + ind] = -2 * H["deltagZdR" + ind] + 1 / 3 * fac[i, j]

            W["phiud" + ind] = -2 * H["deltagWqR" + ind]

    warsaw = wcxf.Basis["SMEFT", "Warsaw up"]
    all_wcs = set(warsaw.all_wcs)  # to speed up lookup
    return {k: v for k, v in W.items() if k in all_wcs}


def warsaw_up_to_higgslike(C, parameters=None, sectors=None):

    basis = wcxf.Basis["SMEFT", "Warsaw up"]
    W = {k: C.get(k, 0) for k in basis.all_wcs}
    H = W.copy()

    p = default_parameters.copy()
    if parameters is not None:
        # if parameters are passed in, overwrite the default values
        p.update(parameters)
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["delta"])
    Vdag = V.T.conjugate()

    # parameters
    # neglecting dim.-6 corrections here as these are higher-order
    # when multiplying dim.-6 WCs!
    v = 1 / sqrt(sqrt(2) * p["GF"])  # Higgs VEV
    lam = p["m_h"] ** 2 / (2 * v ** 2)  # Higgs self-coupling
    e = sqrt(4 * pi * p["alpha_e"])
    s2w = 1 / 2 * (1 - sqrt(1 - (4 * pi * p['alpha_e']) / (sqrt(2) * p['GF'] * p['m_Z']**2)))
    gs = sqrt(4 * pi * p["alpha_s"])
    sw = sqrt(s2w)
    g = gw = e / sw
    c2w = 1 - s2w
    cw = sqrt(c2w)
    gp = e / cw
    gw2 = g ** 2
    gp2 = gp ** 2
    gs2 = gs ** 2

    def f(T3, Q, i, j):  # [eqn (4.11)]
        if i == j:
            Acoeff = -gw * gp / (gw2 - gp2) * W["phiWB"]
            Zcoeff = (
                W["ll_1221"] / 4
                - W["phil3_11"].real / 2
                - W["phil3_22"].real / 2
                - W["phiD"] / 4
            )
            return Acoeff * Q + Zcoeff * (T3 + Q * gp2 / (gw2 - gp2))
        else:
            return 0

    # rotate Cphiq3 + Cphiq1 to down basis
    Cphiq1 = np.zeros((3, 3), complex)
    Cphiq3 = np.zeros((3, 3), complex)
    for i in range(3):
        for j in range(3):
            if j >= i:
                ind = "_{}{}".format(i + 1, j + 1)
                Cphiq1[i, j] = W["phiq1" + ind]
                Cphiq3[i, j] = W["phiq3" + ind]
    # symmetrize
    Cphiq1 = Cphiq1 + Cphiq1.T.conjugate() - np.diag(np.diag(Cphiq1))
    Cphiq3 = Cphiq3 + Cphiq3.T.conjugate() - np.diag(np.diag(Cphiq3))
    Cphiq13rot = Vdag @ (Cphiq3 + Cphiq1) @ V

    # W/Z chiral coupling deviations
    for i in range(3):
        for j in range(3):
            ind = "_{}{}".format(i + 1, j + 1)
            if j >= i:
                H["deltagWlL" + ind] = (
                    W["phil3" + ind] + f(1 / 2, 0, i, j) - f(-1 / 2, -1, i, j)
                )

                H["deltagZeL" + ind] = (
                    -1 / 2 * W["phil3" + ind]
                    - 1 / 2 * W["phil1" + ind]
                    + f(-1 / 2, -1, i, j)
                )

                H["deltagZeR" + ind] = -1 / 2 * W["phie" + ind] + f(0, -1, i, j)

                H["deltagZuL" + ind] = (
                    1 / 2 * W["phiq3" + ind]
                    - 1 / 2 * W["phiq1" + ind]
                    + f(1 / 2, 2 / 3, i, j)
                )

                H["deltagZdL" + ind] = -1 / 2 * Cphiq13rot[i, j] + f(
                    -1 / 2, -1 / 3, i, j
                )

                H["deltagZuR" + ind] = -1 / 2 * W["phiu" + ind] + f(0, 2 / 3, i, j)

                H["deltagZdR" + ind] = -1 / 2 * W["phid" + ind] + f(0, -1 / 3, i, j)

            H["deltagWqR" + ind] = -W["phiud" + ind] / 2

    H["deltalambda3"] = (
        lam
        * (
            3 * W["phiBox"]
            - 3 / 4 * W["phiD"]
            + 1 / 4 * W["ll_1221"]
            - W["phil3_11"] / 2
            - W["phil3_22"] / 2
        )
        - W["phi"]
    )

    H["deltacz"] = (
        W["phiBox"]
        - W["phiD"] / 4
        + 3 / 4 * W["ll_1221"]
        - 3 / 2 * W["phil3_11"]
        - 3 / 2 * W["phil3_22"]
    )

    # Two derivative field strength interactions
    H["czBox"] = (
        -W["ll_1221"] / 2 + W["phiD"] / 2 + W["phil3_11"] + W["phil3_22"]
    ) / gw2

    H["cgg"] = (4 / gs2) * W["phiG"]

    H["cgammagamma"] = 4 * (W["phiW"] / gw2 + W["phiB"] / gp2 - W["phiWB"] / gw / gp)

    H["czz"] = (
        4
        * (gw2 * W["phiW"] + gp2 * W["phiB"] + gw * gp * W["phiWB"])
        / (gw2 + gp2) ** 2
    )

    H["czgamma"] = (
        4 * W["phiW"] - 4 * W["phiB"] - 2 * (gw2 - gp2) / (gw * gp) * W["phiWB"]
    ) / (gw2 + gp2)

    H["cggtilde"] = (4 / gs2) * W["phiGtilde"]

    H["cgammagammatilde"] = 4 * (
        W["phiWtilde"] / gw2 + W["phiBtilde"] / gp2 - W["phiWtildeB"] / gw / gp
    )

    H["czztilde"] = (
        4
        * (gw2 * W["phiWtilde"] + gp2 * W["phiBtilde"] + gw * gp * W["phiWtildeB"])
        / (gw2 + gp2) ** 2
    )

    H["czgammatilde"] = (
        4 * W["phiWtilde"]
        - 4 * W["phiBtilde"]
        - 2 * (gw2 - gp2) / (gw * gp) * W["phiWtildeB"]
    ) / (gw2 + gp2)

    basis = wcxf.Basis["SMEFT", "Higgs-Warsaw up"]
    all_wcs = set(basis.all_wcs)  # to speed up lookup
    return {k: v for k, v in H.items() if k in all_wcs}
