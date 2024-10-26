"""Definitions of auxiliary objects and operator properties."""
import numpy as np
from wilson.util import smeftutil
from wilson.util import nusmeftutil


def flavor_rotation_smeft(C_in, Uq, Uu, Ud, Ul, Ue, sm_parameters=True):
    """Gauge-invariant $U(3)^5$ flavor rotation of all Wilson coefficients and
    SM parameters."""
    C = {}
    if sm_parameters:
        # nothing to do for scalar SM parameters
        for k in ['g', 'gp', 'gs', 'Lambda', 'm2']:
            C[k] = C_in[k]
        C['Ge'] = Ul.conj().T @ C_in['Ge'] @ Ue
        C['Gu'] = Uq.conj().T @ C_in['Gu'] @ Uu
        C['Gd'] = Uq.conj().T @ C_in['Gd'] @ Ud
    # nothing to do for purely bosonic operators
    for k in smeftutil.WC_keys_0f:
        C[k] = C_in[k]
    # see 1704.03888 table 4 (but staying SU(2) invariant here)
    # LR
    for k in ['ephi', 'eW', 'eB']:
        C[k] = Ul.conj().T @ C_in[k] @ Ue
    for k in ['uphi', 'uW', 'uB', 'uG']:
        C[k] = Uq.conj().T @ C_in[k] @ Uu
    for k in ['dphi', 'dW', 'dB', 'dG']:
        C[k] = Uq.conj().T @ C_in[k] @ Ud
    # LL
    for k in ['phil1', 'phil3']:
        C[k] = Ul.conj().T @ C_in[k] @ Ul
    for k in ['phiq1', 'phiq3']:
        C[k] = Uq.conj().T @ C_in[k] @ Uq
    C['llphiphi'] = Ul.T @ C_in['llphiphi'] @ Ul
    # RR
    C['phie'] = Ue.conj().T @ C_in['phie'] @ Ue
    C['phiu'] = Uu.conj().T @ C_in['phiu'] @ Uu
    C['phid'] = Ud.conj().T @ C_in['phid'] @ Ud
    C['phiud'] = Uu.conj().T @ C_in['phiud'] @ Ud
    # 4-fermion
    C['ll'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ul, Ul, Ul.conj(), Ul.conj(), C_in['ll'])
    C['ee'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ue, Ue, Ue.conj(), Ue.conj(), C_in['ee'])
    C['le'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ul, Ue, Ul.conj(), Ue.conj(), C_in['le'])
    C['qq1'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Uq, Uq.conj(), Uq.conj(), C_in['qq1'])
    C['qq3'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Uq, Uq.conj(), Uq.conj(), C_in['qq3'])
    C['dd'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ud, Ud, Ud.conj(), Ud.conj(), C_in['dd'])
    C['uu'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uu, Uu, Uu.conj(), Uu.conj(), C_in['uu'])
    C['ud1'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uu, Ud, Uu.conj(), Ud.conj(), C_in['ud1'])
    C['ud8'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uu, Ud, Uu.conj(), Ud.conj(), C_in['ud8'])
    C['qu1'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Uu, Uq.conj(), Uu.conj(), C_in['qu1'])
    C['qu8'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Uu, Uq.conj(), Uu.conj(), C_in['qu8'])
    C['qd1'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Ud, Uq.conj(), Ud.conj(), C_in['qd1'])
    C['qd8'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Ud, Uq.conj(), Ud.conj(), C_in['qd8'])
    C['quqd1'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uu, Ud, Uq.conj(), Uq.conj(), C_in['quqd1'])
    C['quqd8'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uu, Ud, Uq.conj(), Uq.conj(), C_in['quqd8'])
    C['lq1'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ul, Uq, Ul.conj(), Uq.conj(), C_in['lq1'])
    C['lq3'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ul, Uq, Ul.conj(), Uq.conj(), C_in['lq3'])
    C['ld'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ul, Ud, Ul.conj(), Ud.conj(), C_in['ld'])
    C['lu'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ul, Uu, Ul.conj(), Uu.conj(), C_in['lu'])
    C['qe'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Ue, Uq.conj(), Ue.conj(), C_in['qe'])
    C['ed'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ue, Ud, Ue.conj(), Ud.conj(), C_in['ed'])
    C['eu'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ue, Uu, Ue.conj(), Uu.conj(), C_in['eu'])
    C['ledq'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ue, Uq, Ul.conj(), Ud.conj(), C_in['ledq'])
    C['lequ1'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ue, Uu, Ul.conj(), Uq.conj(), C_in['lequ1'])
    C['lequ3'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ue, Uu, Ul.conj(), Uq.conj(), C_in['lequ3'])
    C['duql'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uu, Ul, Ud, Uq, C_in['duql'])
    C['qque'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Ue, Uq, Uu, C_in['qque'])
    C['qqql'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Ul, Uq, Uq, C_in['qqql'])
    C['duue'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uu, Ue, Ud, Uu, C_in['duue'])
    return C

#assumption - Un is unitary matrix (mass and flavour basis are same).
def flavor_rotation_nusmeft(C_in, Uq, Uu, Ud, Ul, Ue, Un, sm_parameters=True):
    """Gauge-invariant $U(3)^6$ flavor rotation of all Wilson coefficients and
    SM parameters."""
    C = {}

    # smeft
    C = flavor_rotation_smeft(C_in, Uq, Uu, Ud, Ul, Ue, sm_parameters)

    if sm_parameters:
        C['Gn'] = Ul.conj().T @ C_in['Gn'] @ Un #TODO

    # 2-fermion nusmeft
    for k in ['nphi', 'nW', 'nB']: # new terms added 20 oct
        C[k] = Ul.conj().T @ C_in[k] @ Un

    C['phin']  = Un.conj().T @ C_in['phin'] @ Un #
    C['phine'] = Un.conj().T @ C_in['phine'] @ Ue #

    # 4-fermion nusmeft
    C['nd'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Un, Ud, Un.conj(), Ud.conj(), C_in['nd']) #
    C['nu'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Un, Uu, Un.conj(), Uu.conj(), C_in['nu']) #
    C['ln'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ul, Un, Ul.conj(), Un.conj(), C_in['ln']) #
    C['qn'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Un, Uq.conj(), Un.conj(), C_in['qn'])#
    C['ne'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Un, Ue, Un.conj(), Ue.conj(), C_in['ne']) #
    C['nn'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Un, Un, Un.conj(), Un.conj(), C_in['nn']) #
    C['nedu']  = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ue, Uu, Un.conj(), Ud.conj(), C_in['nedu']) #
    C['lnle']  = np.einsum('jb,ld,ia,kc,ijkl->abcd', Un, Ue, Ul.conj(), Ul.conj(), C_in['lnle'])#
    C['lnqd1'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Un, Ud, Ul.conj(), Uq.conj(), C_in['lnqd1'])#
    C['lnqd3'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Un, Ud, Ul.conj(), Uq.conj(), C_in['lnqd3'])#
    C['lnuq']  = np.einsum('jb,ld,ia,kc,ijkl->abcd', Un, Uq, Ul.conj(), Uu.conj(), C_in['lnuq'])#
    return C






