"""Utility functions and dictionaries useful for the manipulation of SMEFT Wilson coefficients.
"""

import numpy as np
import wilson


# names and shape of SM parameters
dim4_keys_shape = {
   'g': 1,
   'gp': 1,
   'gs': 1,
   'Lambda': 1,
   'm2': 1,
   'Gu': (3, 3),
   'Gd': (3, 3),
   'Ge': (3, 3),
}


# names of Wilson coefficients with the same fermionic symmetry properties
C_symm_keys = {}
# 0 0F scalar object
C_symm_keys[0] = ['G', 'Gtilde', 'W', 'Wtilde', 'phi', 'phiBox', 'phiD', 'phiG',
                  'phiB', 'phiW', 'phiWB', 'phiGtilde', 'phiBtilde',
                  'phiWtilde', 'phiWtildeB'] + ['g', 'gp', 'gs', 'Lambda', 'm2']
# 1 2F general 3x3 matrix
C_symm_keys[1] = ["uphi", "dphi", "ephi", "eW", "eB", "uG", "uW", "uB", "dG",
                  "dW", "dB", "phiud"] + ['Gu', 'Gd', 'Ge']
# 2 2F Hermitian matrix
C_symm_keys[2] = ["phil1", "phil3", "phie", "phiq1", "phiq3", "phiu", "phid",]
# 3 4F general 3x3x3x3 object
C_symm_keys[3] = ["ledq", "quqd1", "quqd8", "lequ1", "lequ3", "duql", "duue"]
# 4 4F two identical ffbar currents
# hermitian currents
C_symm_keys[4] = ["ll", "qq1", "qq3", "uu", "dd",]
# 5 4F two independent ffbar currents
C_symm_keys[5] = ["lq1", "lq3", "eu", "ed", "ud1", "ud8", "le", "lu", "ld",
                  "qe", "qu1", "qd1", "qu8", "qd8",]
# 6 4F two identical ffbar currents - special case Cee
C_symm_keys[6] = ["ee",]
# 7 4F Baryon-number-violating - special case Cqque
C_symm_keys[7] = ["qque",]
# 8 4F Baryon-number-violating - special case Cqqql
C_symm_keys[8] = ["qqql",]
# 9 2F symmetric matrix
C_symm_keys[9] = ["llphiphi"]


def flavor_rotation(C_in, Uq, Uu, Ud, Ul, Ue):
    """Gauge-invariant $U(3)^5$ flavor rotation of all Wilson coefficients."""
    C = {}
    # nothing to do for purely bosonic operators
    for k in wilson.util.smeftutil.WC_keys_0f:
        if k in C_in:
            C[k] = C_in[k]
    # see 1704.03888 table 4 (but staying SU(2) invariant here)
    # LR
    for k in ['ephi', 'eW', 'eB']:
        if k in C_in:
            C[k] = Ul.conj().T @ C_in[k] @ Ue
    for k in ['uphi', 'uW', 'uB', 'uG']:
        if k in C_in:
            C[k] = Uq.conj().T @ C_in[k] @ Uu
    for k in ['dphi', 'dW', 'dB', 'dG']:
        if k in C_in:
            C[k] = Uq.conj().T @ C_in[k] @ Ud
    # LL
    for k in ['phil1', 'phil3']:
        if k in C_in:
            C[k] = Ul.conj().T @ C_in[k] @ Ul
    for k in ['phiq1', 'phiq3']:
        if k in C_in:
            C[k] = Uq.conj().T @ C_in[k] @ Uq
    if 'llphiphi' in C_in:
        C['llphiphi'] = Ul.T @ C_in['llphiphi'] @ Ul
    # RR
    if 'phie' in C_in:
        C['phie'] = Ue.conj().T @ C_in['phie'] @ Ue
    if 'phiu' in C_in:
        C['phiu'] = Uu.conj().T @ C_in['phiu'] @ Uu
    if 'phid' in C_in:
        C['phid'] = Ud.conj().T @ C_in['phid'] @ Ud
    if 'phiud' in C_in:
        C['phiud'] = Uu.conj().T @ C_in['phiud'] @ Ud
    # 4-fermion
    if 'll' in C_in:
        C['ll'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ul, Ul, Ul.conj(), Ul.conj(), C_in['ll'])
    if 'ee' in C_in:
        C['ee'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ue, Ue, Ue.conj(), Ue.conj(), C_in['ee'])
    if 'le' in C_in:
        C['le'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ul, Ue, Ul.conj(), Ue.conj(), C_in['le'])
    if 'qq1' in C_in:
        C['qq1'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Uq, Uq.conj(), Uq.conj(), C_in['qq1'])
    if 'qq3' in C_in:
        C['qq3'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Uq, Uq.conj(), Uq.conj(), C_in['qq3'])
    if 'dd' in C_in:
        C['dd'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ud, Ud, Ud.conj(), Ud.conj(), C_in['dd'])
    if 'uu' in C_in:
        C['uu'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uu, Uu, Uu.conj(), Uu.conj(), C_in['uu'])
    if 'ud8' in C_in:
        C['ud8'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uu, Ud, Uu.conj(), Ud.conj(), C_in['ud8'])
    if 'ud1' in C_in:
        C['ud1'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uu, Ud, Uu.conj(), Ud.conj(), C_in['ud1'])
    if 'qu1' in C_in:
        C['qu1'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Uu, Uq.conj(), Uu.conj(), C_in['qu1'])
    if 'qu8' in C_in:
        C['qu8'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Uu, Uq.conj(), Uu.conj(), C_in['qu8'])
    if 'qd1' in C_in:
        C['qd1'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Ud, Uq.conj(), Ud.conj(), C_in['qd1'])
    if 'qd8' in C_in:
        C['qd8'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Ud, Uq.conj(), Ud.conj(), C_in['qd8'])
    if 'quqd1' in C_in:
        C['quqd1'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uu, Ud, Uq.conj(), Uq.conj(), C_in['quqd1'])
    if 'quqd8' in C_in:
        C['quqd8'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uu, Ud, Uq.conj(), Uq.conj(), C_in['quqd8'])
    if 'lq1' in C_in:
        C['lq1'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ul, Uq, Ul.conj(), Uq.conj(), C_in['lq1'])
    if 'lq3' in C_in:
        C['lq3'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ul, Uq, Ul.conj(), Uq.conj(), C_in['lq3'])
    if 'ld' in C_in:
        C['ld'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ul, Ud, Ul.conj(), Ud.conj(), C_in['ld'])
    if 'lu' in C_in:
        C['lu'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ul, Uu, Ul.conj(), Uu.conj(), C_in['lu'])
    if 'qe' in C_in:
        C['qe'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Ue, Uq.conj(), Ue.conj(), C_in['qe'])
    if 'ed' in C_in:
        C['ed'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ue, Ud, Ue.conj(), Ud.conj(), C_in['ed'])
    if 'eu' in C_in:
        C['eu'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ue, Uu, Ue.conj(), Uu.conj(), C_in['eu'])
    if 'ledq' in C_in:
        C['ledq'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ue, Uq, Ul.conj(), Ud.conj(), C_in['ledq'])
    if 'lequ1' in C_in:
        C['lequ1'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ue, Uu, Ul.conj(), Uq.conj(), C_in['lequ1'])
    if 'lequ3' in C_in:
        C['lequ3'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Ue, Uu, Ul.conj(), Uq.conj(), C_in['lequ3'])
    if 'duql' in C_in:
        C['duql'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uu, Ul, Ud, Uq, C_in['duql'])
    if 'qque' in C_in:
        C['qque'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Ue, Uq, Uu, C_in['qque'])
    if 'qqql' in C_in:
        C['qqql'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uq, Ul, Uq, Uq, C_in['qqql'])
    if 'duue' in C_in:
        C['duue'] = np.einsum('jb,ld,ia,kc,ijkl->abcd', Uu, Ue, Ud, Uu, C_in['duue'])
    return C
