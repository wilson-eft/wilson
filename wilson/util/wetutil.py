"""Utility functions and dictionaries useful for the manipulation of WET Wilson coefficients.
"""

import wcxf
import numpy as np
import ckmutil
from wilson.util.smeftutil import _d_4, _d_6, _d_7


# names of Wilson coefficients with the same fermionic symmetry properties
# numbering is inspired by the corresponding categorization in SMEFT
C_symm_keys = {}
# 0 0F scalar object
C_symm_keys[0] = ["G", "Gtilde"]
# 1 2F general 3x3 matrix
C_symm_keys[1] = ["egamma", "uG","dG", "ugamma", "dgamma"]
# 3 4F general 3x3x3x3 object
C_symm_keys[3] = ['S1udRR', 'S1udduRR', 'S8udRR', 'S8udduRR', 'SedRL',
'SedRR', 'SeuRL', 'SeuRR', 'SnueduRL', 'SnueduRR', 'TedRR', 'TeuRR',
'TnueduRR', 'V1udduLR', 'V8udduLR', 'VnueduLL', 'VnueduLR',
'SuddLL', 'SduuLL', 'SduuLR', 'SduuRL', 'SdudRL', 'SduuRR',]
# 4 4F two identical ffbar currents
# hermitian currents
C_symm_keys[4] = ['VuuRR', 'VddRR', 'VuuLL', 'VddLL']
# non-hermitian currents
C_symm_keys[41] = ['S1ddRR', 'S1uuRR', 'S8uuRR', 'SeeRR', 'S8ddRR']
# 5 4F two independent ffbar currents, hermitian
C_symm_keys[5] = ["VnueLL", "VnuuLL", "VnudLL", "VeuLL", "VedLL", "V1udLL",
        "V8udLL", "VeuRR", "VedRR", "V1udRR", "V8udRR", "VnueLR",
        "VeeLR", "VnuuLR", "VnudLR", "VeuLR", "VedLR", "VueLR", "VdeLR",
        "V1uuLR", "V8uuLR", "V1udLR", "V8udLR", "V1duLR", "V8duLR",
        "V1ddLR", "V8ddLR"]
# 6 4F two identical ffbar currents + Fierz symmetry
C_symm_keys[6] = ['VeeLL', 'VeeRR', 'VnunuLL']
# 4F antisymmetric in first 2 indices
C_symm_keys[9] = ['SuudLR', 'SuudRL', 'SdduRL']


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

def _antisymm_12(C):
    """To get rid of NaNs produced by _scalar2array, antisymmetrize the first
    two indices of operators where C_ijkl = -C_jikl"""
    nans = np.isnan(C)
    C[nans] = -np.einsum('jikl', C)[nans]
    return C


def JMS_to_array(C, sectors=None):
    """For a dictionary with JMS Wilson coefficients, return a dictionary
    of arrays."""
    if sectors is None:
        wc_keys = wcxf.Basis['WET', 'JMS'].all_wcs
    else:
        try:
            wc_keys = [k for s in sectors for k in wcxf.Basis['WET', 'JMS'].sectors[s]]
        except KeyError:
            print(sectors)
    # fill in zeros for missing coefficients
    C_complete = {k: C.get(k, 0) for k in wc_keys}
    Ca = _scalar2array(C_complete)
    for k in Ca:
        if k in C_symm_keys[5]:
            Ca[k] = _symm_herm(Ca[k])
        if k in C_symm_keys[41]:
            Ca[k] = _symm_current(Ca[k])
        if k in C_symm_keys[4]:
            Ca[k] = _symm_herm(_symm_current(Ca[k]))
        if k in C_symm_keys[9]:
            Ca[k] = _antisymm_12(Ca[k])
    return Ca


def symmetrize_JMS_dict(C):
    """For a dictionary with JMS Wilson coefficients but keys that might not be
    in the non-redundant basis, return a dictionary with keys from the basis
    and values conjugated if necessary."""
    wc_keys = set(wcxf.Basis['WET', 'JMS'].all_wcs)
    Cs = {}
    for op, v in C.items():
        if '_' not in op or op in wc_keys:
            Cs[op] = v
            continue
        name, ind = op.split('_')
        if name in C_symm_keys[5]:
            i, j, k, l = ind
            indnew = ''.join([j, i, l, k])
            Cs['_'.join([name, indnew])] = v.conjugate()
        elif name in C_symm_keys[41]:
            i, j, k, l = ind
            indnew = ''.join([k, l, i, j])
            Cs['_'.join([name, indnew])] = v
        elif name in C_symm_keys[4]:
            i, j, k, l = ind
            indnew = ''.join([l, k, j, i])
            newname = '_'.join([name, indnew])
            if newname in wc_keys:
                Cs[newname] = v.conjugate()
            else:
                indnew = ''.join([j, i, l, k])
                newname = '_'.join([name, indnew])
                if newname in wc_keys:
                    Cs[newname] = v.conjugate()
                else:
                    indnew = ''.join([k, l, i, j])
                    newname = '_'.join([name, indnew])
                    Cs[newname] = v
        elif name in C_symm_keys[9]:
            i, j, k, l = ind
            indnew = ''.join([j, i, k, l])
            Cs['_'.join([name, indnew])] = -v
    return Cs


def rotate_down(C_in, p):
    """Redefinition of all Wilson coefficients in the JMS basis when rotating
    down-type quark fields from the flavour to the mass basis.

    C_in is expected to be an array-valued dictionary containg a key
    for all Wilson coefficient matrices."""
    C = C_in.copy()
    V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["delta"])
    UdL = V
    ## B conserving operators
    # type dL dR (dipoles)
    for k in ['dgamma', 'dG']:
        C[k] = np.einsum('ia,ij->aj',
                         UdL.conj(),
                         C_in[k])
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
    for k in ['S1ddRR', 'S8ddRR']:
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
    ## DeltaB=DeltaL=1 operators
    # type dL X X X
    for k in ['SduuLL',  'SduuLR']:
        C[k] = np.einsum('ia,ijkl->ajkl',
                         UdL,
                         C_in[k])
    # type X X dL X
    for k in ['SuudRL', 'SdudRL']:
        C[k] = np.einsum('kc,ijkl->ijcl',
                         UdL,
                         C_in[k])
    # type X dL dL X
    for k in ['SuddLL']:
        C[k] = np.einsum('jb,kc,ijkl->ibcl',
                         UdL, UdL,
                         C_in[k])
    return C


_scale_dict = {}
for k in C_symm_keys[0]:
    _scale_dict[k] = 1
for k in C_symm_keys[1]:
    _scale_dict[k] = np.ones((3, 3))
for k in C_symm_keys[3] + C_symm_keys[5]:
    _scale_dict[k] = np.ones((3, 3, 3, 3))
for k in C_symm_keys[4] + C_symm_keys[41]:
    _scale_dict[k] = _d_4
for k in C_symm_keys[6]:
    _scale_dict[k] = _d_6
for k in C_symm_keys[9]:
    # while _d_7 contains the symmetry factors for the case of coefficients
    # *symmetric* under the 1st 2 indices, they are actually the same as for
    # the case where they are *antisymmetric*
    _scale_dict[k] = _d_7


def scale_dict_wet(C):
    """To account for the fact that arXiv:Jenkins:2017jig uses a flavour
    non-redundant basis in contrast to WCxf, symmetry factors of two have to
    be introduced in several places for operators that are symmetric
    under the interchange of two currents."""
    return {k: v / _scale_dict[k] for k, v in C.items()}


def unscale_dict_wet(C):
    """Undo the scaling applied in `scale_dict_wet`."""
    return {k: _scale_dict[k] * v for k, v in C.items()}
