"""Utility functions and dictionaries useful for the manipulation of WET Wilson coefficients.
"""

import wcxf
import numpy as np
import ckmutil


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


def JMS_to_array(C):
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
        if name in ["VnueLL", "VnuuLL", "VnudLL", "VeuLL", "VedLL", "V1udLL",
                "V8udLL", "VeuRR", "VedRR", "V1udRR", "V8udRR", "VnueLR",
                "VeeLR", "VnuuLR", "VnudLR", "VeuLR", "VedLR", "VueLR", "VdeLR",
                "V1uuLR", "V8uuLR", "V1udLR", "V8udLR", "V1duLR", "V8duLR",
                "V1ddLR", "V8ddLR"]:
            i, j, k, l = ind
            indnew = ''.join([j, i, l, k])
            Cs['_'.join([name, indnew])] = v.conjugate()
        elif name in ["S1uuRR", "S8uuRR", "S1ddRR", "S8ddRR"]:
            i, j, k, l = ind
            indnew = ''.join([k, l, i, j])
            Cs['_'.join([name, indnew])] = v
        elif name in ["VuuLL", "VddLL", "VuuRR", "VddRR"]:
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
    return Cs


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
