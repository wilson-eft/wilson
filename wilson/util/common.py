import numpy as np
from collections import OrderedDict
from itertools import chain
from functools import reduce
import operator
from wilson import wcxf


class EFTutil:
    """Utility class useful for the manipulation of EFT Wilson coefficients.
    """
    _symmetry_class_defintions = {
        #The keys are sorted n-tuples of 2-tuples in which the first entry is
        #a string specifying the indices of a Wilson coefficient in the non-
        #redundant two-flavour basis and the second entry is True (False) if the
        #Wilson coefficient is real (complex). The values are integer ids
        #denoting the symmetry class.
        (): 0, #0F scalar object
        (
            ('11', False), ('12', False), ('21', False), ('22', False),
        ): 1, #2F general 3x3 matrix
        (
            ('11', True), ('12', False), ('22', True),
        ): 2, # 2F Hermitian matrix
        (
            ('11', False), ('12', False), ('22', False),
        ): 9, # 2F symmetric matrix
        (
            ('1111', True), ('1112', False), ('1122', True), ('1212', False),
            ('1221', True), ('1222', False), ('2222', True),
        ): 4, # 4F two identical ffbar currents, hermitian
        (
            ('1111', False), ('1112', False), ('1121', False), ('1122', False),
            ('1212', False), ('1221', False), ('1222', False), ('2121', False),
            ('2122', False), ('2222', False),
        ): 41, # 4F two identical ffbar currents, non-hermitian
        (
            ('1111', True), ('1112', False), ('1122', True), ('1211', False),
            ('1212', False), ('1221', False), ('1222', False), ('2211', True),
            ('2212', False), ('2222', True),
        ): 5, # 4F two independent ffbar currents
        (
            ('1111', True), ('1112', False), ('1122', True), ('1212', False),
            ('1222', False), ('2222', True),
        ): 6, # 4F two identical ffbar currents
        (
            ('1111', False), ('1112', False), ('1121', False), ('1122', False),
            ('1211', False), ('1212', False), ('1221', False), ('1222', False),
            ('2111', False), ('2112', False), ('2121', False), ('2122', False),
            ('2211', False), ('2212', False), ('2221', False), ('2222', False),
        ): 3, # 4F general 3x3x3x3 object
        (
            ('1111', False), ('1112', False), ('1121', False), ('1122', False),
            ('1211', False), ('1212', False), ('1221', False), ('1222', False),
            ('2211', False), ('2212', False), ('2221', False), ('2222', False),
        ): 7, # 4F symmetric in first two indices
        (
            ('1211', False), ('1212', False), ('1221', False), ('1222', False),
        ): 71, # 4F antisymmetric in first two indices
        (
            ('1111', False), ('1112', False), ('1121', False), ('1122', False),
            ('1211', False), ('1212', False), ('1221', False), ('1222', False),
            ('2121', False), ('2122', False), ('2221', False), ('2222', False),
        ): 8, # 4F Baryon-number-violating - special case Cqqql
    }

    def __init__(self, eft, basis, dim4_keys_shape, dim4_symm_keys):
        self._eft = eft
        self._basis = basis
        self._dim4_keys_shape = dim4_keys_shape
        self._dim4_symm_keys = dim4_symm_keys
        keys_and_shapes = self._get_keys_and_shapes()
        self.WC_keys_0f = keys_and_shapes['WC_keys_0f']
        self.WC_keys_2f = keys_and_shapes['WC_keys_2f']
        self.WC_keys_4f = keys_and_shapes['WC_keys_4f']
        self.WC_keys = keys_and_shapes['WC_keys']
        self.C_keys_shape = keys_and_shapes['C_keys_shape']
        self.C_keys = keys_and_shapes['C_keys']
        self.dim4_keys = keys_and_shapes['dim4_keys']
        self.C_symm_keys = self._get_symm_keys()
        self._scale_dict, self._d_4, self._d_6, self._d_7 = self._get_scale_dict()

    def _get_keys_and_shapes(self):
        all_wcs = wcxf.Basis[self._eft, self._basis].all_wcs
        WC_keys_0f = list(dict.fromkeys(v for v in all_wcs if '_' not in v))
        WC_keys_2f = list(dict.fromkeys(
            v.split("_")[0] for v in all_wcs
            if '_' in v and len(v.split("_")[1]) == 2
        ))
        WC_keys_4f = list(dict.fromkeys(
            v.split("_")[0] for v in all_wcs
            if '_' in v and len(v.split("_")[1]) == 4
        ))
        WC_keys = WC_keys_0f + WC_keys_2f + WC_keys_4f
        index_dict = {k:[] for k in WC_keys}
        for v in all_wcs:
            v_split = v.split('_')
            if len(v_split) == 2:
                index_dict[v_split[0]].append(v_split[1])
        WC_keys_shape = {
            k: tuple(np.max(
                [[int(i) for i in index_list] for index_list in v],
                axis=0
            )) if v else 1
            for k,v in index_dict.items()
        }
        dim4_keys_shape = self._dim4_keys_shape
        C_keys_shape = {**dim4_keys_shape, **WC_keys_shape}
        C_keys = list(C_keys_shape.keys())
        return {
            'WC_keys_0f': WC_keys_0f,
            'WC_keys_2f': WC_keys_2f,
            'WC_keys_4f': WC_keys_4f,
            'WC_keys': WC_keys,
            'dim4_keys': list(dim4_keys_shape.keys()),
            'C_keys': C_keys,
            'C_keys_shape': C_keys_shape,
        }

    def _get_symm_keys(self):
        sectors = wcxf.Basis[self._eft, self._basis].sectors
        all_wcs = wcxf.Basis[self._eft, self._basis].all_wcs
        C_keys_complex = dict(chain.from_iterable(
            ((k2,v2.get('real', False)) for k2,v2 in v1.items())
            for k1,v1 in sectors.items()
        ))
        index_complex_dict = {k:[] for k in self.WC_keys}
        for v in all_wcs:
            v_split = v.split('_')
            if len(v_split) == 2 and '3' not in v_split[1]:
                index_complex_dict[v_split[0]].append(
                    (v_split[1], C_keys_complex[v])
                )
        C_symm_keys = {}
        for k,v in index_complex_dict.items():
            key = self._symmetry_class_defintions[tuple(sorted(set(v)))]
            if key in C_symm_keys:
                C_symm_keys[key].append(k)
            else:
                C_symm_keys[key] = [k]
        for k,v in self._dim4_symm_keys.items():
            C_symm_keys[k] += v
        return C_symm_keys

    def _get_scale_dict(self):
        # computing the scale vector required for symmetrize_nonred
        # initialize with factor 1
        d_4 = np.zeros((3,3,3,3))
        d_6 = np.zeros((3,3,3,3))
        d_7 = np.zeros((3,3,3,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        # class 4: symmetric under interachange of currents
                        d_4[i, j, k, l] = len({(i, j, k, l), (k, l, i, j)})
                        # class 6: symmetric under interachange of currents + Fierz
                        d_6[i, j, k, l] = len({(i, j, k, l), (k, l, i, j), (k, j, i, l), (i, l, k, j)})
                        # class 7: symmetric under interachange of first two indices
                        d_7[i, j, k, l] = len({(i, j, k, l), (j, i, k, l)})
        scale_dict = self.C_array2dict(np.ones(9999))
        for k in self.C_symm_keys.get(4, []) + self.C_symm_keys.get(41, []):
            scale_dict[k] = d_4
        for k in self.C_symm_keys.get(6, ()):
            scale_dict[k] = d_6
        for k in self.C_symm_keys.get(7, []) + self.C_symm_keys.get(71, []):
            scale_dict[k] = d_7
        return scale_dict, d_4, d_6, d_7

    def C_array2dict(self, C):
        """Convert a 1D array containing C values to a dictionary."""
        d = OrderedDict()
        i=0
        for k in self.C_keys:
            s = self.C_keys_shape[k]
            if s == 1:
                j = i+1
                d[k] = C[i]
            else:
                j = i \
          + reduce(operator.mul, s, 1)
                d[k] = C[i:j].reshape(s)
            i = j
        return d

    def C_dict2array(self, C):
        """Convert an OrderedDict containing C values to a 1D array."""
        return np.hstack([np.asarray(C[k]).ravel() for k in self.C_keys])

    @staticmethod
    def arrays2wcxf(C):
        """Convert a dictionary with Wilson coefficient names as keys and
        numbers or numpy arrays as values to a dictionary with a Wilson coefficient
        name followed by underscore and numeric indices as keys and numbers as
        values. This is needed for the output in WCxf format."""
        d = {}
        for k, v in C.items():
            if np.shape(v) == () or np.shape(v) == (1,):
                d[k] = v
            else:
                ind = np.indices(v.shape).reshape(v.ndim, v.size).T
                for i in ind:
                    name = k + '_' + ''.join([str(int(j) + 1) for j in i])
                    d[name] = v[tuple(i)]
        return d

    def wcxf2arrays(self, d):
        """Convert a dictionary with a Wilson coefficient
        name followed by underscore and numeric indices as keys and numbers as
        values to a dictionary with Wilson coefficient names as keys and
        numbers or numpy arrays as values. This is needed for the parsing
        of input in WCxf format."""
        C = {}
        for k, v in d.items():
            name = k.split('_')[0]
            s = self.C_keys_shape[name]
            if s == 1:
                C[k] = v
            else:
                ind = k.split('_')[-1]
                if name not in C:
                    C[name] = np.zeros(s, dtype=complex)
                C[name][tuple([int(i) - 1 for i in ind])] = v
        return C

    def add_missing(self, C):
        """Add arrays with zeros for missing Wilson coefficient keys"""
        C_out = C.copy()
        for k in (set(self.WC_keys) - set(C.keys())):
            s = self.C_keys_shape[k]
            if s == 1:
                C_out[k] = 0
            else:
                C_out[k] = np.zeros(self.C_keys_shape[k])
        return C_out

    @staticmethod
    def symmetrize_2(b):
        a = np.array(b, copy=True, dtype=complex)
        a[1, 0] = a[0, 1].conj()
        a[2, 0] = a[0, 2].conj()
        a[2, 1] = a[1, 2].conj()
        a.imag[0, 0] = 0
        a.imag[1, 1] = 0
        a.imag[2, 2] = 0
        return a

    @staticmethod
    def symmetrize_4(b):
        a = np.array(b, copy=True, dtype=complex)
        a.real[0, 0, 1, 0] = a.real[0, 0, 0, 1]
        a.real[0, 0, 2, 0] = a.real[0, 0, 0, 2]
        a.real[0, 0, 2, 1] = a.real[0, 0, 1, 2]
        a.real[0, 1, 0, 0] = a.real[0, 0, 0, 1]
        a.real[0, 2, 0, 0] = a.real[0, 0, 0, 2]
        a.real[0, 2, 0, 1] = a.real[0, 1, 0, 2]
        a.real[0, 2, 1, 0] = a.real[0, 1, 2, 0]
        a.real[1, 0, 0, 0] = a.real[0, 0, 0, 1]
        a.real[1, 0, 0, 1] = a.real[0, 1, 1, 0]
        a.real[1, 0, 0, 2] = a.real[0, 1, 2, 0]
        a.real[1, 0, 1, 0] = a.real[0, 1, 0, 1]
        a.real[1, 0, 1, 1] = a.real[0, 1, 1, 1]
        a.real[1, 0, 1, 2] = a.real[0, 1, 2, 1]
        a.real[1, 0, 2, 0] = a.real[0, 1, 0, 2]
        a.real[1, 0, 2, 1] = a.real[0, 1, 1, 2]
        a.real[1, 0, 2, 2] = a.real[0, 1, 2, 2]
        a.real[1, 1, 0, 0] = a.real[0, 0, 1, 1]
        a.real[1, 1, 0, 1] = a.real[0, 1, 1, 1]
        a.real[1, 1, 0, 2] = a.real[0, 2, 1, 1]
        a.real[1, 1, 1, 0] = a.real[0, 1, 1, 1]
        a.real[1, 1, 2, 0] = a.real[0, 2, 1, 1]
        a.real[1, 1, 2, 1] = a.real[1, 1, 1, 2]
        a.real[1, 2, 0, 0] = a.real[0, 0, 1, 2]
        a.real[1, 2, 0, 1] = a.real[0, 1, 1, 2]
        a.real[1, 2, 0, 2] = a.real[0, 2, 1, 2]
        a.real[1, 2, 1, 0] = a.real[0, 1, 2, 1]
        a.real[1, 2, 1, 1] = a.real[1, 1, 1, 2]
        a.real[1, 2, 2, 0] = a.real[0, 2, 2, 1]
        a.real[2, 0, 0, 0] = a.real[0, 0, 0, 2]
        a.real[2, 0, 0, 1] = a.real[0, 1, 2, 0]
        a.real[2, 0, 0, 2] = a.real[0, 2, 2, 0]
        a.real[2, 0, 1, 0] = a.real[0, 1, 0, 2]
        a.real[2, 0, 1, 1] = a.real[0, 2, 1, 1]
        a.real[2, 0, 1, 2] = a.real[0, 2, 2, 1]
        a.real[2, 0, 2, 0] = a.real[0, 2, 0, 2]
        a.real[2, 0, 2, 1] = a.real[0, 2, 1, 2]
        a.real[2, 0, 2, 2] = a.real[0, 2, 2, 2]
        a.real[2, 1, 0, 0] = a.real[0, 0, 1, 2]
        a.real[2, 1, 0, 1] = a.real[0, 1, 2, 1]
        a.real[2, 1, 0, 2] = a.real[0, 2, 2, 1]
        a.real[2, 1, 1, 0] = a.real[0, 1, 1, 2]
        a.real[2, 1, 1, 1] = a.real[1, 1, 1, 2]
        a.real[2, 1, 1, 2] = a.real[1, 2, 2, 1]
        a.real[2, 1, 2, 0] = a.real[0, 2, 1, 2]
        a.real[2, 1, 2, 1] = a.real[1, 2, 1, 2]
        a.real[2, 1, 2, 2] = a.real[1, 2, 2, 2]
        a.real[2, 2, 0, 0] = a.real[0, 0, 2, 2]
        a.real[2, 2, 0, 1] = a.real[0, 1, 2, 2]
        a.real[2, 2, 0, 2] = a.real[0, 2, 2, 2]
        a.real[2, 2, 1, 0] = a.real[0, 1, 2, 2]
        a.real[2, 2, 1, 1] = a.real[1, 1, 2, 2]
        a.real[2, 2, 1, 2] = a.real[1, 2, 2, 2]
        a.real[2, 2, 2, 0] = a.real[0, 2, 2, 2]
        a.real[2, 2, 2, 1] = a.real[1, 2, 2, 2]
        a.imag[0, 0, 0, 0] = 0
        a.imag[0, 0, 1, 0] = -a.imag[0, 0, 0, 1]
        a.imag[0, 0, 1, 1] = 0
        a.imag[0, 0, 2, 0] = -a.imag[0, 0, 0, 2]
        a.imag[0, 0, 2, 1] = -a.imag[0, 0, 1, 2]
        a.imag[0, 0, 2, 2] = 0
        a.imag[0, 1, 0, 0] = a.imag[0, 0, 0, 1]
        a.imag[0, 1, 1, 0] = 0
        a.imag[0, 2, 0, 0] = a.imag[0, 0, 0, 2]
        a.imag[0, 2, 0, 1] = a.imag[0, 1, 0, 2]
        a.imag[0, 2, 1, 0] = -a.imag[0, 1, 2, 0]
        a.imag[0, 2, 2, 0] = 0
        a.imag[1, 0, 0, 0] = -a.imag[0, 0, 0, 1]
        a.imag[1, 0, 0, 1] = 0
        a.imag[1, 0, 0, 2] = -a.imag[0, 1, 2, 0]
        a.imag[1, 0, 1, 0] = -a.imag[0, 1, 0, 1]
        a.imag[1, 0, 1, 1] = -a.imag[0, 1, 1, 1]
        a.imag[1, 0, 1, 2] = -a.imag[0, 1, 2, 1]
        a.imag[1, 0, 2, 0] = -a.imag[0, 1, 0, 2]
        a.imag[1, 0, 2, 1] = -a.imag[0, 1, 1, 2]
        a.imag[1, 0, 2, 2] = -a.imag[0, 1, 2, 2]
        a.imag[1, 1, 0, 0] = 0
        a.imag[1, 1, 0, 1] = a.imag[0, 1, 1, 1]
        a.imag[1, 1, 0, 2] = a.imag[0, 2, 1, 1]
        a.imag[1, 1, 1, 0] = -a.imag[0, 1, 1, 1]
        a.imag[1, 1, 1, 1] = 0
        a.imag[1, 1, 2, 0] = -a.imag[0, 2, 1, 1]
        a.imag[1, 1, 2, 1] = -a.imag[1, 1, 1, 2]
        a.imag[1, 1, 2, 2] = 0
        a.imag[1, 2, 0, 0] = a.imag[0, 0, 1, 2]
        a.imag[1, 2, 0, 1] = a.imag[0, 1, 1, 2]
        a.imag[1, 2, 0, 2] = a.imag[0, 2, 1, 2]
        a.imag[1, 2, 1, 0] = -a.imag[0, 1, 2, 1]
        a.imag[1, 2, 1, 1] = a.imag[1, 1, 1, 2]
        a.imag[1, 2, 2, 0] = -a.imag[0, 2, 2, 1]
        a.imag[1, 2, 2, 1] = 0
        a.imag[2, 0, 0, 0] = -a.imag[0, 0, 0, 2]
        a.imag[2, 0, 0, 1] = a.imag[0, 1, 2, 0]
        a.imag[2, 0, 0, 2] = 0
        a.imag[2, 0, 1, 0] = -a.imag[0, 1, 0, 2]
        a.imag[2, 0, 1, 1] = -a.imag[0, 2, 1, 1]
        a.imag[2, 0, 1, 2] = -a.imag[0, 2, 2, 1]
        a.imag[2, 0, 2, 0] = -a.imag[0, 2, 0, 2]
        a.imag[2, 0, 2, 1] = -a.imag[0, 2, 1, 2]
        a.imag[2, 0, 2, 2] = -a.imag[0, 2, 2, 2]
        a.imag[2, 1, 0, 0] = -a.imag[0, 0, 1, 2]
        a.imag[2, 1, 0, 1] = a.imag[0, 1, 2, 1]
        a.imag[2, 1, 0, 2] = a.imag[0, 2, 2, 1]
        a.imag[2, 1, 1, 0] = -a.imag[0, 1, 1, 2]
        a.imag[2, 1, 1, 1] = -a.imag[1, 1, 1, 2]
        a.imag[2, 1, 1, 2] = 0
        a.imag[2, 1, 2, 0] = -a.imag[0, 2, 1, 2]
        a.imag[2, 1, 2, 1] = -a.imag[1, 2, 1, 2]
        a.imag[2, 1, 2, 2] = -a.imag[1, 2, 2, 2]
        a.imag[2, 2, 0, 0] = 0
        a.imag[2, 2, 0, 1] = a.imag[0, 1, 2, 2]
        a.imag[2, 2, 0, 2] = a.imag[0, 2, 2, 2]
        a.imag[2, 2, 1, 0] = -a.imag[0, 1, 2, 2]
        a.imag[2, 2, 1, 1] = 0
        a.imag[2, 2, 1, 2] = a.imag[1, 2, 2, 2]
        a.imag[2, 2, 2, 0] = -a.imag[0, 2, 2, 2]
        a.imag[2, 2, 2, 1] = -a.imag[1, 2, 2, 2]
        a.imag[2, 2, 2, 2] = 0
        return a

    @staticmethod
    def symmetrize_41(b):
        a = np.array(b, copy=True, dtype=complex)
        a[0,1,0,0] = a[0,0,0,1]
        a[0,2,0,0] = a[0,0,0,2]
        a[0,2,0,1] = a[0,1,0,2]
        a[1,0,0,0] = a[0,0,1,0]
        a[1,0,0,1] = a[0,1,1,0]
        a[1,0,0,2] = a[0,2,1,0]
        a[1,1,0,0] = a[0,0,1,1]
        a[1,1,0,1] = a[0,1,1,1]
        a[1,1,0,2] = a[0,2,1,1]
        a[1,1,1,0] = a[1,0,1,1]
        a[1,2,0,0] = a[0,0,1,2]
        a[1,2,0,1] = a[0,1,1,2]
        a[1,2,0,2] = a[0,2,1,2]
        a[1,2,1,0] = a[1,0,1,2]
        a[1,2,1,1] = a[1,1,1,2]
        a[2,0,0,0] = a[0,0,2,0]
        a[2,0,0,1] = a[0,1,2,0]
        a[2,0,0,2] = a[0,2,2,0]
        a[2,0,1,0] = a[1,0,2,0]
        a[2,0,1,1] = a[1,1,2,0]
        a[2,0,1,2] = a[1,2,2,0]
        a[2,1,0,0] = a[0,0,2,1]
        a[2,1,0,1] = a[0,1,2,1]
        a[2,1,0,2] = a[0,2,2,1]
        a[2,1,1,0] = a[1,0,2,1]
        a[2,1,1,1] = a[1,1,2,1]
        a[2,1,1,2] = a[1,2,2,1]
        a[2,1,2,0] = a[2,0,2,1]
        a[2,2,0,0] = a[0,0,2,2]
        a[2,2,0,1] = a[0,1,2,2]
        a[2,2,0,2] = a[0,2,2,2]
        a[2,2,1,0] = a[1,0,2,2]
        a[2,2,1,1] = a[1,1,2,2]
        a[2,2,1,2] = a[1,2,2,2]
        a[2,2,2,0] = a[2,0,2,2]
        a[2,2,2,1] = a[2,1,2,2]
        return a

    @staticmethod
    def symmetrize_5(b):
        a = np.array(b, copy=True, dtype=complex)
        a.real[0, 0, 1, 0] = a.real[0, 0, 0, 1]
        a.real[0, 0, 2, 0] = a.real[0, 0, 0, 2]
        a.real[0, 0, 2, 1] = a.real[0, 0, 1, 2]
        a.real[1, 0, 0, 0] = a.real[0, 1, 0, 0]
        a.real[1, 0, 0, 1] = a.real[0, 1, 1, 0]
        a.real[1, 0, 0, 2] = a.real[0, 1, 2, 0]
        a.real[1, 0, 1, 0] = a.real[0, 1, 0, 1]
        a.real[1, 0, 1, 1] = a.real[0, 1, 1, 1]
        a.real[1, 0, 1, 2] = a.real[0, 1, 2, 1]
        a.real[1, 0, 2, 0] = a.real[0, 1, 0, 2]
        a.real[1, 0, 2, 1] = a.real[0, 1, 1, 2]
        a.real[1, 0, 2, 2] = a.real[0, 1, 2, 2]
        a.real[1, 1, 1, 0] = a.real[1, 1, 0, 1]
        a.real[1, 1, 2, 0] = a.real[1, 1, 0, 2]
        a.real[1, 1, 2, 1] = a.real[1, 1, 1, 2]
        a.real[2, 0, 0, 0] = a.real[0, 2, 0, 0]
        a.real[2, 0, 0, 1] = a.real[0, 2, 1, 0]
        a.real[2, 0, 0, 2] = a.real[0, 2, 2, 0]
        a.real[2, 0, 1, 0] = a.real[0, 2, 0, 1]
        a.real[2, 0, 1, 1] = a.real[0, 2, 1, 1]
        a.real[2, 0, 1, 2] = a.real[0, 2, 2, 1]
        a.real[2, 0, 2, 0] = a.real[0, 2, 0, 2]
        a.real[2, 0, 2, 1] = a.real[0, 2, 1, 2]
        a.real[2, 0, 2, 2] = a.real[0, 2, 2, 2]
        a.real[2, 1, 0, 0] = a.real[1, 2, 0, 0]
        a.real[2, 1, 0, 1] = a.real[1, 2, 1, 0]
        a.real[2, 1, 0, 2] = a.real[1, 2, 2, 0]
        a.real[2, 1, 1, 0] = a.real[1, 2, 0, 1]
        a.real[2, 1, 1, 1] = a.real[1, 2, 1, 1]
        a.real[2, 1, 1, 2] = a.real[1, 2, 2, 1]
        a.real[2, 1, 2, 0] = a.real[1, 2, 0, 2]
        a.real[2, 1, 2, 1] = a.real[1, 2, 1, 2]
        a.real[2, 1, 2, 2] = a.real[1, 2, 2, 2]
        a.real[2, 2, 1, 0] = a.real[2, 2, 0, 1]
        a.real[2, 2, 2, 0] = a.real[2, 2, 0, 2]
        a.real[2, 2, 2, 1] = a.real[2, 2, 1, 2]
        a.imag[0, 0, 0, 0] = 0
        a.imag[0, 0, 1, 0] = -a.imag[0, 0, 0, 1]
        a.imag[0, 0, 1, 1] = 0
        a.imag[0, 0, 2, 0] = -a.imag[0, 0, 0, 2]
        a.imag[0, 0, 2, 1] = -a.imag[0, 0, 1, 2]
        a.imag[0, 0, 2, 2] = 0
        a.imag[1, 0, 0, 0] = -a.imag[0, 1, 0, 0]
        a.imag[1, 0, 0, 1] = -a.imag[0, 1, 1, 0]
        a.imag[1, 0, 0, 2] = -a.imag[0, 1, 2, 0]
        a.imag[1, 0, 1, 0] = -a.imag[0, 1, 0, 1]
        a.imag[1, 0, 1, 1] = -a.imag[0, 1, 1, 1]
        a.imag[1, 0, 1, 2] = -a.imag[0, 1, 2, 1]
        a.imag[1, 0, 2, 0] = -a.imag[0, 1, 0, 2]
        a.imag[1, 0, 2, 1] = -a.imag[0, 1, 1, 2]
        a.imag[1, 0, 2, 2] = -a.imag[0, 1, 2, 2]
        a.imag[1, 1, 0, 0] = 0
        a.imag[1, 1, 1, 0] = -a.imag[1, 1, 0, 1]
        a.imag[1, 1, 1, 1] = 0
        a.imag[1, 1, 2, 0] = -a.imag[1, 1, 0, 2]
        a.imag[1, 1, 2, 1] = -a.imag[1, 1, 1, 2]
        a.imag[1, 1, 2, 2] = 0
        a.imag[2, 0, 0, 0] = -a.imag[0, 2, 0, 0]
        a.imag[2, 0, 0, 1] = -a.imag[0, 2, 1, 0]
        a.imag[2, 0, 0, 2] = -a.imag[0, 2, 2, 0]
        a.imag[2, 0, 1, 0] = -a.imag[0, 2, 0, 1]
        a.imag[2, 0, 1, 1] = -a.imag[0, 2, 1, 1]
        a.imag[2, 0, 1, 2] = -a.imag[0, 2, 2, 1]
        a.imag[2, 0, 2, 0] = -a.imag[0, 2, 0, 2]
        a.imag[2, 0, 2, 1] = -a.imag[0, 2, 1, 2]
        a.imag[2, 0, 2, 2] = -a.imag[0, 2, 2, 2]
        a.imag[2, 1, 0, 0] = -a.imag[1, 2, 0, 0]
        a.imag[2, 1, 0, 1] = -a.imag[1, 2, 1, 0]
        a.imag[2, 1, 0, 2] = -a.imag[1, 2, 2, 0]
        a.imag[2, 1, 1, 0] = -a.imag[1, 2, 0, 1]
        a.imag[2, 1, 1, 1] = -a.imag[1, 2, 1, 1]
        a.imag[2, 1, 1, 2] = -a.imag[1, 2, 2, 1]
        a.imag[2, 1, 2, 0] = -a.imag[1, 2, 0, 2]
        a.imag[2, 1, 2, 1] = -a.imag[1, 2, 1, 2]
        a.imag[2, 1, 2, 2] = -a.imag[1, 2, 2, 2]
        a.imag[2, 2, 0, 0] = 0
        a.imag[2, 2, 1, 0] = -a.imag[2, 2, 0, 1]
        a.imag[2, 2, 1, 1] = 0
        a.imag[2, 2, 2, 0] = -a.imag[2, 2, 0, 2]
        a.imag[2, 2, 2, 1] = -a.imag[2, 2, 1, 2]
        a.imag[2, 2, 2, 2] = 0
        return a

    @staticmethod
    def symmetrize_6(b):
        a = np.array(b, copy=True, dtype=complex)
        a.real[0, 0, 1, 0] = a.real[0, 0, 0, 1]
        a.real[0, 0, 2, 0] = a.real[0, 0, 0, 2]
        a.real[0, 0, 2, 1] = a.real[0, 0, 1, 2]
        a.real[0, 1, 0, 0] = a.real[0, 0, 0, 1]
        a.real[0, 1, 1, 0] = a.real[0, 0, 1, 1]
        a.real[0, 1, 2, 0] = a.real[0, 0, 1, 2]
        a.real[0, 2, 0, 0] = a.real[0, 0, 0, 2]
        a.real[0, 2, 0, 1] = a.real[0, 1, 0, 2]
        a.real[0, 2, 1, 0] = a.real[0, 0, 1, 2]
        a.real[0, 2, 1, 1] = a.real[0, 1, 1, 2]
        a.real[0, 2, 2, 0] = a.real[0, 0, 2, 2]
        a.real[0, 2, 2, 1] = a.real[0, 1, 2, 2]
        a.real[1, 0, 0, 0] = a.real[0, 0, 0, 1]
        a.real[1, 0, 0, 1] = a.real[0, 0, 1, 1]
        a.real[1, 0, 0, 2] = a.real[0, 0, 1, 2]
        a.real[1, 0, 1, 0] = a.real[0, 1, 0, 1]
        a.real[1, 0, 1, 1] = a.real[0, 1, 1, 1]
        a.real[1, 0, 1, 2] = a.real[0, 1, 2, 1]
        a.real[1, 0, 2, 0] = a.real[0, 1, 0, 2]
        a.real[1, 0, 2, 1] = a.real[0, 1, 1, 2]
        a.real[1, 0, 2, 2] = a.real[0, 1, 2, 2]
        a.real[1, 1, 0, 0] = a.real[0, 0, 1, 1]
        a.real[1, 1, 0, 1] = a.real[0, 1, 1, 1]
        a.real[1, 1, 0, 2] = a.real[0, 1, 1, 2]
        a.real[1, 1, 1, 0] = a.real[0, 1, 1, 1]
        a.real[1, 1, 2, 0] = a.real[0, 1, 1, 2]
        a.real[1, 1, 2, 1] = a.real[1, 1, 1, 2]
        a.real[1, 2, 0, 0] = a.real[0, 0, 1, 2]
        a.real[1, 2, 0, 1] = a.real[0, 1, 1, 2]
        a.real[1, 2, 0, 2] = a.real[0, 2, 1, 2]
        a.real[1, 2, 1, 0] = a.real[0, 1, 2, 1]
        a.real[1, 2, 1, 1] = a.real[1, 1, 1, 2]
        a.real[1, 2, 2, 0] = a.real[0, 1, 2, 2]
        a.real[1, 2, 2, 1] = a.real[1, 1, 2, 2]
        a.real[2, 0, 0, 0] = a.real[0, 0, 0, 2]
        a.real[2, 0, 0, 1] = a.real[0, 0, 1, 2]
        a.real[2, 0, 0, 2] = a.real[0, 0, 2, 2]
        a.real[2, 0, 1, 0] = a.real[0, 1, 0, 2]
        a.real[2, 0, 1, 1] = a.real[0, 1, 1, 2]
        a.real[2, 0, 1, 2] = a.real[0, 1, 2, 2]
        a.real[2, 0, 2, 0] = a.real[0, 2, 0, 2]
        a.real[2, 0, 2, 1] = a.real[0, 2, 1, 2]
        a.real[2, 0, 2, 2] = a.real[0, 2, 2, 2]
        a.real[2, 1, 0, 0] = a.real[0, 0, 1, 2]
        a.real[2, 1, 0, 1] = a.real[0, 1, 2, 1]
        a.real[2, 1, 0, 2] = a.real[0, 1, 2, 2]
        a.real[2, 1, 1, 0] = a.real[0, 1, 1, 2]
        a.real[2, 1, 1, 1] = a.real[1, 1, 1, 2]
        a.real[2, 1, 1, 2] = a.real[1, 1, 2, 2]
        a.real[2, 1, 2, 0] = a.real[0, 2, 1, 2]
        a.real[2, 1, 2, 1] = a.real[1, 2, 1, 2]
        a.real[2, 1, 2, 2] = a.real[1, 2, 2, 2]
        a.real[2, 2, 0, 0] = a.real[0, 0, 2, 2]
        a.real[2, 2, 0, 1] = a.real[0, 1, 2, 2]
        a.real[2, 2, 0, 2] = a.real[0, 2, 2, 2]
        a.real[2, 2, 1, 0] = a.real[0, 1, 2, 2]
        a.real[2, 2, 1, 1] = a.real[1, 1, 2, 2]
        a.real[2, 2, 1, 2] = a.real[1, 2, 2, 2]
        a.real[2, 2, 2, 0] = a.real[0, 2, 2, 2]
        a.real[2, 2, 2, 1] = a.real[1, 2, 2, 2]
        a.imag[0, 0, 0, 0] = 0
        a.imag[0, 0, 1, 0] = -a.imag[0, 0, 0, 1]
        a.imag[0, 0, 1, 1] = 0
        a.imag[0, 0, 2, 0] = -a.imag[0, 0, 0, 2]
        a.imag[0, 0, 2, 1] = -a.imag[0, 0, 1, 2]
        a.imag[0, 0, 2, 2] = 0
        a.imag[0, 1, 0, 0] = a.imag[0, 0, 0, 1]
        a.imag[0, 1, 1, 0] = 0
        a.imag[0, 1, 2, 0] = -a.imag[0, 0, 1, 2]
        a.imag[0, 2, 0, 0] = a.imag[0, 0, 0, 2]
        a.imag[0, 2, 0, 1] = a.imag[0, 1, 0, 2]
        a.imag[0, 2, 1, 0] = a.imag[0, 0, 1, 2]
        a.imag[0, 2, 1, 1] = a.imag[0, 1, 1, 2]
        a.imag[0, 2, 2, 0] = 0
        a.imag[0, 2, 2, 1] = a.imag[0, 1, 2, 2]
        a.imag[1, 0, 0, 0] = -a.imag[0, 0, 0, 1]
        a.imag[1, 0, 0, 1] = 0
        a.imag[1, 0, 0, 2] = a.imag[0, 0, 1, 2]
        a.imag[1, 0, 1, 0] = -a.imag[0, 1, 0, 1]
        a.imag[1, 0, 1, 1] = -a.imag[0, 1, 1, 1]
        a.imag[1, 0, 1, 2] = -a.imag[0, 1, 2, 1]
        a.imag[1, 0, 2, 0] = -a.imag[0, 1, 0, 2]
        a.imag[1, 0, 2, 1] = -a.imag[0, 1, 1, 2]
        a.imag[1, 0, 2, 2] = -a.imag[0, 1, 2, 2]
        a.imag[1, 1, 0, 0] = 0
        a.imag[1, 1, 0, 1] = a.imag[0, 1, 1, 1]
        a.imag[1, 1, 0, 2] = a.imag[0, 1, 1, 2]
        a.imag[1, 1, 1, 0] = -a.imag[0, 1, 1, 1]
        a.imag[1, 1, 1, 1] = 0
        a.imag[1, 1, 2, 0] = -a.imag[0, 1, 1, 2]
        a.imag[1, 1, 2, 1] = -a.imag[1, 1, 1, 2]
        a.imag[1, 1, 2, 2] = 0
        a.imag[1, 2, 0, 0] = a.imag[0, 0, 1, 2]
        a.imag[1, 2, 0, 1] = a.imag[0, 1, 1, 2]
        a.imag[1, 2, 0, 2] = a.imag[0, 2, 1, 2]
        a.imag[1, 2, 1, 0] = -a.imag[0, 1, 2, 1]
        a.imag[1, 2, 1, 1] = a.imag[1, 1, 1, 2]
        a.imag[1, 2, 2, 0] = -a.imag[0, 1, 2, 2]
        a.imag[1, 2, 2, 1] = 0
        a.imag[2, 0, 0, 0] = -a.imag[0, 0, 0, 2]
        a.imag[2, 0, 0, 1] = -a.imag[0, 0, 1, 2]
        a.imag[2, 0, 0, 2] = 0
        a.imag[2, 0, 1, 0] = -a.imag[0, 1, 0, 2]
        a.imag[2, 0, 1, 1] = -a.imag[0, 1, 1, 2]
        a.imag[2, 0, 1, 2] = -a.imag[0, 1, 2, 2]
        a.imag[2, 0, 2, 0] = -a.imag[0, 2, 0, 2]
        a.imag[2, 0, 2, 1] = -a.imag[0, 2, 1, 2]
        a.imag[2, 0, 2, 2] = -a.imag[0, 2, 2, 2]
        a.imag[2, 1, 0, 0] = -a.imag[0, 0, 1, 2]
        a.imag[2, 1, 0, 1] = a.imag[0, 1, 2, 1]
        a.imag[2, 1, 0, 2] = a.imag[0, 1, 2, 2]
        a.imag[2, 1, 1, 0] = -a.imag[0, 1, 1, 2]
        a.imag[2, 1, 1, 1] = -a.imag[1, 1, 1, 2]
        a.imag[2, 1, 1, 2] = 0
        a.imag[2, 1, 2, 0] = -a.imag[0, 2, 1, 2]
        a.imag[2, 1, 2, 1] = -a.imag[1, 2, 1, 2]
        a.imag[2, 1, 2, 2] = -a.imag[1, 2, 2, 2]
        a.imag[2, 2, 0, 0] = 0
        a.imag[2, 2, 0, 1] = a.imag[0, 1, 2, 2]
        a.imag[2, 2, 0, 2] = a.imag[0, 2, 2, 2]
        a.imag[2, 2, 1, 0] = -a.imag[0, 1, 2, 2]
        a.imag[2, 2, 1, 1] = 0
        a.imag[2, 2, 1, 2] = a.imag[1, 2, 2, 2]
        a.imag[2, 2, 2, 0] = -a.imag[0, 2, 2, 2]
        a.imag[2, 2, 2, 1] = -a.imag[1, 2, 2, 2]
        a.imag[2, 2, 2, 2] = 0
        return a

    @staticmethod
    def symmetrize_7(b):
        a = np.array(b, copy=True, dtype=complex)
        a[1, 0, 0, 0] = a[0, 1, 0, 0]
        a[1, 0, 0, 1] = a[0, 1, 0, 1]
        a[1, 0, 0, 2] = a[0, 1, 0, 2]
        a[1, 0, 1, 0] = a[0, 1, 1, 0]
        a[1, 0, 1, 1] = a[0, 1, 1, 1]
        a[1, 0, 1, 2] = a[0, 1, 1, 2]
        a[1, 0, 2, 0] = a[0, 1, 2, 0]
        a[1, 0, 2, 1] = a[0, 1, 2, 1]
        a[1, 0, 2, 2] = a[0, 1, 2, 2]
        a[2, 0, 0, 0] = a[0, 2, 0, 0]
        a[2, 0, 0, 1] = a[0, 2, 0, 1]
        a[2, 0, 0, 2] = a[0, 2, 0, 2]
        a[2, 0, 1, 0] = a[0, 2, 1, 0]
        a[2, 0, 1, 1] = a[0, 2, 1, 1]
        a[2, 0, 1, 2] = a[0, 2, 1, 2]
        a[2, 0, 2, 0] = a[0, 2, 2, 0]
        a[2, 0, 2, 1] = a[0, 2, 2, 1]
        a[2, 0, 2, 2] = a[0, 2, 2, 2]
        a[2, 1, 0, 0] = a[1, 2, 0, 0]
        a[2, 1, 0, 1] = a[1, 2, 0, 1]
        a[2, 1, 0, 2] = a[1, 2, 0, 2]
        a[2, 1, 1, 0] = a[1, 2, 1, 0]
        a[2, 1, 1, 1] = a[1, 2, 1, 1]
        a[2, 1, 1, 2] = a[1, 2, 1, 2]
        a[2, 1, 2, 0] = a[1, 2, 2, 0]
        a[2, 1, 2, 1] = a[1, 2, 2, 1]
        a[2, 1, 2, 2] = a[1, 2, 2, 2]
        return a

    @staticmethod
    def symmetrize_71(b):
        a = np.array(b, copy=True, dtype=complex)
        a[1, 0, 0, 0] = -a[0, 1, 0, 0]
        a[1, 0, 0, 1] = -a[0, 1, 0, 1]
        a[1, 0, 0, 2] = -a[0, 1, 0, 2]
        a[1, 0, 1, 0] = -a[0, 1, 1, 0]
        a[1, 0, 1, 1] = -a[0, 1, 1, 1]
        a[1, 0, 1, 2] = -a[0, 1, 1, 2]
        a[1, 0, 2, 0] = -a[0, 1, 2, 0]
        a[1, 0, 2, 1] = -a[0, 1, 2, 1]
        a[1, 0, 2, 2] = -a[0, 1, 2, 2]
        a[2, 0, 0, 0] = -a[0, 2, 0, 0]
        a[2, 0, 0, 1] = -a[0, 2, 0, 1]
        a[2, 0, 0, 2] = -a[0, 2, 0, 2]
        a[2, 0, 1, 0] = -a[0, 2, 1, 0]
        a[2, 0, 1, 1] = -a[0, 2, 1, 1]
        a[2, 0, 1, 2] = -a[0, 2, 1, 2]
        a[2, 0, 2, 0] = -a[0, 2, 2, 0]
        a[2, 0, 2, 1] = -a[0, 2, 2, 1]
        a[2, 0, 2, 2] = -a[0, 2, 2, 2]
        a[2, 1, 0, 0] = -a[1, 2, 0, 0]
        a[2, 1, 0, 1] = -a[1, 2, 0, 1]
        a[2, 1, 0, 2] = -a[1, 2, 0, 2]
        a[2, 1, 1, 0] = -a[1, 2, 1, 0]
        a[2, 1, 1, 1] = -a[1, 2, 1, 1]
        a[2, 1, 1, 2] = -a[1, 2, 1, 2]
        a[2, 1, 2, 0] = -a[1, 2, 2, 0]
        a[2, 1, 2, 1] = -a[1, 2, 2, 1]
        a[2, 1, 2, 2] = -a[1, 2, 2, 2]
        return a

    @staticmethod
    def symmetrize_8(b):
        """Symmetrize class-8 coefficients.

        Note that this function does not correctly take into account the
        translation between a basis where Wilson coefficients are symmetrized
        like the operators and the non-redundant WCxf basis!
        """
        a = np.array(b, copy=True, dtype=complex)
        a[1, 0, 0, 0] = a[0, 0, 1, 0]
        a[1, 0, 0, 1] = a[0, 0, 1, 1]
        a[1, 0, 0, 2] = a[0, 0, 1, 2]
        a[1, 1, 0, 0] = a[0, 1, 1, 0]
        a[1, 1, 0, 1] = a[0, 1, 1, 1]
        a[1, 1, 0, 2] = a[0, 1, 1, 2]
        a[2, 0, 0, 0] = a[0, 0, 2, 0]
        a[2, 0, 0, 1] = a[0, 0, 2, 1]
        a[2, 0, 0, 2] = a[0, 0, 2, 2]
        a[2, 0, 1, 0] = a[1, 2, 0, 0]+a[1, 0, 2, 0]-a[0, 2, 1, 0]
        a[2, 0, 1, 1] = a[1, 2, 0, 1]+a[1, 0, 2, 1]-a[0, 2, 1, 1]
        a[2, 0, 1, 2] = a[1, 2, 0, 2]+a[1, 0, 2, 2]-a[0, 2, 1, 2]
        a[2, 1, 0, 0] = a[0, 2, 1, 0]+a[0, 1, 2, 0]-a[1, 2, 0, 0]
        a[2, 1, 0, 1] = a[0, 2, 1, 1]+a[0, 1, 2, 1]-a[1, 2, 0, 1]
        a[2, 1, 0, 2] = a[0, 2, 1, 2]+a[0, 1, 2, 2]-a[1, 2, 0, 2]
        a[2, 1, 1, 0] = a[1, 1, 2, 0]
        a[2, 1, 1, 1] = a[1, 1, 2, 1]
        a[2, 1, 1, 2] = a[1, 1, 2, 2]
        a[2, 2, 0, 0] = a[0, 2, 2, 0]
        a[2, 2, 0, 1] = a[0, 2, 2, 1]
        a[2, 2, 0, 2] = a[0, 2, 2, 2]
        a[2, 2, 1, 0] = a[1, 2, 2, 0]
        a[2, 2, 1, 1] = a[1, 2, 2, 1]
        a[2, 2, 1, 2] = a[1, 2, 2, 2]
        return a

    @staticmethod
    def scale_8(b):
        """Translations necessary for class-8 coefficients
        to go from a basis with only non-redundant WCxf
        operators to a basis where the Wilson coefficients are symmetrized like
        the operators."""
        a = np.array(b, copy=True, dtype=complex)
        for i in range(3):
            a[0, 0, 1, i] = 1/2 * b[0, 0, 1, i]
            a[0, 0, 2, i] = 1/2 * b[0, 0, 2, i]
            a[0, 1, 1, i] = 1/2 * b[0, 1, 1, i]
            a[0, 1, 2, i] = 2/3 * b[0, 1, 2, i] - 1/6 * b[0, 2, 1, i] - 1/6 * b[1, 0, 2, i] + 1/6 * b[1, 2, 0, i]
            a[0, 2, 1, i] = - (1/6) * b[0, 1, 2, i] + 2/3 * b[0, 2, 1, i] + 1/6 * b[1, 0, 2, i] + 1/3 * b[1, 2, 0, i]
            a[0, 2, 2, i] = 1/2 * b[0, 2, 2, i]
            a[1, 0, 2, i] = - (1/6) * b[0, 1, 2, i] + 1/6 * b[0, 2, 1, i] + 2/3 * b[1, 0, 2, i] - 1/6 * b[1, 2, 0, i]
            a[1, 1, 2, i] = 1/2 * b[1, 1, 2, i]
            a[1, 2, 0, i] = 1/6 * b[0, 1, 2, i] + 1/3 * b[0, 2, 1, i] - 1/6 * b[1, 0, 2, i] + 2/3 * b[1, 2, 0, i]
            a[1, 2, 2, i] = 1/2 * b[1, 2, 2, i]
        return a

    @staticmethod
    def unscale_8(b):
        """Translations necessary for class-8 coefficients
        to go from a basis where the Wilson coefficients are symmetrized like
        the operators to a basis with only non-redundant WCxf operators."""
        a = np.array(b, copy=True, dtype=complex)
        for i in range(3):
            a[0, 0, 1, i] = 2 * b[0, 0, 1, i]
            a[0, 0, 2, i] = 2 * b[0, 0, 2, i]
            a[0, 1, 1, i] = 2 * b[0, 1, 1, i]
            a[0, 1, 2, i] = 2 * b[0, 1, 2, i] + b[0, 2, 1, i] - b[1, 2, 0, i]
            a[0, 2, 1, i] = b[0, 1, 2, i] + 3 * b[0, 2, 1, i] - b[1, 0, 2, i] - 2 * b[1, 2, 0, i]
            a[0, 2, 2, i] = 2 * b[0, 2, 2, i]
            a[1, 0, 2, i] = - b[0, 2, 1, i] + 2 * b[1, 0, 2, i] + b[1, 2, 0, i]
            a[1, 1, 2, i] = 2 * b[1, 1, 2, i]
            a[1, 2, 0, i] = - b[0, 1, 2, i] - 2 * b[0, 2, 1, i] + b[1, 0, 2, i] + 3 * b[1, 2, 0, i]
            a[1, 2, 2, i] = 2 * b[1, 2, 2, i]
        return a

    @staticmethod
    def symmetrize_9(b):
        a = np.array(b, copy=True, dtype=complex)
        a[1, 0] = a[0, 1]
        a[2, 0] = a[0, 2]
        a[2, 1] = a[1, 2]
        return a

    def unscale_dict(self, C):
        """Undo the scaling applied in `scale_dict`."""
        C_out = {k: self._scale_dict[k] * v for k, v in C.items()}
        for k in self.C_symm_keys.get(8,()):
            C_out[k] = self.unscale_8(C_out[k])
        return C_out

    def symmetrize(self, C):
        """Symmetrize the Wilson coefficient arrays.

        Note that this function does not take into account the symmetry factors
        that occur when transitioning from a basis with only non-redundant operators
        (like in WCxf) to a basis where the Wilson coefficients are symmetrized
        like the operators. See `symmetrize_nonred` for this case."""
        C_symm = {}
        for i, v in C.items():
            if i in self.C_symm_keys.get(0, ()):
                C_symm[i] = v.real
            elif i in self.C_symm_keys.get(1, []) + self.C_symm_keys.get(3, []):
                C_symm[i] = v # nothing to do
            elif i in self.C_symm_keys.get(2, ()):
                C_symm[i] = self.symmetrize_2(C[i])
            elif i in self.C_symm_keys.get(4, ()):
                C_symm[i] = self.symmetrize_4(C[i])
            elif i in self.C_symm_keys.get(5, ()):
                C_symm[i] = self.symmetrize_5(C[i])
            elif i in self.C_symm_keys.get(6, ()):
                C_symm[i] = self.symmetrize_6(C[i])
            elif i in self.C_symm_keys.get(7, ()):
                C_symm[i] = self.symmetrize_7(C[i])
            elif i in self.C_symm_keys.get(8, ()):
                C_symm[i] = self.symmetrize_8(C[i])
            elif i in self.C_symm_keys.get(9, ()):
                C_symm[i] = self.symmetrize_9(C[i])
        return C_symm

    def symmetrize_nonred(self, C):
        """Symmetrize the Wilson coefficient arrays.

        This function takes into account the symmetry factors
        that occur when transitioning from a basis with only non-redundant operators
        (like in WCxf) to a basis where the Wilson coefficients are symmetrized
        like the operators."""
        C_symm = {}
        for i, v in C.items():
            if i in self.C_symm_keys.get(0, ()):
                C_symm[i] = v.real
            elif i in self.C_symm_keys.get(1, []) + self.C_symm_keys.get(3, []):
                C_symm[i] = v # nothing to do
            elif i in self.C_symm_keys.get(2, ()):
                C_symm[i] = self.symmetrize_2(C[i])
            elif i in self.C_symm_keys.get(4, ()):
                C_symm[i] = self.symmetrize_4(C[i])
                C_symm[i] = C_symm[i] / self._d_4
            elif i in self.C_symm_keys.get(41, ()):
                C_symm[i] = self.symmetrize_41(C[i])
                C_symm[i] = C_symm[i] / self._d_4
            elif i in self.C_symm_keys.get(5, ()):
                C_symm[i] = self.symmetrize_5(C[i])
            elif i in self.C_symm_keys.get(6, ()):
                C_symm[i] = self.symmetrize_6(C[i])
                C_symm[i] = C_symm[i] / self._d_6
            elif i in self.C_symm_keys.get(7, ()):
                C_symm[i] = self.symmetrize_7(C[i])
                C_symm[i] = C_symm[i] / self._d_7
            elif i in self.C_symm_keys.get(71, ()):
                C_symm[i] = self.symmetrize_71(C[i])
                C_symm[i] = C_symm[i] / self._d_7
            elif i in self.C_symm_keys.get(8, ()):
                C_symm[i] = self.scale_8(C[i])
                C_symm[i] = self.symmetrize_8(C_symm[i])
            elif i in self.C_symm_keys.get(9, ()):
                C_symm[i] = self.symmetrize_9(C[i])
        return C_symm

    def wcxf2arrays_symmetrized(self, d):
        """Convert a dictionary with a Wilson coefficient
        name followed by underscore and numeric indices as keys and numbers as
        values to a dictionary with Wilson coefficient names as keys and
        numbers or numpy arrays as values.


        In contrast to `wcxf2arrays`, here the numpy arrays fulfill the same
        symmetry relations as the operators (i.e. they contain redundant entries)
        and they do not contain undefined indices.

        Zero arrays are added for missing coefficients."""
        C = self.wcxf2arrays(d)
        C = self.symmetrize_nonred(C)
        C = self.add_missing(C)
        return C

    def arrays2wcxf_nonred(self, C):
        """Convert a dictionary with Wilson coefficient names as keys and
        numbers or numpy arrays as values to a dictionary with a Wilson coefficient
        name followed by underscore and numeric indices as keys and numbers as
        values.

        In contrast to `arrays2wcxf`, here the Wilson coefficient arrays are assumed
        to fulfill the same symmetry relations as the operators, i.e. contain
        redundant entries, while the WCxf output refers to the non-redundant basis."""
        C_out = self.unscale_dict(C)
        d = self.arrays2wcxf(C_out)
        return d
