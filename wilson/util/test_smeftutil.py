import unittest
import numpy as np
import numpy.testing as npt
from wilson.run.smeft import beta
from wilson.util import smeftutil
from wilson.util.common import EFTutil
from wilson.run.smeft.tests import test_beta
from wilson.test_wilson import get_random_wc
from wilson import wcxf
from numbers import Number

C = test_beta.C.copy()
for i in C:
    if i in smeftutil.WC_keys_2f + smeftutil.WC_keys_4f:
        # make Wilson coefficients involving fermions complex!
        C[i] = C[i] + 1j*C[i]

class TestSymm(unittest.TestCase):
    def test_keys(self):
        # check no parameter or WC was forgotten in the C_symm_keys lists
        self.assertEqual(
          set(smeftutil.C_keys),
          {c for cs in smeftutil.C_symm_keys.values() for c in cs}
        )

    def test_symmetrize_symmetric(self):
        a = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        npt.assert_array_equal(EFTutil.symmetrize_2(a), a)
        b = np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]])
        npt.assert_array_equal(EFTutil.symmetrize_2(b), a)

    def test_symmetrize_hermitian(self):
        a = np.array([[1, 2j, 3j], [-2j, 4, 5j], [-3j, -5j, 6]])
        npt.assert_array_equal(EFTutil.symmetrize_2(a), a)
        b = np.array([[1, 2j, 3j], [0, 4, 5j], [0, 0, 6]])
        npt.assert_array_equal(EFTutil.symmetrize_2(b), a)

    def test_symmetrize_C(self):
        C_symm = smeftutil.symmetrize(C)
        # check all keys are present
        self.assertSetEqual(set(C.keys()), set(C_symm.keys()))
        for i, v in C_symm.items():
            # check trivial cases are the same
            if i in smeftutil.C_symm_keys[0] + smeftutil.C_symm_keys[1] + smeftutil.C_symm_keys[3]:
                if smeftutil.C_keys_shape[i] == 1:
                    self.assertEqual(v, C[i])
                else:
                    npt.assert_array_equal(v, C[i])
            # check symmetric
            elif i in smeftutil.C_symm_keys[9]:
                npt.assert_array_equal(v, v.T)
            # check hermitian
            elif i in smeftutil.C_symm_keys[2]:
                npt.assert_array_equal(v, v.T.conj())
            # check 2 identical FFbar
            elif i in smeftutil.C_symm_keys[4]:
                npt.assert_array_equal(v, v.transpose((2, 3, 0, 1)))
                npt.assert_array_equal(v, v.transpose((1, 0, 3, 2)).conj())
            # check 2 independent FFbar
            elif i in smeftutil.C_symm_keys[5]:
                npt.assert_array_equal(v, v.transpose((1, 0, 3, 2)).conj())
            # check special case ee
            elif i in smeftutil.C_symm_keys[6]:
                npt.assert_array_equal(v, v.transpose((2, 3, 0, 1)))
                npt.assert_array_equal(v, v.transpose((0, 3, 2, 1)))
                npt.assert_array_equal(v, v.transpose((2, 1, 0, 3)))
            # check special case qque
            elif i in smeftutil.C_symm_keys[7]:
                npt.assert_array_equal(v, v.transpose((1, 0, 2, 3)))
            # check special case qqql
            elif i in smeftutil.C_symm_keys[8]:
                # see eq. (10) of arXiv:1405.0486
                npt.assert_array_almost_equal(v + v.transpose((1, 0, 2, 3)), v.transpose((1, 2, 0, 3)) + v.transpose((2, 1, 0, 3)), decimal=15)

    def test_wcxf2array(self):
        wc = get_random_wc('SMEFT', 'Warsaw', 160)
        C = smeftutil.wcxf2arrays_symmetrized(wc.dict)
        d = smeftutil.arrays2wcxf_nonred(C)
        self.assertEqual(wc.dict.keys(), d.keys())
        for k, v in wc.dict.items():
            self.assertAlmostEqual(v, d[k], msg=f"Failed for {k}")

    def test_wcxf2array_incomplete(self):
        wc = wcxf.WC('SMEFT', 'Warsaw', 160, {'G': 1e-10})
        C = smeftutil.wcxf2arrays_symmetrized(wc.dict)
        d = smeftutil.arrays2wcxf_nonred(C)
        self.assertEqual(wc.dict.keys(), d.keys())
        for k, v in d.items():
            self.assertEqual(v, wc[k], msg=f"Failed for {k}")
            self.assertIsInstance(v, Number)


class TestKeysShapesSymm(unittest.TestCase):

    # names of SM parameters
    SM_keys = ['g', 'gp', 'gs', 'Lambda', 'm2', 'Gu', 'Gd', 'Ge',]

    # names of WCs with 0, 2, or 4 fermions (i.e. scalars, 3x3 matrices,
    # and 3x3x3x3 tensors)
    WC_keys_0f = ["G", "Gtilde", "W", "Wtilde", "phi", "phiBox", "phiD", "phiG",
                  "phiB", "phiW", "phiWB", "phiGtilde", "phiBtilde", "phiWtilde",
                  "phiWtildeB"]
    WC_keys_2f = ["uphi", "dphi", "ephi", "eW", "eB", "uG", "uW", "uB", "dG", "dW",
                  "dB", "phil1", "phil3", "phie", "phiq1", "phiq3", "phiu", "phid",
                  "phiud", "llphiphi"]
    WC_keys_4f = ["ll", "qq1", "qq3", "lq1", "lq3", "ee", "uu", "dd", "eu", "ed",
                  "ud1", "ud8", "le", "lu", "ld",  "qe", "qu1", "qd1", "qu8",
                  "qd8", "ledq", "quqd1", "quqd8", "lequ1", "lequ3", "duql",
                  "qque", "qqql", "duue"]

    C_keys = SM_keys + WC_keys_0f + WC_keys_2f + WC_keys_4f
    WC_keys = WC_keys_0f + WC_keys_2f + WC_keys_4f

    C_keys_shape = {
       'g': 1,
       'gp': 1,
       'gs': 1,
       'Lambda': 1,
       'm2': 1,
       'Gu': (3, 3),
       'Gd': (3, 3),
       'Ge': (3, 3),
       'G': 1,
       'Gtilde': 1,
       'W': 1,
       'Wtilde': 1,
       'phi': 1,
       'phiBox': 1,
       'phiD': 1,
       'phiG': 1,
       'phiB': 1,
       'phiW': 1,
       'phiWB': 1,
       'phiGtilde': 1,
       'phiBtilde': 1,
       'phiWtilde': 1,
       'phiWtildeB': 1,
       'uphi': (3, 3),
       'dphi': (3, 3),
       'ephi': (3, 3),
       'eW': (3, 3),
       'eB': (3, 3),
       'uG': (3, 3),
       'uW': (3, 3),
       'uB': (3, 3),
       'dG': (3, 3),
       'dW': (3, 3),
       'dB': (3, 3),
       'phil1': (3, 3),
       'phil3': (3, 3),
       'phie': (3, 3),
       'phiq1': (3, 3),
       'phiq3': (3, 3),
       'phiu': (3, 3),
       'phid': (3, 3),
       'phiud': (3, 3),
       'llphiphi': (3, 3),
       'll': (3, 3, 3, 3),
       'qq1': (3, 3, 3, 3),
       'qq3': (3, 3, 3, 3),
       'lq1': (3, 3, 3, 3),
       'lq3': (3, 3, 3, 3),
       'ee': (3, 3, 3, 3),
       'uu': (3, 3, 3, 3),
       'dd': (3, 3, 3, 3),
       'eu': (3, 3, 3, 3),
       'ed': (3, 3, 3, 3),
       'ud1': (3, 3, 3, 3),
       'ud8': (3, 3, 3, 3),
       'le': (3, 3, 3, 3),
       'lu': (3, 3, 3, 3),
       'ld': (3, 3, 3, 3),
       'qe': (3, 3, 3, 3),
       'qu1': (3, 3, 3, 3),
       'qd1': (3, 3, 3, 3),
       'qu8': (3, 3, 3, 3),
       'qd8': (3, 3, 3, 3),
       'ledq': (3, 3, 3, 3),
       'quqd1': (3, 3, 3, 3),
       'quqd8': (3, 3, 3, 3),
       'lequ1': (3, 3, 3, 3),
       'lequ3': (3, 3, 3, 3),
       'duql': (3, 3, 3, 3),
       'qque': (3, 3, 3, 3),
       'qqql': (3, 3, 3, 3),
       'duue': (3, 3, 3, 3),
    }

    # names of Wilson coefficients with the same fermionic symmetry properties
    C_symm_keys = {}
    # 0 0F scalar object
    C_symm_keys[0] = WC_keys_0f + ['g', 'gp', 'gs', 'Lambda', 'm2',]
    # 1 2F general 3x3 matrix
    C_symm_keys[1] = ["uphi", "dphi", "ephi", "eW", "eB", "uG", "uW", "uB", "dG", "dW", "dB", "phiud"] + ['Gu', 'Gd', 'Ge']
    # 2 2F Hermitian matrix
    C_symm_keys[2] = ["phil1", "phil3", "phie", "phiq1", "phiq3", "phiu", "phid",]
    # 3 4F general 3x3x3x3 object
    C_symm_keys[3] = ["ledq", "quqd1", "quqd8", "lequ1", "lequ3", "duql", "duue"]
    # 4 4F two identical ffbar currents
    C_symm_keys[4] = ["ll", "qq1", "qq3", "uu", "dd",]
    # 5 4F two independent ffbar currents
    C_symm_keys[5] = ["lq1", "lq3", "eu", "ed", "ud1", "ud8", "le", "lu", "ld", "qe", "qu1", "qd1", "qu8", "qd8",]
    # 6 4F two identical ffbar currents - special case Cee
    C_symm_keys[6] = ["ee",]
    # 7 4F Baryon-number-violating - special case Cqque
    C_symm_keys[7] = ["qque",]
    # 8 4F Baryon-number-violating - special case Cqqql
    C_symm_keys[8] = ["qqql",]
    # 9 2F symmetric matrix
    C_symm_keys[9] = ["llphiphi"]

    def test_keys(self):
        self.assertEqual(self.SM_keys, smeftutil.dim4_keys)
        self.assertEqual(self.WC_keys_0f, smeftutil.WC_keys_0f)
        self.assertEqual(self.WC_keys_2f, smeftutil.WC_keys_2f)
        self.assertEqual(self.WC_keys_4f, smeftutil.WC_keys_4f)
        self.assertEqual(self.C_keys, smeftutil.C_keys)
        self.assertEqual(self.WC_keys, smeftutil.WC_keys)

    def test_shape(self):
        self.assertEqual(self.C_keys_shape, smeftutil.C_keys_shape)

    def test_symm_keys(self):
        self.assertEqual(self.C_symm_keys.keys(), smeftutil.C_symm_keys.keys())
        for k, v in self.C_symm_keys.items():
            self.assertEqual(set(v), set(smeftutil.C_symm_keys[k]))

    def test_needs_padding(self):
        self.assertEqual(smeftutil._needs_padding, False)
