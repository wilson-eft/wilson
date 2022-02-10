import unittest
from wilson.util import wet_jms, wetutil
import numpy as np
from wilson.test_wilson import get_random_wc
import numbers
from numpy.testing import assert_array_equal

class TestWETutil(unittest.TestCase):
    def test_scalar2array(self):
        d = {'bla_123': 3, 'blo': 5j}
        da = wet_jms._scalar2array(d)
        self.assertEqual(da['blo'], 5j)
        self.assertIn('bla', da)
        self.assertEqual(da['bla'].shape, (3, 3, 3))
        self.assertTrue(np.isnan(da['bla'][0, 0, 0]))
        self.assertEqual(da['bla'][0, 1, 2], 3)

    def test_padding(self):
        wc = get_random_wc('WET', 'JMS', 90)
        C = wetutil.wcxf2arrays(wc.dict)
        C_shapes = {
            k: 1
            if isinstance(v, numbers.Number)
            else v.shape
            for k,v in C.items()
        }
        self.assertGreaterEqual(wetutil.C_keys_shape.items(), C_shapes.items())
        C_padded = wetutil.pad_C(C)
        C_padded_shapes = {
            k: 1
            if isinstance(v, numbers.Number)
            else v.shape
            for k,v in C_padded.items()
        }
        self.assertEqual(
            3, min((min(v) for v in C_padded_shapes.values() if v != 1))
        )
        for k, v in wetutil.unpad_C(C_padded).items():
            assert_array_equal(C[k], v)

    def test_symmetrize(self):
        wc = get_random_wc('WET', 'JMS', 90)
        C = wetutil.wcxf2arrays_symmetrized(wc.dict)
        d = wetutil.arrays2wcxf_nonred(C)
        self.assertEqual(wc.dict, d)


class TestKeysShapesSymm(unittest.TestCase):

    # names of Wilson coefficients with the same fermionic symmetry properties
    # numbering is inspired by the corresponding categorization in SMEFT
    C_symm_keys = {}
    # 0 0F scalar object
    C_symm_keys[0] = ["G", "Gtilde"] + ["e", "gs"]
    # 1 2F general 3x3 matrix
    C_symm_keys[1] = ["egamma", "uG","dG", "ugamma", "dgamma"] + ["Mnu", "Mu",
        "Me", "Md"]
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
    C_symm_keys[71] = ['SuudLR', 'SuudRL', 'SdduRL']

    def test_symm_keys(self):
        self.assertEqual(self.C_symm_keys.keys(), wetutil.C_symm_keys.keys())
        for k, v in self.C_symm_keys.items():
            self.assertEqual(set(v), set(wetutil.C_symm_keys[k]))

    def test_needs_padding(self):
        self.assertEqual(wetutil._needs_padding, True)
