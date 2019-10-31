import unittest
from wilson.run import wet
from wilson import wcxf
import numpy as np
import numpy.testing as npt
from wilson.run.wet import rge

np.random.seed(112)


def get_random_wc(eft, basis, scale, cmax=1e-2):
    """Generate a random wcxf.WC instance for a given basis."""
    basis_obj = wcxf.Basis[eft, basis]
    _wc = {}
    for s in basis_obj.sectors.values():
        for name, d in s.items():
            _wc[name] = cmax * np.random.rand()
            if 'real' not in d or not d['real']:
                _wc[name] += 1j * cmax * np.random.rand()
    return wcxf.WC(eft, basis, scale, wcxf.WC.dict2values(_wc))


class TestDef(unittest.TestCase):

    def test_sectors(self):
        for sname, clist in wet.definitions.coeffs.items():
            self.assertIn(sname, wcxf.Basis['WET', 'JMS'].sectors.keys())
            allkeys = wcxf.Basis['WET', 'JMS'].sectors[sname].keys()
            # for c in clist:
                # self.assertIn(c, allkeys, msg="{} not found in {}".format(c, sname))
            self.assertSetEqual(set(clist), set(allkeys), msg="Failed for {}".format(sname))


class TestClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.wc = get_random_wc('WET', 'JMS', 160)
        cls.wet = wet.WETrunner(cls.wc)

    def test_init(self):
        with self.assertRaises(AssertionError):
            wet.WETrunner(0)  # argument is not a WC instance
        wcf = get_random_wc('WET', 'flavio', 160)  # wrong basis
        with self.assertRaises(AssertionError):
            wet.WETrunner(wcf)

    def test_attr(self):
        self.assertEqual(self.wet.scale_in, 160)
        self.assertEqual(self.wet.C_in, self.wc.dict)

    def test_wcxf(self):
        wc = self.wet.run(4.2)
        wc.validate()

    def test_run(self):
        C_out = self.wet.run(4.2).dict
        # assert all input WCs are present in the output
        # (not vice versa as RGE can generate them from zero)
        for k in self.wet.C_in:
            self.assertTrue(k in C_out,
                            msg='{} missing in output'.format(k))


class TestEvolutionMatrices(unittest.TestCase):
    def test_inverse_s(self):
        # check inverse of QCD evolution matrices
        args = (5, 0.12, 1/128, 0, 0, 0.1, 1.2, 4.2, 0, 0.106, 1.77)
        for c in ['I', 'II', 'III', 'IV', 'IVe', 'sb', 'db', 'sd', 'mue', 'cu', 'Vb']:
            npt.assert_array_almost_equal(rge.getUs(c, 0.123, *args),
                                          np.linalg.inv(rge.getUs(c, 1/0.123, *args),),
                                          err_msg="Failed for {}".format(c))


class TestClassWET4(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.wc = get_random_wc('WET-4', 'JMS', 4)
        cls.wet = wet.WETrunner(cls.wc)

    def test_wcxf(self):
        wc = self.wet.run(1.2)
        wc.validate()

    def test_run(self):
        C_out = self.wet.run(1.2).dict
        # assert all input WCs are present in the output
        # (not vice versa as RGE can generate them from zero)
        for k in self.wet.C_in:
            self.assertTrue(k in C_out,
                            msg='{} missing in output'.format(k))

class TestOrder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wc = get_random_wc('WET', 'JMS', 160)
        cls.wet_11 = wet.WETrunner(cls.wc)
        cls.wet_11_ae0 = wet.WETrunner(cls.wc)
        cls.wet_11_ae0.parameters['alpha_e'] = 0
        cls.wet_11_as0 = wet.WETrunner(cls.wc)
        cls.wet_11_as0.parameters['alpha_s'] = 1e-16  # avoid div by 0
        cls.wet_10 = wet.WETrunner(cls.wc, qed_order=0)
        cls.wet_01 = wet.WETrunner(cls.wc, qcd_order=0)
        cls.wet_00 = wet.WETrunner(cls.wc, qed_order=0, qcd_order=0)

    def test_00(self):
        # running with order 0 = not running
        wc_hi = self.wet_11.run(160, ('sbsb',))
        wc_lo = self.wet_00.run(4, ('sbsb',))
        for k, v in wc_lo.dict.items():
            self.assertAlmostEqual(v, wc_hi[k], msg="Failed for {}".format(k))

    def test_10(self):
        # running without QED = running with alpha_e = 0
        wc_10 = self.wet_10.run(4, ('sbsb',))
        wc_11 = self.wet_11_ae0.run(4, ('sbsb',))
        for k, v in wc_10.dict.items():
            self.assertAlmostEqual(v, wc_11[k], msg="Failed for {}".format(k))

    def test_01(self):
        # running without QED = running with alpha_e = 0
        wc_01 = self.wet_01.run(4, ('sbsb',))
        wc_11 = self.wet_11_as0.run(4, ('sbsb',))
        for k, v in wc_01.dict.items():
            self.assertAlmostEqual(v, wc_11[k], msg="Failed for {}".format(k))
