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
            self.assertSetEqual(set(clist), set(allkeys), msg=f"Failed for {sname}")


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
                            msg=f'{k} missing in output')


class TestEvolutionMatrices(unittest.TestCase):
    def test_inverse_s(self):
        # check inverse of QCD evolution matrices
        args = (5, 0.12, 1/128, 0, 0, 0.1, 1.2, 4.2, 0, 0.106, 1.77)
        for c in ['I', 'II', 'III', 'IV', 'IVe', 'sb', 'db', 'sd', 'mue', 'cu', 'Vb']:
            npt.assert_array_almost_equal(rge.getUs(c, 0.123, *args),
                                          np.linalg.inv(rge.getUs(c, 1/0.123, *args),),
                                          err_msg=f"Failed for {c}")

    def test_Ue_expm(self):
        # compare QED evolution matrices with the upper right block of the
        # matrix exponential exp([[X, gamma_e^T], [0, X + 1]] log(eta_s)),
        # X = gamma_s^T / (2 beta_0). The f=3 classes 'sd' and 'dF0' have
        # eigenvalue differences close to 1 (numerically delicate).
        from scipy.linalg import expm
        for f, classes in [(5, ['I', 'sb', 'dF0', 'mue']), (3, ['sd', 'dF0'])]:
            masses = (0.0022, 0.0047, 0.095, 1.28, 4.18, 0.000511, 0.106, 1.777)
            alpha_s, alpha_e, eta_s = 0.2, 1 / 128, 0.6
            b0s = 11 - 2 * f / 3
            for c in classes:
                perm = rge.get_permissible_wcs(c, f)
                sel = (lambda M: M) if perm == 'all' else (lambda M: M[perm][:, perm])
                X = sel(getattr(rge.adm, 'adm_s_' + c)(f, *masses)).T / (2 * b0s)
                E = sel(getattr(rge.adm, 'adm_e_' + c)(f, *masses)).T
                n = len(X)
                B = np.block([[X, E], [np.zeros((n, n)), X + np.eye(n)]])
                Ue_ref = alpha_e / (2 * b0s * alpha_s) * expm(B * np.log(eta_s))[:n, n:]
                Ue = rge.getUe(c, eta_s, f, alpha_s, alpha_e, *masses)
                npt.assert_allclose(Ue, Ue_ref, rtol=0, atol=1e-10 * np.max(np.abs(Ue_ref)),
                                    err_msg=f"Failed for {c}, f={f}")


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
                            msg=f'{k} missing in output')

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
            self.assertAlmostEqual(v, wc_hi[k], msg=f"Failed for {k}")

    def test_10(self):
        # running without QED = running with alpha_e = 0
        wc_10 = self.wet_10.run(4, ('sbsb',))
        wc_11 = self.wet_11_ae0.run(4, ('sbsb',))
        for k, v in wc_10.dict.items():
            self.assertAlmostEqual(v, wc_11[k], msg=f"Failed for {k}")

    def test_01(self):
        # running without QED = running with alpha_e = 0
        wc_01 = self.wet_01.run(4, ('sbsb',))
        wc_11 = self.wet_11_as0.run(4, ('sbsb',))
        for k, v in wc_01.dict.items():
            self.assertAlmostEqual(v, wc_11[k], msg=f"Failed for {k}")
