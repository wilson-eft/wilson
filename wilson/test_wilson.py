import unittest
import wcxf
import wilson
import numpy as np
import pkgutil


np.random.seed(235)


def get_random_wc(eft, basis, scale, cmax=1e-6):
    """Generate a random Wilson coefficient instance for a given basis."""
    basis_obj = wcxf.Basis[eft, basis]
    _wc = {}
    for s in basis_obj.sectors.values():
        for name, d in s.items():
            _wc[name] = cmax * np.random.rand()
            if 'real' not in d or not d['real']:
                _wc[name] += 1j * cmax * np.random.rand()
    return wcxf.WC(eft, basis, scale, wcxf.WC.dict2values(_wc))


par = {
    'm_Z': 91.1876,
    'm_b': 4.18,
    'm_d': 4.8e-3,
    'm_s': 0.095,
    'm_t': 173.3,
    'm_c': 1.27,
    'm_u': 2.3e-3,
    'alpha_e': 1/127.944,
    'alpha_s': 0.1185,
    'GF': 1.1663787e-5,
}


class TestWilson(unittest.TestCase):
    def test_class(self):
        wc = wcxf.WC('SMEFT', 'Warsaw', 1000, {'qd1_1123': 1})
        wilson.Wilson(wc)

    def test_set_initial_wcxf(self):
        test_file = pkgutil.get_data('wilson', 'data/tests/wcxf-flavio-example.yml')
        wcxf_wc = wcxf.WC.load(test_file.decode('utf-8'))
        wcxf_wc.validate()
        wilson_wc = wilson.Wilson(wcxf_wc)
        wc_out = wilson_wc.match_run('WET', 'flavio', 160, 'sb')
        self.assertEqual(wc_out.dict['C9_bsee'], -1+0.01j)
        self.assertEqual(wc_out.dict['C9p_bsee'], 0.1)
        self.assertEqual(wc_out.dict['C10_bsee'], 0.05j)
        self.assertEqual(wc_out.dict['C10p_bsee'], 0.1-0.3j)
        self.assertEqual(wc_out.dict.get('CS_bsee', 0), 0)

    def test_set_initial_wcxf_minimal(self):
        for eft in ['WET', 'WET-4', 'WET-3']:
            wc = wcxf.WC(eft, 'flavio', 120, {'CVLL_sdsd': {'Im': 1}})
            ww = wilson.Wilson(wc)
            self.assertEqual(ww.match_run(eft, 'flavio', 120, 'sdsd').dict['CVLL_sdsd'], 1j)
            pf = 4 * par['GF'] / np.sqrt(2)
            wc = wcxf.WC(eft, 'Bern', 120, {'1dsds': {'Im': 1/pf}})
            ww = wilson.Wilson(wc)
            self.assertAlmostEqual(ww.match_run(eft, 'flavio', 120, 'sdsd').dict['CVLL_sdsd'], 1j)

    def tets_repr(self):
        wc = wilson.Wilson(wc)
        wc._repr_markdown_()
        wc.set_initial({'C7_bs': -0.1}, 5)
        wc._repr_markdown_()

    def test_run_wcxf(self):
        for eft in [('WET', 'WET', 120, 120), ('WET', 'WET-4', 120, 3), ('WET', 'WET-3', 120, 2),
                    ('WET-4', 'WET-4', 3, 2), ('WET-4', 'WET-3', 3, 2),
                    ('WET-3', 'WET-3', 2, 1), ]:
            wc = wcxf.WC(eft[0], 'flavio', eft[2], {'CVLL_sdsd': {'Im': 1}})
            ww = wilson.Wilson(wc)
            wc_out = ww.match_run(eft[1], 'flavio', eft[3])
            wc_out.validate()


class TestRGsolution(unittest.TestCase):
    def test_rgsolution_smeft(self):
        wc = wcxf.WC('SMEFT', 'Warsaw', 1e5, {'qu1_1233': 1e-7})
        wc.validate()
        smeft = wilson.run.smeft.SMEFT(wc)
        sol = smeft.run_continuous(160)
        x, y = sol.plotdata('qu1_1233')
        self.assertTupleEqual(x.shape, (50,))
        self.assertTupleEqual(y.shape, (50,))
        self.assertEqual(x.dtype, float)
        self.assertEqual(y.dtype, float)
