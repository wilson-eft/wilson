import unittest
import wcxf
import wilson
import numpy as np
import pkgutil
from wilson.parameters import p
import ckmutil.ckm, ckmutil.diag
import voluptuous as vol


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


def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        s for c in cls.__subclasses__() for s in all_subclasses(c))


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
        wilson.Wilson({'qd1_1123': 1}, 1000, 'SMEFT', 'Warsaw')

    def test_from_wc(self):
        wc = wcxf.WC('SMEFT', 'Warsaw', 1000, {'qd1_1123': 1})
        w1 = wilson.Wilson.from_wc(wc)
        w2 = wilson.Wilson({'qd1_1123': 1}, 1000, 'SMEFT', 'Warsaw')
        self.assertDictEqual(w1.wc.dict, w2.wc.dict)

    def test_set_initial_wcxf(self):
        test_file = pkgutil.get_data('wilson', 'data/tests/wcxf-flavio-example.yml')
        wcxf_wc = wcxf.WC.load(test_file.decode('utf-8'))
        wcxf_wc.validate()
        wilson_wc = wilson.Wilson.from_wc(wcxf_wc)
        wc_out = wilson_wc.match_run(160, 'WET', 'flavio', 'sb')
        self.assertEqual(wc_out.dict['C9_bsee'], -1+0.01j)
        self.assertEqual(wc_out.dict['C9p_bsee'], 0.1)
        self.assertEqual(wc_out.dict['C10_bsee'], 0.05j)
        self.assertEqual(wc_out.dict['C10p_bsee'], 0.1-0.3j)
        self.assertEqual(wc_out.dict.get('CS_bsee', 0), 0)

    def test_load_initial(self):
        test_file = pkgutil.get_data('wilson', 'data/tests/wcxf-flavio-example.yml')
        wilson_wc = wilson.Wilson.load_wc(test_file.decode('utf-8'))
        wc_out = wilson_wc.match_run(160, 'WET', 'flavio', 'sb')
        self.assertEqual(wc_out.dict['C9_bsee'], -1+0.01j)
        self.assertEqual(wc_out.dict['C9p_bsee'], 0.1)
        self.assertEqual(wc_out.dict['C10_bsee'], 0.05j)
        self.assertEqual(wc_out.dict['C10p_bsee'], 0.1-0.3j)
        self.assertEqual(wc_out.dict.get('CS_bsee', 0), 0)

    def test_set_initial_wcxf_minimal(self):
        for eft in ['WET', 'WET-4', 'WET-3']:
            wc = wcxf.WC(eft, 'flavio', 120, {'CVLL_sdsd': {'Im': 1}})
            ww = wilson.Wilson.from_wc(wc)
            self.assertEqual(ww.match_run(120, eft, 'flavio', ('sdsd',)).dict['CVLL_sdsd'], 1j)
            wc = wcxf.WC(eft, 'JMS', 120, {'VddLL_1212': {'Im': 1}})
            wc.validate()
            ww = wilson.Wilson.from_wc(wc)
            self.assertAlmostEqual(ww.match_run(120, eft, 'flavio', ('sdsd',)).dict['CVLL_sdsd'], 1j)

    def tets_repr(self):
        wc = wilson.Wilson.from_wc(wc)
        wc._repr_markdown_()
        wc.set_initial({'C7_bs': -0.1}, 5)
        wc._repr_markdown_()

    def test_run_wcxf(self):
        for eft in [('WET', 'WET', 120, 120), ('WET', 'WET-4', 120, 3), ('WET', 'WET-3', 120, 2),
                    ('WET-4', 'WET-4', 3, 2), ('WET-4', 'WET-3', 3, 2),
                    ('WET-3', 'WET-3', 2, 1), ]:
            wc = wcxf.WC(eft[0], 'flavio', eft[2], {'CVLL_sdsd': {'Im': 1}})
            ww = wilson.Wilson.from_wc(wc)
            wc_out = ww.match_run(eft[3], eft[1], 'flavio')
            wc_out.validate()

    def test_run_smeft(self):
        w = wilson.Wilson({'qd1_1123': 1}, 1000, 'SMEFT', 'Warsaw')
        wc = w.match_run(160, 'SMEFT', 'Warsaw up')
        wc.validate()        

 
    def test__translate_warsaw_to_warsawup(self):
        w_in = wilson.Wilson({'qd1_1211': 1e-6}, 1e3, 'SMEFT', 'Warsaw')
        wc_out = w_in.match_run(1e3, 'SMEFT', 'Warsaw up')
        V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["delta"])
        self.assertAlmostEqual(wc_out.dict['qd1_1111'], V[0, 1] * V[0, 0].conjugate() * 1e-6 + V[0, 0] * V[0, 1].conjugate() * 1e-6,places=11)
        self.assertAlmostEqual(wc_out.dict['qd1_1211'], V[0, 1] * V[1, 0].conjugate() * 1e-6 + V[0, 0] * V[1, 1].conjugate() * 1e-6,places=11)
        self.assertAlmostEqual(wc_out.dict['qd1_1311'], V[0, 1] * V[2, 0].conjugate() * 1e-6 + V[0, 0] * V[2, 1].conjugate() * 1e-6,places=11)
        self.assertAlmostEqual(wc_out.dict['qd1_2211'], V[1, 1] * V[1, 0].conjugate() * 1e-6 + V[1, 0] * V[1, 1].conjugate() * 1e-6,places=11)
        self.assertAlmostEqual(wc_out.dict['qd1_2311'], V[1, 1] * V[2, 0].conjugate() * 1e-6 + V[1, 0] * V[2, 1].conjugate() * 1e-6,places=11)
        self.assertAlmostEqual(wc_out.dict['qd1_3311'], V[2, 1] * V[2, 0].conjugate() * 1e-6 + V[2, 0] * V[2, 1].conjugate() * 1e-6,places=11)


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

class TestWilsonConfig(unittest.TestCase):
    def test_schema(self):
        for subclass in all_subclasses(wilson.classes.ConfigurableClass):
            # check that all options in schema have a default option
            self.assertEqual(set(subclass._option_schema.schema.keys()),
                             set(subclass._default_options.keys()))
            # check that default options pass validation
            subclass._option_schema(subclass._default_options)

    def test_config(self):
        wilson.Wilson._default_options['smeft_accuracy'] = 'leadinglog'
        w = wilson.Wilson({'qd1_1123': 1}, 1000, 'SMEFT', 'Warsaw')
        self.assertEqual(w.get_option('smeft_accuracy'), 'leadinglog')
        wilson.Wilson.set_default_option('smeft_accuracy', 'integrate')
        self.assertEqual(w.get_option('smeft_accuracy'), 'leadinglog')  # not changed!
        w2 = wilson.Wilson({'qd1_1123': 1}, 1000, 'SMEFT', 'Warsaw')
        self.assertEqual(w2.get_option('smeft_accuracy'), 'integrate')  # changed!
        w.set_option('smeft_accuracy', 'leadinglog')
        self.assertEqual(w.get_option('smeft_accuracy'), 'leadinglog')
        with self.assertRaises(KeyError):
            w.get_option('my_config_doesntexist')

    def test_config_parameters(self):
        w = wilson.Wilson({'qd1_1123': 1}, 1000, 'SMEFT', 'Warsaw')
        with self.assertRaises(vol.MultipleInvalid):
            # value must be dict
            w.set_option('parameters', 4)
        with self.assertRaises(vol.MultipleInvalid):
            # dict value must be number
            w.set_option('parameters', {'bla': 'blo'})
        # int should be OK but corced to float
        w.set_option('parameters', {'bla': 1})
        self.assertTrue(type(w.get_option('parameters')['bla']), float)
        self.assertEqual(w.get_option('parameters'), {'bla': 1.})
        w.set_option('parameters', {'m_b': 4.0})
        self.assertEqual(w.get_option('parameters'), {'m_b': 4.0})
        self.assertEqual(w.parameters['m_b'], 4.0)

    def test_clearcache(self):
        w = wilson.Wilson({'CVLL_sdsd': 1}, 160, 'WET', 'flavio')
        # after init, cache empty
        self.assertDictEqual(w._cache, {})
        # run
        w.match_run(140, 'WET', 'flavio')
        # now cache not empty
        self.assertIsInstance(w._cache['WET'][140]['flavio']['all'], wcxf.WC)
        w.clear_cache()
        # now cache empty again
        self.assertDictEqual(w._cache, {})
        # check that setting option empties cache
        wilson.Wilson._default_options['smeft_accuracy'] = 666
        w = wilson.Wilson({'CVLL_sdsd': 1}, 160, 'WET', 'flavio')
        w.match_run(140, 'WET', 'flavio')
        w.set_option('smeft_accuracy', 'leadinglog')
        self.assertDictEqual(w._cache, {})

    def test_smeft_matchingscale(self):
        w = wilson.Wilson({'lq1_2223': 1e-8}, 1000, 'SMEFT', 'Warsaw')
        w.set_option('smeft_accuracy', 'leadinglog')
        w.set_option('smeft_matchingscale', 145)
        w.set_option('mb_matchingscale', 4)
        w.set_option('mc_matchingscale', 2)
        w.match_run(80, 'WET', 'JMS')
        self.assertSetEqual(set(w._cache['WET'].keys()), {145, 80})
        w.match_run(1, 'WET-3', 'JMS')
        self.assertEqual(w.get_option('mb_matchingscale'), 4)
        self.assertEqual(w.get_option('mc_matchingscale'), 2)

    def test_smeft_matching_order(self):
        # with tree-level matching (and no SMEFT running): no effect
        w = wilson.Wilson({'uG_33': 1e-8}, 100, 'SMEFT', 'Warsaw')
        w.set_option('smeft_accuracy', 'leadinglog')
        w.set_option('smeft_matchingscale', 100)
        wc = w.match_run(5, 'WET', 'JMS')
        self.assertEqual(wc['dG_23'], 0)
        # now with 1-loop matching
        w = wilson.Wilson({'uG_33': 1e-8}, 100, 'SMEFT', 'Warsaw')
        w.set_option('smeft_accuracy', 'leadinglog')
        w.set_option('smeft_matchingscale', 100)
        w.set_option('smeft_matching_order', 1)
        wc = w.match_run(5, 'WET', 'JMS')
        self.assertNotEqual(wc['dG_23'], 0)