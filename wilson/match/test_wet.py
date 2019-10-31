import unittest
import numpy as np
from wilson import wcxf
import wilson

np.random.seed(321)


def get_random_wc(eft, basis, cmax=1e-2):
    """Generate a random Wilson coefficient instance for a given basis."""
    basis_obj = wcxf.Basis[eft, basis]
    _wc = {}
    for s in basis_obj.sectors.values():
        for name, d in s.items():
            _wc[name] = cmax * np.random.rand()
            if 'real' not in d or not d['real']:
                _wc[name] += 1j * cmax * np.random.rand()
    return wcxf.WC(eft, basis, 80., wcxf.WC.dict2values(_wc))


class TestWETflavio(unittest.TestCase):
    def test_wet_wet4(self):
        from_wc = get_random_wc('WET', 'flavio')
        to_wc = from_wc.match('WET-4', 'flavio')
        to_wc.validate()
        fkeys = set(to_wc.values.keys())
        fkeys_all = set(wcxf.Basis['WET-4', 'flavio'].all_wcs)
        self.assertSetEqual(fkeys_all - fkeys, set(), msg="Missing coefficients")
        for k, v in to_wc.dict.items():
            self.assertEqual(v, from_wc.dict[k], msg="Failed for {}".format(k))

    def test_wet4_wet3(self):
        from_wc = get_random_wc('WET-4', 'flavio')
        to_wc = from_wc.match('WET-3', 'flavio')
        to_wc.validate()
        fkeys = set(to_wc.values.keys())
        fkeys_all = set(wcxf.Basis['WET-3', 'flavio'].all_wcs)
        self.assertSetEqual(fkeys_all - fkeys, set(), msg="Missing coefficients")
        for k, v in to_wc.dict.items():
            self.assertEqual(v, from_wc.dict[k], msg="Failed for {}".format(k))


class TestWETBern(unittest.TestCase):
    def test_wet_wet4(self):
        from_wc = get_random_wc('WET', 'Bern')
        to_wc = from_wc.match('WET-4', 'Bern')
        to_wc.validate()
        fkeys = set(to_wc.values.keys())
        fkeys_all = set(wcxf.Basis['WET-4', 'Bern'].all_wcs)
        self.assertSetEqual(fkeys_all - fkeys, set(), msg="Missing coefficients")
        for k, v in to_wc.dict.items():
            self.assertEqual(v, from_wc.dict[k], msg="Failed for {}".format(k))

    def test_wet4_wet3(self):
        from_wc = get_random_wc('WET-4', 'Bern')
        to_wc = from_wc.match('WET-3', 'Bern')
        to_wc.validate()
        fkeys = set(to_wc.values.keys())
        fkeys_all = set(wcxf.Basis['WET-3', 'Bern'].all_wcs)
        self.assertSetEqual(fkeys_all - fkeys, set(), msg="Missing coefficients")
        for k, v in to_wc.dict.items():
            self.assertEqual(v, from_wc.dict[k], msg="Failed for {}".format(k))


class TestWETJMS(unittest.TestCase):
    def test_wet_wet4(self):
        from_wc = get_random_wc('WET', 'JMS')
        to_wc = from_wc.match('WET-4', 'JMS')
        to_wc.validate()
        fkeys = set(to_wc.values.keys())
        fkeys_all = set(wcxf.Basis['WET-4', 'JMS'].all_wcs)
        self.assertSetEqual(fkeys_all - fkeys, set(), msg="Missing coefficients")
        for k, v in to_wc.dict.items():
            self.assertEqual(v, from_wc.dict[k], msg="Failed for {}".format(k))

    def test_wet4_wet3(self):
        from_wc = get_random_wc('WET-4', 'JMS')
        to_wc = from_wc.match('WET-3', 'JMS')
        to_wc.validate()
        fkeys = set(to_wc.values.keys())
        fkeys_all = set(wcxf.Basis['WET-3', 'JMS'].all_wcs)
        self.assertSetEqual(fkeys_all - fkeys, set(), msg="Missing coefficients")
        for k, v in to_wc.dict.items():
            self.assertEqual(v, from_wc.dict[k], msg="Failed for {}".format(k))

    def test_wet3_wet2(self):
        from_wc = get_random_wc('WET-3', 'JMS')
        to_wc = from_wc.match('WET-2', 'JMS')
        to_wc.validate()
        fkeys = set(to_wc.values.keys())
        fkeys_all = set(wcxf.Basis['WET-2', 'JMS'].all_wcs)
        self.assertSetEqual(fkeys_all - fkeys, set(), msg="Missing coefficients")
        for k, v in to_wc.dict.items():
            self.assertEqual(v, from_wc.dict[k], msg="Failed for {}".format(k))
