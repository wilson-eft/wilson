import unittest
import numpy as np
import numpy.testing as npt
import wcxf
from . import wet
from math import sqrt
import wilson

np.random.seed(87)

def get_random_wc(eft, basis, cmax=1e-2):
    """Generate a random Wilson coefficient instance for a given basis."""
    basis_obj = wcxf.Basis[eft, basis]
    _wc = {}
    for s in basis_obj.sectors.values():
        for name, d in s.items():
            _wc[name] = cmax*np.random.rand()
            if 'real' not in d or not d['real']:
                _wc[name] += 1j*cmax*np.random.rand()
    return wcxf.WC(eft, basis, 80., wcxf.WC.dict2values(_wc))


class TestTranslateWET(unittest.TestCase):
    def test_scalar2array(self):
        d = {'bla_123': 3, 'blo': 5j}
        da = wet._scalar2array(d)
        self.assertEqual(da['blo'], 5j)
        self.assertIn('bla', da)
        self.assertEqual(da['bla'].shape, (3, 3, 3))
        self.assertTrue(np.isnan(da['bla'][0, 0, 0]))
        self.assertEqual(da['bla'][0, 1, 2], 3)


class TestJMS2Flavio(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        jms_wc = get_random_wc('WET', 'JMS')
        cls.flavio_wc = jms_wc.translate('flavio')

    def test_validate(self):
        self.flavio_wc.validate()

    def test_nan(self):
        for k, v in self.flavio_wc.dict.items():
            self.assertFalse(np.isnan(v), msg="{} is NaN".format(k))

    def test_missing(self):
        fkeys = set(self.flavio_wc.values.keys())
        fkeys_all = set([k for s in wcxf.Basis['WET', 'flavio'].sectors.values()
                         for k in s])
        self.assertSetEqual(fkeys_all - fkeys, set(), msg="Missing coefficients")

    def test_incomplete_input(self):
        # generate and input WC instance with just 1 non-zero coeff.
        jms_wc = wcxf.WC('WET', 'JMS', 80, {'VddRR_2323': {'Im': -1}})
        flavio_wc = jms_wc.translate('flavio')
        flavio_wc.validate()
        # the output WC instance should contain only one as well
        self.assertEqual(list(flavio_wc.dict.keys()), ['CVRR_bsbs'])


class TestJMS2FlavioWET3(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        jms_wc = get_random_wc('WET-3', 'JMS')
        cls.flavio_wc = jms_wc.translate('flavio')

    def test_validate(self):
        self.flavio_wc.validate()

    def test_nan(self):
        for k, v in self.flavio_wc.dict.items():
            self.assertFalse(np.isnan(v), msg="{} is NaN".format(k))

    def test_missing(self):
        fkeys = set(self.flavio_wc.values.keys())
        fkeys_all = set([k for s in wcxf.Basis['WET-3', 'flavio'].sectors.values()
                         for k in s])
        self.assertSetEqual(fkeys_all - fkeys, set(), msg="Missing coefficients")

    def test_incomplete_input(self):
        # generate and input WC instance with just 1 non-zero coeff.
        jms_wc = wcxf.WC('WET', 'JMS', 80, {'VddRR_1212': {'Im': -1}})
        flavio_wc = jms_wc.translate('flavio')
        flavio_wc.validate()
        # the output WC instance should contain only one as well
        self.assertEqual(list(flavio_wc.dict.keys()), ['CVRR_sdsd'])


class TestJMS2Bern(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        jms_wc = get_random_wc('WET', 'JMS')
        cls.bern_wc = jms_wc.translate('Bern')

    def test_validate(self):
        self.bern_wc.validate()

    def test_nan(self):
        for k, v in self.bern_wc.dict.items():
            self.assertFalse(np.isnan(v), msg="{} is NaN".format(k))

    def test_missing(self):
        bkeys = set(self.bern_wc.values.keys())
        bkeys_all = set([k for s in wcxf.Basis['WET', 'Bern'].sectors.values()
                         for k in s])
        self.assertSetEqual(bkeys_all - bkeys, set(), msg="Missing coefficients")


class TestBern2JMS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.from_wc = get_random_wc('WET', 'Bern')
        cls.to_wc = cls.from_wc.translate('JMS')
        # TODO!
        cls.classes_implemented = ['I', 'Iu', 'II',]
        cls.sectors_implemented = [s for ci in cls.classes_implemented
                                   for s in wilson.run.wet.definitions.classes[ci]]

    def test_validate(self):
        self.to_wc.validate()

    def test_nan(self):
        for k, v in self.to_wc.dict.items():
            self.assertFalse(np.isnan(v), msg="{} is NaN".format(k))

    def test_missing(self):
        to_keys = set(self.to_wc.values.keys())
        to_keys_all = set([k for s in wcxf.Basis['WET', 'Bern'].sectors.values()
                         for k in s
                         if s in self.sectors_implemented])
        self.assertSetEqual(to_keys_all - to_keys, set(), msg="Missing coefficients")

    def test_roundtrip(self):
        round_wc = self.to_wc.translate('Bern')
        for k, v in round_wc.dict.items():
            self.assertAlmostEqual(v, self.from_wc.dict[k],
                                   delta=1e-12,
                                   msg="Failed for {}".format(k))


class TestFlavio2JMS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.from_wc = get_random_wc('WET', 'flavio')
        cls.to_wc = cls.from_wc.translate('JMS')
        # TODO!
        cls.classes_implemented = ['I', 'Iu', 'II',]
        cls.sectors_implemented = [s for ci in cls.classes_implemented
                                   for s in wilson.run.wet.definitions.classes[ci]]

    def test_validate(self):
        self.to_wc.validate()

    def test_nan(self):
        for k, v in self.to_wc.dict.items():
            self.assertFalse(np.isnan(v), msg="{} is NaN".format(k))

    def test_missing(self):
        to_keys = set(self.to_wc.values.keys())
        to_keys_all = set([k for s in wcxf.Basis['WET', 'flavio'].sectors.values()
                         for k in s
                         if s in self.sectors_implemented])
        self.assertSetEqual(to_keys_all - to_keys, set(), msg="Missing coefficients")

    def test_roundtrip(self):
        round_wc = self.to_wc.translate('flavio')
        for k, v in round_wc.dict.items():
            self.assertAlmostEqual(v, self.from_wc.dict[k],
                                   delta=1e-12,
                                   msg="Failed for {}".format(k))


class TestJMS2BernWET3(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        jms_wc = get_random_wc('WET-3', 'JMS')
        cls.bern_wc = jms_wc.translate('Bern')

    def test_validate(self):
        self.bern_wc.validate()

    def test_nan(self):
        for k, v in self.bern_wc.dict.items():
            self.assertFalse(np.isnan(v), msg="{} is NaN".format(k))

    def test_missing(self):
        bkeys = set(self.bern_wc.values.keys())
        bkeys_all = set([k for s in wcxf.Basis['WET-3', 'Bern'].sectors.values()
                         for k in s
                         if 'b' in k]) # for the time being, only look at b operators
        self.assertSetEqual(bkeys_all - bkeys, set(), msg="Missing coefficients")


class TestJMS2EOS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        jms_wc = get_random_wc('WET', 'JMS')
        cls.eos_wc = jms_wc.translate('EOS')

    def test_validate(self):
        self.eos_wc.validate()

    def test_nan(self):
        for k, v in self.eos_wc.dict.items():
            self.assertFalse(np.isnan(v), msg="{} is NaN".format(k))

    def test_missing(self):
        fkeys = set(self.eos_wc.values.keys())
        fkeys_all = set([k for s in wcxf.Basis['WET', 'EOS'].sectors.values()
                         for k in s])
        self.assertSetEqual(fkeys_all - fkeys, set(), msg="Missing coefficients")

class TestJMS2FormFlavor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        jms_wc = get_random_wc('WET', 'JMS')
        cls.formflavor_wc = jms_wc.translate('formflavor')

    def test_validate(self):
        self.formflavor_wc.validate()

    def test_nan(self):
        for k, v in self.formflavor_wc.dict.items():
            self.assertFalse(np.isnan(v), msg="{} is NaN".format(k))

    def test_missing(self):
        fkeys = set(self.formflavor_wc.values.keys())
        fkeys_all = set([k for s in wcxf.Basis['WET', 'formflavor'].sectors.values()
                         for k in s])
        self.assertSetEqual(fkeys_all - fkeys, set(), msg="Missing coefficients")


class TestFlavorKit2JMS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fk_wc = get_random_wc('WET', 'FlavorKit')
        cls.jms_wc = cls.fk_wc.translate('JMS')

    def test_validate(self):
        self.jms_wc.validate()

    def test_nan(self):
        for k, v in self.jms_wc.dict.items():
            self.assertFalse(np.isnan(v), msg="{} is NaN".format(k))

    def count_nonzero(self):
        self.assertEqual(len(self.jms_wc.dict), len(self.fk_wc.dict))

    def test_incomplete_input(self):
        # generate and input WC instance with just 1 non-zero coeff.
        jms_wc = wcxf.WC('WET', 'FlavorKit', 80, {'DVLL_2323': {'Im': -1}})
        to_wc = jms_wc.translate('JMS')
        to_wc.validate()
        # the output WC instance should contain only one as well
        self.assertEqual(list(to_wc.dict.keys()), ['VddLL_2323'])
        self.assertAlmostEqual(to_wc.dict['VddLL_2323'], +1j)


class TestJMS2FlavorKit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jms_wc = get_random_wc('WET', 'JMS')
        cls.fk_wc = jms_wc.translate('FlavorKit')

    def test_validate(self):
        self.fk_wc.validate()

    def test_nan(self):
        for k, v in self.fk_wc.dict.items():
            self.assertFalse(np.isnan(v), msg="{} is NaN".format(k))

    def count_nonzero(self):
        self.assertEqual(len(self.fk_wc.dict), len(self.fk_wc.dict))

    def test_missing(self):
        fkeys = set(self.fk_wc.values.keys())
        fkeys_all = set([k for s in wcxf.Basis['WET', 'FlavorKit'].sectors.values()
                         for k in s])
        self.assertSetEqual(fkeys_all - fkeys, set(), msg="Missing coefficients")

    def test_incomplete_input(self):
        # generate and input WC instance with just 1 non-zero coeff.
        jms_wc = wcxf.WC('WET', 'JMS', 80, {'VddLL_2323': {'Im': -1}})
        to_wc = jms_wc.translate('FlavorKit')
        to_wc.validate()
        # the output WC instance should contain only one as well
        self.assertEqual(list(to_wc.dict.keys()), ['DVLL_2323'])
        self.assertAlmostEqual(to_wc.dict['DVLL_2323'], +1j)


class TestBern2flavio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.from_wc = get_random_wc('WET', 'Bern')
        cls.to_wc = cls.from_wc.translate('flavio')

    def test_validate(self):
        self.to_wc.validate()

    def test_nan(self):
        for k, v in self.to_wc.dict.items():
            self.assertFalse(np.isnan(v), msg="{} is NaN".format(k))

    def count_nonzero(self):
        self.assertEqual(len(self.to_wc.dict), len(self.to_wc.dict))

    def test_roundtrip(self):
        round_wc = self.to_wc.translate('Bern')
        for k, v in round_wc.dict.items():
            if k[0] != '7':  # to avoid problem with flavio tensors missing
                self.assertAlmostEqual(v, self.from_wc.dict[k],
                                       delta=1e-12,
                                       msg="Failed for {}".format(k))

    def test_jms(self):
        jms_wc = get_random_wc('WET', 'JMS')
        jms_direct = jms_wc.translate('flavio')
        jms_indirect = jms_wc.translate('Bern').translate('flavio')
        for k, v in jms_direct.dict.items():
            self.assertAlmostEqual(v, jms_indirect.dict[k],
                                   delta=1e-6,
                                   msg="Failed for {}".format(k))

    def test_incomplete_input(self):
        # generate and input WC instance with just 1 non-zero coeff.
        jms_wc = wcxf.WC('WET', 'Bern', 80, {'1sbsb': {'Im': -1}})
        flavio_wc = jms_wc.translate('flavio')
        flavio_wc.validate()
        # the output WC instance should contain only one as well
        self.assertEqual(list(flavio_wc.dict.keys()), ['CVLL_bsbs'])
        GF = wilson.parameters.p['GF']
        pre = 4*GF/sqrt(2)
        self.assertAlmostEqual(flavio_wc.dict['CVLL_bsbs'], -1j * pre)

    def test_missing(self):
        fkeys = set(self.to_wc.values.keys())
        fkeys_all = set([k for s in wcxf.Basis['WET', 'flavio'].sectors.values()
                         for k in s])
        self.assertSetEqual(fkeys_all - fkeys, set(), msg="Missing coefficients")

class Testflavio2Bern(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.from_wc = get_random_wc('WET', 'flavio')
        cls.to_wc = cls.from_wc.translate('Bern')

    def test_validate(self):
        self.to_wc.validate()

    def test_nan(self):
        for k, v in self.to_wc.dict.items():
            self.assertFalse(np.isnan(v), msg="{} is NaN".format(k))

    def count_nonzero(self):
        self.assertEqual(len(self.to_wc.dict), len(self.to_wc.dict))

    def test_roundtrip(self):
        round_wc = self.to_wc.translate('flavio')
        for k, v in round_wc.dict.items():
            self.assertAlmostEqual(v, self.from_wc.dict[k],
                                   delta=1e-12,
                                   msg="Failed for {} {}".format(k, v))

    def test_jms(self):
        jms_wc = get_random_wc('WET', 'JMS')
        jms_direct = jms_wc.translate('Bern')
        jms_indirect = jms_wc.translate('flavio').translate('Bern')
        for k, v in jms_direct.dict.items():
            if k in jms_indirect.dict:  # since flavio misses 4Q operators
                if k[0] != '7':  # to avoid problem with flavio tensors missing
                    self.assertAlmostEqual(v, jms_indirect.dict[k],
                                           delta=1e-8,
                                           msg="Failed for {}".format(k))


    def test_incomplete_input(self):
        # generate and input WC instance with just 1 non-zero coeff.
        jms_wc = wcxf.WC('WET', 'flavio', 80, {'CVLL_bsbs': {'Im': -1e-6}})
        bern_wc = jms_wc.translate('Bern')
        bern_wc.validate()
        # the output WC instance should contain only one as well
        self.assertEqual(list(bern_wc.dict.keys()), ['1sbsb'])
        GF = wilson.parameters.p['GF']
        pre = 4*GF/sqrt(2)
        self.assertAlmostEqual(bern_wc.dict['1sbsb'], -1e-6j / pre)


class TestBern2flavioWET3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.from_wc = get_random_wc('WET-3', 'Bern')
        cls.to_wc = cls.from_wc.translate('flavio')

    def test_validate(self):
        self.to_wc.validate()

    def test_nan(self):
        for k, v in self.to_wc.dict.items():
            self.assertFalse(np.isnan(v), msg="{} is NaN".format(k))

    def count_nonzero(self):
        self.assertEqual(len(self.to_wc.dict), len(self.to_wc.dict))

    def test_roundtrip(self):
        round_wc = self.to_wc.translate('Bern')
        for k, v in round_wc.dict.items():
            if k[0] != '7':  # to avoid problem with flavio tensors missing
                self.assertAlmostEqual(v, self.from_wc.dict[k],
                                       delta=1e-12,
                                       msg="Failed for {}".format(k))

    def test_jms(self):
        jms_wc = get_random_wc('WET-3', 'JMS')
        jms_direct = jms_wc.translate('flavio')
        jms_indirect = jms_wc.translate('Bern').translate('flavio')
        for k, v in jms_direct.dict.items():
            self.assertAlmostEqual(v, jms_indirect.dict[k],
                                   delta=1e-6,
                                   msg="Failed for {}".format(k))

    def test_incomplete_input(self):
        # generate and input WC instance with just 1 non-zero coeff.
        jms_wc = wcxf.WC('WET-3', 'Bern', 80, {'1dsds': {'Im': -1}})
        flavio_wc = jms_wc.translate('flavio')
        flavio_wc.validate()
        # the output WC instance should contain only one as well
        self.assertEqual(list(flavio_wc.dict.keys()), ['CVLL_sdsd'])
        GF = wilson.parameters.p['GF']
        pre = 4*GF/sqrt(2)
        self.assertAlmostEqual(flavio_wc.dict['CVLL_sdsd'], -1j * pre)

    def test_missing(self):
        fkeys = set(self.to_wc.values.keys())
        fkeys_all = set([k for s in wcxf.Basis['WET-3', 'flavio'].sectors.values()
                         for k in s])
        self.assertSetEqual(fkeys_all - fkeys, set(), msg="Missing coefficients")

class Testflavio2BernWET3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.from_wc = get_random_wc('WET-3', 'flavio')
        cls.to_wc = cls.from_wc.translate('Bern')

    def test_validate(self):
        self.to_wc.validate()

    def test_nan(self):
        for k, v in self.to_wc.dict.items():
            self.assertFalse(np.isnan(v), msg="{} is NaN".format(k))

    def count_nonzero(self):
        self.assertEqual(len(self.to_wc.dict), len(self.to_wc.dict))

    def test_roundtrip(self):
        round_wc = self.to_wc.translate('flavio')
        for k, v in round_wc.dict.items():
            self.assertAlmostEqual(v, self.from_wc.dict[k],
                                   delta=1e-12,
                                   msg="Failed for {} {}".format(k, v))

    def test_jms(self):
        jms_wc = get_random_wc('WET-3', 'JMS')
        jms_direct = jms_wc.translate('Bern')
        jms_indirect = jms_wc.translate('flavio').translate('Bern')
        for k, v in jms_direct.dict.items():
            if k in jms_indirect.dict:  # since flavio misses 4Q operators
                if k[0] != '7':  # to avoid problem with flavio tensors missing
                    self.assertAlmostEqual(v, jms_indirect.dict[k],
                                           delta=1e-8,
                                           msg="Failed for {}".format(k))


    def test_incomplete_input(self):
        # generate and input WC instance with just 1 non-zero coeff.
        jms_wc = wcxf.WC('WET-3', 'flavio', 80, {'CVLL_sdsd': {'Im': -1e-6}})
        bern_wc = jms_wc.translate('Bern')
        bern_wc.validate()
        # the output WC instance should contain only one as well
        self.assertEqual(list(bern_wc.dict.keys()), ['1dsds'])
        GF = wilson.parameters.p['GF']
        pre = 4*GF/sqrt(2)
        self.assertAlmostEqual(bern_wc.dict['1dsds'], -1e-6j / pre)


class TestFlavorKit2flavio(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.from_wc = get_random_wc('WET', 'FlavorKit')
        cls.to_wc = cls.from_wc.translate('flavio')

    def test_validate(self):
        self.to_wc.validate()

    def test_detour(self):
        to_wc_2 = self.from_wc.translate('JMS').translate('flavio')
        for k, v in self.to_wc.dict.items():
            self.assertEqual(v, to_wc_2.dict[k],
                             msg="Failed for {}".format(k))
