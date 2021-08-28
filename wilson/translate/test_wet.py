import unittest
import numpy as np
from wilson import wcxf
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


def _test_missing(self, wc, from_basis, ignore_sec=[], ignore_coeffs=[]):
    to_keys = set(wc.values.keys())
    for sec, coeffs in wcxf.Basis[wc.eft, wc.basis].sectors.items():
        if sec in ignore_sec or sec not in wcxf.Basis[wc.eft, from_basis].sectors:
            continue
        to_keys_all = set(coeffs.keys())
        self.assertSetEqual(to_keys_all - to_keys - set(ignore_coeffs), set(),
                            msg=f"Missing coefficients in sector {sec}")


class TestJMS2Flavio(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        jms_wc = get_random_wc('WET', 'JMS')
        cls.flavio_wc = jms_wc.translate('flavio')

    def test_validate(self):
        self.flavio_wc.validate()

    def test_nan(self):
        for k, v in self.flavio_wc.dict.items():
            self.assertFalse(np.isnan(v), msg=f"{k} is NaN")

    def test_missing(self):
        fkeys = set(self.flavio_wc.values.keys())
        fkeys_all = {k for s in wcxf.Basis['WET', 'flavio'].sectors.values()
                         for k in s}
        self.assertSetEqual(fkeys_all - fkeys, set(), msg="Missing coefficients")

    def test_incomplete_input(self):
        # generate and input WC instance with just 1 non-zero coeff.
        jms_wc = wcxf.WC('WET', 'JMS', 80, {'VddRR_2323': {'Im': -1}})
        flavio_wc = jms_wc.translate('flavio')
        flavio_wc.validate()
        # the output WC instance should contain only one as well
        self.assertEqual(list(flavio_wc.dict.keys()), ['CVRR_bsbs'])

    def test_sectors(self):
        jms_wc = get_random_wc('WET', 'JMS')
        sectors = wcxf.Basis['WET', 'flavio'].sectors.keys()
        flavio_wc_1 = jms_wc.translate('flavio')
        flavio_wc_2 = jms_wc.translate('flavio', sectors=None)
        flavio_wc_3 = jms_wc.translate('flavio', sectors=sectors)
        self.assertDictEqual(flavio_wc_1.dict, flavio_wc_2.dict)
        self.assertDictEqual(flavio_wc_1.dict, flavio_wc_3.dict)


class TestJMS2FlavioWET3(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        jms_wc = get_random_wc('WET-3', 'JMS')
        cls.flavio_wc = jms_wc.translate('flavio')

    def test_validate(self):
        self.flavio_wc.validate()

    def test_nan(self):
        for k, v in self.flavio_wc.dict.items():
            self.assertFalse(np.isnan(v), msg=f"{k} is NaN")

    def test_missing(self):
        fkeys = set(self.flavio_wc.values.keys())
        fkeys_all = {k for s in wcxf.Basis['WET-3', 'flavio'].sectors.values()
                         for k in s}
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
            self.assertFalse(np.isnan(v), msg=f"{k} is NaN")

    def test_missing(self):
        bkeys = set(self.bern_wc.values.keys())
        bkeys_all = {k for sname, s in wcxf.Basis['WET', 'Bern'].sectors.items()
                         for k in s
                         if sname not in ['dF=0', ]}
        self.assertSetEqual(bkeys_all - bkeys, set(), msg="Missing coefficients")


class TestBern2JMS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.from_wc = get_random_wc('WET', 'Bern')
        cls.to_wc = cls.from_wc.translate('JMS')

    def test_validate(self):
        self.to_wc.validate()

    def test_nan(self):
        for k, v in self.to_wc.dict.items():
            self.assertFalse(np.isnan(v), msg=f"{k} is NaN")

    def test_missing(self):
        _test_missing(self, self.to_wc, 'Bern',
                      ignore_sec=('dF=0',))


    def test_roundtrip(self):
        round_wc = self.to_wc.translate('Bern')
        for k, v in round_wc.dict.items():
            self.assertAlmostEqual(v, self.from_wc.dict[k],
                                   delta=1e-12,
                                   msg=f"Failed for {k}")


class TestFlavio2JMS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.from_wc = get_random_wc('WET', 'flavio')
        cls.to_wc = cls.from_wc.translate('JMS')

    def test_validate(self):
        self.to_wc.validate()

    def test_nan(self):
        for k, v in self.to_wc.dict.items():
            self.assertFalse(np.isnan(v), msg=f"{k} is NaN")

    def test_missing(self):
        # ignore tensor coeffs
        ignore = [f'TedRR_{i}{j}{k}{l}'
                  for i in '123'
                  for j in '123'
                  for k in '123'
                  for l in '123'
                  ] + [f'TeuRR_{i}{j}{k}{l}'
                            for i in '123'
                            for j in '123'
                            for k in '123'
                            for l in '123']
        _test_missing(self, self.to_wc, 'flavio',
                      ignore_coeffs=ignore,
                      ignore_sec=('dF=0',))

    def test_roundtrip(self):
        round_wc = self.to_wc.translate('flavio')
        for k, v in round_wc.dict.items():
            self.assertAlmostEqual(v, self.from_wc.dict[k],
                                   delta=1e-12,
                                   msg=f"Failed for {k}")


class TestJMS2BernWET3(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        jms_wc = get_random_wc('WET-3', 'JMS')
        cls.bern_wc = jms_wc.translate('Bern')

    def test_validate(self):
        self.bern_wc.validate()

    def test_nan(self):
        for k, v in self.bern_wc.dict.items():
            self.assertFalse(np.isnan(v), msg=f"{k} is NaN")

    def test_missing(self):
        bkeys = set(self.bern_wc.values.keys())
        bkeys_all = {k for s in wcxf.Basis['WET-3', 'Bern'].sectors.values()
                         for k in s
                         if 'b' in k} # for the time being, only look at b operators
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
            self.assertFalse(np.isnan(v), msg=f"{k} is NaN")

    def test_missing(self):
        fkeys = set(self.eos_wc.values.keys())
        fkeys_all = {k for s in wcxf.Basis['WET', 'EOS'].sectors.values()
                         for k in s}
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
            self.assertFalse(np.isnan(v), msg=f"{k} is NaN")

    def test_missing(self):
        fkeys = set(self.formflavor_wc.values.keys())
        fkeys_all = {k for s in wcxf.Basis['WET', 'formflavor'].sectors.values()
                         for k in s}
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
            self.assertFalse(np.isnan(v), msg=f"{k} is NaN")

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
            self.assertFalse(np.isnan(v), msg=f"{k} is NaN")

    def count_nonzero(self):
        self.assertEqual(len(self.fk_wc.dict), len(self.fk_wc.dict))

    def test_missing(self):
        fkeys = set(self.fk_wc.values.keys())
        fkeys_all = {k for s in wcxf.Basis['WET', 'FlavorKit'].sectors.values()
                         for k in s}
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
            self.assertFalse(np.isnan(v), msg=f"{k} is NaN")

    def count_nonzero(self):
        self.assertEqual(len(self.to_wc.dict), len(self.to_wc.dict))

    def test_roundtrip(self):
        round_wc = self.to_wc.translate('Bern')
        for k, v in round_wc.dict.items():
            if k[0] != '7':  # to avoid problem with flavio tensors missing
                self.assertAlmostEqual(v, self.from_wc.dict[k],
                                       delta=1e-12,
                                       msg=f"Failed for {k}")

    def test_jms(self):
        jms_wc = get_random_wc('WET', 'JMS')
        jms_direct = jms_wc.translate('flavio')
        jms_indirect = jms_wc.translate('Bern').translate('flavio')
        for k, v in jms_direct.dict.items():
            if k in jms_indirect.dict:
                self.assertAlmostEqual(v, jms_indirect.dict[k],
                                       delta=1e-5,
                                       msg=f"Failed for {k}")

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
        fkeys_all = {k for sname, s in wcxf.Basis['WET', 'flavio'].sectors.items()
                         for k in s
                         if sname not in ['mue', 'mutau', 'taue', 'nunumue', 'nunumutau', 'nunutaue', 'dF=0', 'ffnunu', 'etauemu', 'muemutau', 'cu']}  # LFV, dF=0, dC=1 not in Bern
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
            self.assertFalse(np.isnan(v), msg=f"{k} is NaN")

    def count_nonzero(self):
        self.assertEqual(len(self.to_wc.dict), len(self.to_wc.dict))

    def test_roundtrip(self):
        round_wc = self.to_wc.translate('flavio')
        for k, v in round_wc.dict.items():
            self.assertAlmostEqual(v, self.from_wc.dict[k],
                                   delta=1e-12,
                                   msg=f"Failed for {k} {v}")

    def test_jms(self):
        jms_wc = get_random_wc('WET', 'JMS')
        jms_direct = jms_wc.translate('Bern')
        jms_indirect = jms_wc.translate('flavio').translate('Bern')
        for k, v in jms_direct.dict.items():
            if k in jms_indirect.dict:  # since flavio misses 4Q operators
                if k[0] != '7':  # to avoid problem with flavio tensors missing
                    self.assertAlmostEqual(v, jms_indirect.dict[k],
                                           delta=1e-8,
                                           msg=f"Failed for {k}")


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
            self.assertFalse(np.isnan(v), msg=f"{k} is NaN")

    def count_nonzero(self):
        self.assertEqual(len(self.to_wc.dict), len(self.to_wc.dict))

    def test_roundtrip(self):
        round_wc = self.to_wc.translate('Bern')
        for k, v in round_wc.dict.items():
            if k[0] != '7':  # to avoid problem with flavio tensors missing
                self.assertAlmostEqual(v, self.from_wc.dict[k],
                                       delta=1e-12,
                                       msg=f"Failed for {k}")

    def test_jms(self):
        jms_wc = get_random_wc('WET-3', 'JMS')
        jms_direct = jms_wc.translate('flavio')
        jms_indirect = jms_wc.translate('Bern').translate('flavio')
        for k, v in jms_direct.dict.items():
            if k in jms_indirect.dict:
                self.assertAlmostEqual(v, jms_indirect.dict[k],
                                       delta=1e-4,
                                       msg=f"Failed for {k}")

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
        fkeys_all = {k for sname, s in wcxf.Basis['WET-3', 'flavio'].sectors.items()
                         for k in s
                         if sname not in ['mue', 'nunumue', 'dF=0', 'ffnunu']}  # LFV, dF=0 not in Bern
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
            self.assertFalse(np.isnan(v), msg=f"{k} is NaN")

    def count_nonzero(self):
        self.assertEqual(len(self.to_wc.dict), len(self.to_wc.dict))

    def test_roundtrip(self):
        round_wc = self.to_wc.translate('flavio')
        for k, v in round_wc.dict.items():
            self.assertAlmostEqual(v, self.from_wc.dict[k],
                                   delta=1e-12,
                                   msg=f"Failed for {k} {v}")

    def test_jms(self):
        jms_wc = get_random_wc('WET-3', 'JMS')
        jms_direct = jms_wc.translate('Bern')
        jms_indirect = jms_wc.translate('flavio').translate('Bern')
        for k, v in jms_direct.dict.items():
            if k in jms_indirect.dict:  # since flavio misses 4Q operators
                if k[0] != '7':  # to avoid problem with flavio tensors missing
                    self.assertAlmostEqual(v, jms_indirect.dict[k],
                                           delta=1e-8,
                                           msg=f"Failed for {k}")


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
                             msg=f"Failed for {k}")
