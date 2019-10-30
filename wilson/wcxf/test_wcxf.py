import unittest
import numpy as np
import numpy.testing as npt
import yaml
import json
import pkgutil
from wilson import wcxf
from wcxf import translators

class TestBasis(unittest.TestCase):
    def test_eft(self):
        f = pkgutil.get_data('wcxf', 'data/test.eft.yml')
        eft = wcxf.EFT.load(f.decode('utf-8'))
        self.assertEqual(eft, wcxf.EFT['MyEFT'])
        self.assertEqual(eft.eft, 'MyEFT')
        self.assertIsInstance(eft.sectors, dict)
        f = pkgutil.get_data('wcxf', 'data/test.basis1.yml')
        basis = wcxf.Basis.load(f.decode('utf-8'))
        self.assertEqual(eft.known_bases, ('MyBasis 1',))
        self.assertIsInstance(eft.dump(fmt='json'), str)
        self.assertIsInstance(eft.dump(fmt='yaml'), str)

    def test_basis(self):
        f = pkgutil.get_data('wcxf', 'data/test.eft.yml')
        eft = wcxf.EFT.load(f.decode('utf-8'))
        f = pkgutil.get_data('wcxf', 'data/test.basis1.yml')
        basis = wcxf.Basis.load(f.decode('utf-8'))
        self.assertEqual(basis, wcxf.Basis['MyEFT', 'MyBasis 1'])
        self.assertEqual(basis._name, ('MyEFT', 'MyBasis 1'))
        self.assertEqual(basis.basis, 'MyBasis 1')
        basis.validate()

    def test_wc(self):
        f = pkgutil.get_data('wcxf', 'data/test.eft.yml')
        eft = wcxf.EFT.load(f.decode('utf-8'))
        f = pkgutil.get_data('wcxf', 'data/test.basis1.yml')
        basis = wcxf.Basis.load(f.decode('utf-8'))
        f = pkgutil.get_data('wcxf', 'data/test.wcs.yml')
        wc = wcxf.WC.load(f.decode('utf-8'))
        self.assertEqual(wc.eft, 'MyEFT')
        self.assertEqual(wc.basis, 'MyBasis 1')
        wc.validate()
        self.assertEqual(wc.scale, 1e16)
        self.assertEqual(wc.dict['C_1'], 0.12)
        self.assertEqual(wc.dict['C_2'], 0.3156-0.53j)

    def test_translator(self):
        # A trivial translator translating from MyBasis 1 to MyBasis 2
        @wcxf.translator('MyEFT', 'MyBasis 1', 'MyBasis 2')
        def f(x, scale, parameters):
            return x
        self.assertIn(('MyEFT', 'MyBasis 1', 'MyBasis 2'), wcxf.Translator.instances)
        f = pkgutil.get_data('wcxf', 'data/test.wcs.yml')
        wc = wcxf.WC.load(f.decode('utf-8'))
        wc_out = wcxf.Translator['MyEFT', 'MyBasis 1', 'MyBasis 2'].translate(wc)
        self.assertDictEqual(
            {k: v for k, v in wc.__dict__.items() if k[0] != '_' and k != 'basis'},
            {k: v for k, v in wc_out.__dict__.items() if k[0] != '_' and k != 'basis'}
        )
        self.assertIn(('MyEFT', 'MyBasis 1', 'MyBasis 2'),
                      wcxf.Basis['MyEFT', 'MyBasis 1'].known_translators['from'])
        self.assertIn(('MyEFT', 'MyBasis 1', 'MyBasis 2'),
                      wcxf.Basis['MyEFT', 'MyBasis 2'].known_translators['to'])
        self.assertIn(('MyEFT', 'MyBasis 1', 'MyBasis 2'),
                      wcxf.EFT['MyEFT'].known_translators)
        # remove dummy translator
        del wcxf.Translator['MyEFT', 'MyBasis 1', 'MyBasis 2']

    def test_matcher(self):
        # A trivial translator translating from MyBasis 1 to MyBasis 2
        @wcxf.matcher('MyEFT', 'MyBasis 1', 'MyOtherEFT', 'MyOtherBasis 1')
        def f(x, scale, parameters):
            return x
        # self.assertIn(('MyEFT', 'MyBasis 1', 'MyBasis 2'), wcxf.Translator.instances)
        f = pkgutil.get_data('wcxf', 'data/test.wcs.yml')
        wc = wcxf.WC.load(f.decode('utf-8'))
        wc_out = wcxf.Matcher['MyEFT', 'MyBasis 1', 'MyOtherEFT', 'MyOtherBasis 1'].match(wc)
        # self.assertIn(('MyEFT', 'MyBasis 1', 'MyBasis 2'),
        #               wcxf.Basis['MyEFT', 'MyBasis 1'].known_translators['from'])
        # self.assertIn(('MyEFT', 'MyBasis 1', 'MyBasis 2'),
        #               wcxf.Basis['MyEFT', 'MyBasis 2'].known_translators['to'])
        # self.assertIn(('MyEFT', 'MyBasis 1', 'MyBasis 2'),
        #               wcxf.EFT['MyEFT'].known_translators)
        # remove dummy matcher
        del  wcxf.Matcher['MyEFT', 'MyBasis 1', 'MyOtherEFT', 'MyOtherBasis 1']

    def test_inheritance(self):
        f = pkgutil.get_data('wcxf', 'data/test.basis1.yml')
        parent = wcxf.Basis.load(f.decode('utf-8'))
        f = pkgutil.get_data('wcxf', 'data/test.basis2.yml')
        child = wcxf.Basis.load(f.decode('utf-8'))
        self.assertDictEqual(parent.sectors, child.sectors)

    def test_inheritance_modifiedsector(self):
        f = pkgutil.get_data('wcxf', 'data/test.basis1.yml')
        parent = wcxf.Basis.load(f.decode('utf-8'))
        f = pkgutil.get_data('wcxf', 'data/test.basis3.yml')
        child = wcxf.Basis.load(f.decode('utf-8'))
        self.assertEqual(set(parent.sectors.keys()), set(child.sectors.keys()))
        self.assertEqual(set(parent.sectors['My Sector 1'].keys()), {'C_1', 'C_2'})
        self.assertEqual(set(child.sectors['My Sector 1'].keys()), {'C_1'})
