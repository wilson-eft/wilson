import unittest
import numpy as np
import wcxf
import wilson

np.random.seed(89)

# generate a random WC instance for the SMEFT Warsaw basis
C_Warsaw_random = {}
basis = wcxf.Basis['SMEFT', 'Warsaw']
for sector, wcs in basis.sectors.items():
    for name, d in wcs.items():
         C_Warsaw_random[name] = 1e-6*np.random.rand()
         if 'real' not in d or d['real'] == False:
             C_Warsaw_random[name] += 1j*1e-6*np.random.rand()
v_Warsaw_random = wcxf.WC.dict2values(C_Warsaw_random)
wc_Warsaw_random = wcxf.WC('SMEFT', 'Warsaw', scale=160,
                           values=v_Warsaw_random)
wc_Warsaw_minimal1 = wcxf.WC('SMEFT', 'Warsaw', scale=160,
                             values={'G': 1.2})
wc_Warsaw_minimal2 = wcxf.WC('SMEFT', 'Warsaw', scale=160,
                             values={'ed_1123':  {'Re': 1.2}})

class TestSMEFTWET(unittest.TestCase):
    def test_warsaw_jms(self):
        for wcW in [wc_Warsaw_random, wc_Warsaw_minimal1, wc_Warsaw_minimal2]:
            wc = wcW.match('WET', 'JMS')
            self.assertEqual(wc.eft, 'WET')
            self.assertEqual(wc.basis, 'JMS')
            wc.validate()

    def test_warsaw_flavio(self):
        for wcW in [wc_Warsaw_random, wc_Warsaw_minimal1, wc_Warsaw_minimal2]:
            wc = wcW.match('WET', 'flavio')
            self.assertEqual(wc.eft, 'WET')
            self.assertEqual(wc.basis, 'flavio')
            wc.validate()

    def test_warsaw_EOS(self):
        for wcW in [wc_Warsaw_random, wc_Warsaw_minimal1, wc_Warsaw_minimal2]:
            wc = wcW.match('WET', 'EOS')
            self.assertEqual(wc.eft, 'WET')
            self.assertEqual(wc.basis, 'EOS')
            wc.validate()

    def test_warsaw_Bern(self):
        for wcW in [wc_Warsaw_random, wc_Warsaw_minimal1, wc_Warsaw_minimal2]:
            wc = wcW.match('WET', 'Bern')
            self.assertEqual(wc.eft, 'WET')
            self.assertEqual(wc.basis, 'Bern')
            wc.validate()
