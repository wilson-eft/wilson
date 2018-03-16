import unittest
import numpy as np
import numpy.testing as npt
import pkgutil
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

class TestWarsawMass(unittest.TestCase):
    def test_smeft_mass(self):
        for wcW in [wc_Warsaw_random, wc_Warsaw_minimal1, wc_Warsaw_minimal2]:
            wcW = wc_Warsaw_random.translate('Warsaw mass')
            p = {'Vub': 3.6e-3}  # pass a parameter, but not all
            wcW = wc_Warsaw_random.translate('Warsaw mass', p)
            wcW.validate()
            # almost all WCs should actually stay the same
            for k, v in wc_Warsaw_random.dict.items():
                if k.split('_')[0] not in ['uphi', 'uG', 'uW', 'uB', 'llphiphi']:
                    self.assertEqual(wcW.dict[k], v,
                                     msg="Not equal for {}".format(k))
            for i in range(3):
                for j in range(3):
                    if i > j:
                        # the off-diagonal neutrino mass matrix elements
                        # must vanish in the mass basis, i.e. be absent
                        self.assertTrue('llphiphi_{}{}'.format(i+1, j+1) not in wcW.dict)


class TestWarsawUp(unittest.TestCase):
    def test_warsaw_up(self):
        for wcWdown in [wc_Warsaw_random, wc_Warsaw_minimal1, wc_Warsaw_minimal2]:
            wcW = wcWdown.translate('Warsaw up')
            wcW.validate()
            # translate back and check that nothing changed
            wc_roundtrip = wcW.translate('Warsaw')
            for k, v in wcWdown.dict.items():
                self.assertAlmostEqual(v, wc_roundtrip.dict[k], places=12,
                                       msg="Failed for {}".format(k))


class TestIO(unittest.TestCase):
    def test_arrays2wcxf(self):
        """Test the functions needed for WCxf IO."""
        from wilson import smeftrunner
        wcout = pkgutil.get_data('wilson', 'smeftrunner/tests/data/Output_SMEFTrunner.dat').decode('utf-8')
        smeft = smeftrunner.SMEFT()
        smeft.load_initial((wcout,))
        d_wcxf = wcxf.translators.smeft.arrays2wcxf(smeft.C_in)
        C_out = wcxf.translators.smeft.wcxf2arrays(d_wcxf)
        C_out = wilson.util.smeftutil.symmetrize(C_out)
        for k, v in smeft.C_in.items():
            npt.assert_array_equal(v, C_out[k],
                                   err_msg="Arrays are not equal for {}".format(k))
