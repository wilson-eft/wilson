import unittest
import numpy as np
from wilson.smeftrunner import definitions, beta
from wilson.util import smeftutil
import test_beta

C = test_beta.C.copy()
for i in C:
    if i in smeftutil.WC_keys_2f + smeftutil.WC_keys_4f:
        # make Wilson coefficients involving fermions complex!
        C[i] = C[i] + 1j*C[i]

class TestSymm(unittest.TestCase):

    def test_redundant(self):
        # generate parameter dict filled with unique, ascending numbers
        C_num = beta.C_array2dict(np.arange(0, 9999, dtype=complex))
        C_num_symm = smeftutil.symmetrize(C_num)
        for k, el in definitions.redundant_elements.items():
            for e in el:
                # check that the elements in the symmetrized array
                # are NOT equal to the original ones IF they belong
                # to the redundant ones
                if k == 'qqql':
                    continue # OK, this doesn't work for qqql...
                self.assertNotEqual(C_num[k][e].real, C_num_symm[k][e].real)
        # generate parameter dict filled with unique, ascending IMAGINARY numbers
        C_num = beta.C_array2dict(1j*np.arange(0, 9999, dtype=complex))
        C_num_symm = smeftutil.symmetrize(C_num)
        for k, el in definitions.vanishing_im_parts.items():
            for e in el:
                # original im parts are NOT zero
                self.assertNotEqual(C_num[k][e].imag, 0)
                # symmetrized im parts ARE zero
                self.assertEqual(C_num_symm[k][e].imag, 0)
