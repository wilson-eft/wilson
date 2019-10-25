import unittest
import numpy as np
import numpy.testing as npt
import yaml
import json
import pkgutil
import wcxf

class TestSMEFT(unittest.TestCase):
    def test_smeft_number(self):
        """Test the number of B and L conserving coefficients in SMEFT"""
        basis = wcxf.Basis['SMEFT', 'Warsaw']
        sector = basis.sectors['dB=dL=0']
        count=0
        for wc in sector:
            if 'real' not in sector[wc] or not sector[wc]['real']:
                count +=2 # complex coeffs
            else:
                count +=1 # real coeffs
        self.assertEqual(count, 2499) # no. of B and L conserving coefficients
