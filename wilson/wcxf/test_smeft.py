import unittest
import numpy as np
import numpy.testing as npt
import yaml
import json
import pkgutil
from wilson import wcxf

sectors_dBdL0 = [
    'dB=de=dmu=dtau=0',
    'mue',
    'taue',
    'mutau',
    'muemue',
    'etauemu',
    'muemutau',
    'tauetaue',
    'tauetaumu',
    'taumutaumu',
]

class TestSMEFT(unittest.TestCase):
    def test_smeft_number(self):
        """Test the number of B and L conserving coefficients in SMEFT"""
        basis = wcxf.Basis['SMEFT', 'Warsaw']
        count=0
        for sector_name in sectors_dBdL0:
            sector = basis.sectors[sector_name]
            for wc in sector:
                if 'real' not in sector[wc] or not sector[wc]['real']:
                    count +=2 # complex coeffs
                else:
                    count +=1 # real coeffs
        self.assertEqual(count, 2499) # no. of B and L conserving coefficients
