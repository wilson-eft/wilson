import unittest
import numpy as np
import numpy.testing as npt
import wcxf
from wcxf.converters import eos
import os

my_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(my_path, '..', 'data')

class TestEOS(unittest.TestCase):
    def test_get_sm_wcs(self):
        eos_sm_dict = eos.get_sm_wcs(data_path)
        self.assertEqual(eos_sm_dict['b->s::Re{c7}']['central'], -0.33726473)
        self.assertEqual(eos_sm_dict['b->s::c5']['central'], 0.00042854)
        self.assertEqual(eos_sm_dict['b->ulnu::Im{cVL}']['min'], 0.0)
    
    def test_wcxf2eos(self):
        sm_wc_dict = eos.get_sm_wcs(data_path)
        wc_dict = { 'b->s::c7': 0.1,
                    'b->uenue::cSL': 0.2j, 'b->umunumu::cSL': 0.2j,
                    'b->uenue::cVL': 0.2, 'b->umunumu::cVL': 0.2,
                    'b->s::c5': -0.03 }
        wc = wcxf.WC('WET', 'EOS', 4.2, wcxf.WC.dict2values(wc_dict))
        eos_dict = eos.wcxf2eos(wc, sm_wc_dict)
        self.assertEqual(eos_dict['b->s::Re{c7}']['central'], -0.33726473 + 0.1)
        self.assertEqual(eos_dict['b->s::c5']['central'], 0.00042854 - 0.03)
        self.assertEqual(eos_dict['b->ulnu::Im{cSL}']['min'], 0.0 + 0.2)
        self.assertEqual(eos_dict['b->ulnu::Re{cVL}']['max'], 1.0 + 0.2)
        
    def test_lfu(self):
        sm_wc_dict = eos.get_sm_wcs(data_path)
        wc_dict = { 'b->uenue::cSL': 0.2j, 'b->umunumu::cSL': 0.1j, }
        wc = wcxf.WC('WET', 'EOS', 4.2, wcxf.WC.dict2values(wc_dict))
        with self.assertRaises(ValueError):
            eos.wcxf2eos(wc, sm_wc_dict)