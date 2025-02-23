import unittest

import numpy as np

from wilson import wcxf
import wilson
from wilson.test_wilson import get_random_wc

from .smeft_smeftsim import warsaw_up_to_SMEFTsim_general, SMEFTsim_general_to_warsaw_up, warsaw_up, smeftsim_general

np.random.seed(56)


class TestSMEFTsimgeneral(unittest.TestCase):
    def test_roundtrip_warsaw_up(self):
        C = {k: np.random.rand()*np.exp(1j*2*np.pi*np.random.uniform()) for k in warsaw_up.all_wcs}
        wc_out = warsaw_up_to_SMEFTsim_general(C)
        wc_in = SMEFTsim_general_to_warsaw_up(wc_out)
        self.assertEqual(
            warsaw_up_to_SMEFTsim_general(SMEFTsim_general_to_warsaw_up(wc_out)),
            wc_out
        )
        self.assertEqual(
            SMEFTsim_general_to_warsaw_up(warsaw_up_to_SMEFTsim_general(wc_in)),
            wc_in
        )
    def test_roundtrip_smeftsim_general(self):
        C = {k: np.random.rand() for k in smeftsim_general.all_wcs}
        wc_out = SMEFTsim_general_to_warsaw_up(C)
        wc_in = warsaw_up_to_SMEFTsim_general(wc_out)

        self.assertEqual(
            SMEFTsim_general_to_warsaw_up(warsaw_up_to_SMEFTsim_general(wc_out)),
            wc_out
        )
        self.assertEqual(
            warsaw_up_to_SMEFTsim_general(SMEFTsim_general_to_warsaw_up(wc_in)),
            wc_in
        )
