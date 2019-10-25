import unittest
from math import pi, sqrt

import numpy as np

from wilson import wcxf
import wilson
from wilson.parameters import p
from wilson.test_wilson import get_random_wc


np.random.seed(56)


class TestHiggs(unittest.TestCase):
    def test_higgs_warsaw_up(self):
        # test cgg, a trivial case
        for t in ["", "tilde"]:
            for up in ["", " up"]:
                wc = wcxf.WC("SMEFT", "Higgs-Warsaw up", 120, {"cgg" + t: 1e-8})
                wcW = wc.translate("Warsaw" + up)
                gs = sqrt(4 * pi * p["alpha_s"])
                self.assertEqual(wcW.dict, {"phiG" + t: 1e-8 * gs ** 2 / 4})

    def test_roundtrip_warsaw(self):
        wc = get_random_wc("SMEFT", "Warsaw up", 120)
        basis = wcxf.Basis["SMEFT", "Warsaw up"]
        for name in basis.all_wcs:
            if name == 'phiD':
                continue
            wc = wcxf.WC("SMEFT", "Warsaw up", 120, {name: 1e-8})
            wc_translated = wc.translate("Higgs-Warsaw up")
            wc_translated.validate()
            # translate back and check that nothing changed
            wc_roundtrip = wc_translated.translate("Warsaw up")
            for k, v in wc.dict.items():
                self.assertAlmostEqual(
                    v, wc_roundtrip.dict[k], places=20, msg="Failed for {}".format(k)
                )
