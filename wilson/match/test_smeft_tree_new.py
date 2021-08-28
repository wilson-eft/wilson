"""Compare the new tree-level SMEFT matching based on aXiv:1908.05295
to the old one based on arXiv:1709.04486."""

import unittest
import numpy as np
import numpy.testing as npt
from wilson import wcxf
import wilson
from wilson.test_wilson import get_random_wc
import wilson.match._smeft_old
import wilson.match.smeft_tree
from wilson.parameters import p


np.random.seed(77)


# These WCs appear quadratically in the "old" matching implementation
wc_sm = ['phiD', 'phiWB']
wc_vert = [k for k in wcxf.Basis['SMEFT', 'Warsaw'].all_wcs
          if 'phiq' in k
          or 'phil' in k
           or 'phiu' in k
           or 'phid' in k
           or 'phie' in k
          ]


_wcr = get_random_wc('SMEFT', 'Warsaw', 100, cmax=1e-8)
wc_linear = wilson.Wilson({k: v for k, v in _wcr.dict.items()
                     if k not in wc_sm + wc_vert}, 100, 'SMEFT', 'Warsaw').wc
wc_quadratic = wilson.Wilson({k: v for k, v in _wcr.dict.items()
                     if k in wc_sm + wc_vert}, 100, 'SMEFT', 'Warsaw').wc


class TestSMEFTWETreimpl(unittest.TestCase):
    def test_linear(self):
        """For WCs entering linearly, agreement should be numerically exact"""
        C = wilson.util.smeftutil.wcxf2arrays_symmetrized(wc_linear.dict)
        c_old = wilson.match._smeft_old.match_all_array(C, p)
        c_new = wilson.match.smeft_tree.match_all_array(C, p)
        for k in c_old:
            npt.assert_almost_equal(c_old[k], c_new[k], decimal=18,
                                    err_msg=f"Failed for {k}")

    def test_quadratic(self):
        """For WCs entering also quadratically, agreement should be good
        up to the size of quadratic terms"""
        C = wilson.util.smeftutil.wcxf2arrays_symmetrized(wc_quadratic.dict)
        c_old = wilson.match._smeft_old.match_all_array(C, p)
        c_new = wilson.match.smeft_tree.match_all_array(C, p)
        for k in c_old:
            npt.assert_almost_equal(c_old[k], c_new[k], decimal=10,
                                    err_msg=f"Failed for {k}")
