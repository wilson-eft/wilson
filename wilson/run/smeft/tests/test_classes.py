import unittest
import numpy as np
import numpy.testing as npt
from wilson.run.smeft import SMEFT, beta
from wilson.test_wilson import get_random_wc
from wilson.util.smeftutil import SM_keys
from wilson import wcxf


class TestSMEFT(unittest.TestCase):
    def test_smeft(self):
        wc = get_random_wc('SMEFT', 'Warsaw', 1000, 1e-8)
        with self.assertRaises(Exception):
            # no initial condition set
            smeft = SMEFT()
        # just check this doesn't raise errors
        smeft = SMEFT(wc)
        wc = smeft.run(900)
        wc.validate()
        wc = smeft.run(900, 'leadinglog')
        wc.validate()

    def test_empty(self):
        wc_sm = wcxf.WC('SMEFT', 'Warsaw', 160, {})
        smeft_sm = SMEFT(wc_sm, get_smpar=False)
        C_in_sm = smeft_sm.C_in
        C_in_sm_2 = beta.C_array2dict(np.zeros(9999))
        for k, v in C_in_sm_2.items():
            if k not in SM_keys:
                npt.assert_array_equal(v, C_in_sm[k],
                                       err_msg="Failed for {}".format(k))
    #
    # def test_rotation(self):
    #     wcout = pkgutil.get_data('wilson', 'run/smeft/tests/data/Output_SMEFTrunner.dat').decode('utf-8')
    #     smeft = SMEFT()
    #     smeft.load_initial((wcout,))
    #     smeft.scale_in = 1000
    #     smeft.scale_high = 1000
    #     C_out = smeft.rgevolve(scale_out=160)
    #     C_rot = smeft.rotate_defaultbasis(C_out)
    #     # check that input & output dicts have same keys
    #     self.assertSetEqual(set(C_out.keys()), set(C_rot.keys()))
    #     # new parameter point with diagonal sorted positive Yukawas and vanishing C_Xphi
    #     C_new = C.copy()
    #     for k in ['Gu', 'Gd', 'Ge']:
    #         C_new[k] = np.diag(np.sort(np.abs(np.diag(C[k]))))
    #     for k in ['uphi', 'dphi', 'ephi', 'llphiphi']:
    #         C_new[k] = np.zeros((3,3))
    #     smeft_new = SMEFT()
    #     smeft_new.set_initial(C_new, scale_in=160, scale_high=1000)
    #     C_new_rot = smeft_new.rotate_defaultbasis(C_new)
    #     for k in C_new:
    #         # now all the WCs & parameters should be rotation invariant!
    #         npt.assert_array_equal(C_new[k], C_new_rot[k])
    #     # rotating again should have no effect as we already are in the basis!
    #     C_rot2 = smeft.rotate_defaultbasis(C_rot)
    #     for k in C_rot:
    #         if 'Theta' in k: # doesn't seem to work for theta terms...
    #             continue
    #         # now all the WCs & parameters should be rotation invariant!
    #         npt.assert_array_almost_equal(C_rot2[k], C_rot[k],
    #                                err_msg="Failed for {}".format(k))
