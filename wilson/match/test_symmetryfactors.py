import unittest
import numpy as np
import numpy.testing as npt
from wilson import wcxf
import wilson
from wilson.run.smeft.smpar import p
import ckmutil
from math import pi, log
from wilson.util.wet_jms import C_symm_keys
import wilson.match.smeft_loop
import wilson.match.smeft_tree

from wilson.run.smeft import beta

np.random.seed(39)

# generate a random WC instance for the SMEFT Warsaw basis
C_Warsaw_random = {}
basis = wcxf.Basis['SMEFT', 'Warsaw']
for sector, wcs in basis.sectors.items():
    for name, d in wcs.items():
         C_Warsaw_random[name] = 1e-6*np.random.rand()
         if 'real' not in d or d['real'] == False:
             C_Warsaw_random[name] += 1j*1e-6*np.random.rand()


class TestMatch(unittest.TestCase):
    def test_match_qq3_1122(self):
        # tests matching of Q_qq^(3) of Warsaw onto O_ud^V8,LL in JMS basis
        from_wc =  wcxf.WC(values = {'qq3_1122': 2e-6} ,
                    scale = 1e3 , eft = 'SMEFT' , basis = 'Warsaw up')
        to_wc = from_wc.match('WET', 'JMS')
        V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["gamma"])
        self.assertAlmostEqual(to_wc['V8udLL_1221']/V[0,0].conjugate()
        /V[1,1].conjugate(),8e-6)

    def test_match_qq3_1322(self):
        # tests matching of Q_qq^(3) of Warsaw onto O_ud^V8,LL in JMS basis
        from_wc =  wcxf.WC(values = {'qq3_1322': 3e-6} ,
                    scale = 1e3 , eft = 'SMEFT' , basis = 'Warsaw up')
        to_wc = from_wc.match('WET', 'JMS')
        V = ckmutil.ckm.ckm_tree(p["Vus"], p["Vub"], p["Vcb"], p["gamma"])
        self.assertAlmostEqual(to_wc['V8udLL_1223']/V[2,2].conjugate()
        /V[1,1].conjugate(),12e-6)

    def test_match_ll_1212(self):
        # tests matching of Q_ll of Warsaw onto O_nue^V,LL in JMS basis
        from_wc =  wcxf.WC(values = {'ll_1212': 2e-6} ,
                    scale = 1e3 , eft = 'SMEFT' , basis = 'Warsaw up')
        to_wc = from_wc.match('WET', 'JMS')
        self.assertAlmostEqual(to_wc['VnueLL_1212'],4e-6)

    def test_match_ll_1312(self):
        # tests matching of Q_ll of Warsaw onto O_nue^V,LL in JMS basis
        from_wc =  wcxf.WC(values = {'ll_1213': 20e-6} ,
                    scale = 1e3 , eft = 'SMEFT' , basis = 'Warsaw up')
        to_wc = from_wc.match('WET', 'JMS')
        self.assertAlmostEqual(to_wc['VnueLL_1312'],20e-6)

    def test_match_lq1_1233(self):
        # tests matching of Q_lq^1 of Warsaw onto O_ed^V,LL in JMS basis
        from_wc =  wcxf.WC(values = {'lq1_1233': 12e-6} ,
                    scale = 1e3 , eft = 'SMEFT' , basis = 'Warsaw up')
        to_wc = from_wc.match('WET', 'JMS')
        self.assertAlmostEqual(to_wc['VedLL_1233'],12e-6)

    def test_match_ee_1233(self):
        # tests matching of Q_ee of Warsaw onto O_ee^V,RR in JMS basis
        from_wc =  wcxf.WC(values = {'ee_1233': 100e-6} ,
                    scale = 1e3 , eft = 'SMEFT' , basis = 'Warsaw up')
        to_wc = from_wc.match('WET', 'JMS')
        self.assertAlmostEqual(to_wc['VeeRR_1233'],100e-6)

    def test_match_uu_1112(self):
        # tests matching of Q_uu of Warsaw onto O_uu^V,RR in JMS basis
        from_wc =  wcxf.WC(values = {'uu_1112': 5e-6} ,
                    scale = 1e3 , eft = 'SMEFT' , basis = 'Warsaw up')
        to_wc = from_wc.match('WET', 'JMS')
        self.assertAlmostEqual(to_wc['VuuRR_1112'],5e-6)

    def test_match_dd_1223(self):
        # tests matching of Q_dd of Warsaw onto O_dd^V,RR in JMS basis
        from_wc =  wcxf.WC(values = {'dd_1223': 51e-6} ,
                    scale = 1e3 , eft = 'SMEFT' , basis = 'Warsaw up')
        to_wc = from_wc.match('WET', 'JMS')
        self.assertAlmostEqual(to_wc['VddRR_1223'],51e-6)


class TestRun(unittest.TestCase):
    def test_run_lq3_3333(self):
        w = wilson.Wilson({'lq3_2333': 1e-6}, 1000, 'SMEFT', 'Warsaw')
        # determine g at input scale
#        g = wilson.run.smeft.SMEFT(w.wc).C_in['g']
        g = wilson.run.smeft.EFTevolve(w.wc, beta.beta).C_in['g']
        # run down
        wc = w.match_run(100, 'SMEFT', 'Warsaw')
        # compare LL to expected value
        sf = 2  # symmetry factor since our 2333 is 2* larger
        self.assertAlmostEqual(wc['ll_2333'],
        sf * 1e-6 / (16 * pi**2) * (-g**2) * log(100 / 1000))


class TestMatchingSymmetryFactors(unittest.TestCase):
    def test_match_symmfac(self):
        """Test that the WET WCs coming out of the matching fulfill
        the correct symmetry relations (namely, have the same symmetries
        as the operators)."""
        C_SMEFT = wilson.util.smeftutil.wcxf2arrays_symmetrized(C_Warsaw_random)
        C = wilson.match.smeft_tree.match_all_array(C_SMEFT, p)
        for k in C:
            if k in C_symm_keys[41] + C_symm_keys[4] + C_symm_keys[6]:
                a = np.einsum('klij', C[k]) # C_ijkl = C_klij
                npt.assert_array_almost_equal(C[k], a, err_msg=f"Failed for {k}",
                                              decimal=20)
            if k in C_symm_keys[5] + C_symm_keys[4] + C_symm_keys[6]:
                a = np.einsum('jilk', C[k]).conj() # C_ijkl = C_jilk*
                npt.assert_array_almost_equal(C[k], a, err_msg=f"Failed for {k}",
                                              decimal=20)
            if k in C_symm_keys[4] + C_symm_keys[6]:
                a = np.einsum('lkji', C[k]).conj() # C_ijkl = C_lkji*
                npt.assert_array_almost_equal(C[k], a, err_msg=f"Failed for {k}",
                                              decimal=20)
            if k in C_symm_keys[6]:
                a = np.einsum('ilkj', C[k]) # C_ijkl = C_ilkj
                npt.assert_array_almost_equal(C[k], a, err_msg=f"Failed for {k}",
                                              decimal=20)
            if k in C_symm_keys[71]:
                a = -np.einsum('jikl', C[k]) # C_ijkl = -C_jikl
                npt.assert_array_almost_equal(C[k], a, err_msg=f"Failed for {k}",
                                              decimal=20)

    def test_match_symmfac_loop(self):
        """Test that the WET WCs coming out of the matching fulfill
        the correct symmetry relations (namely, have the same symmetries
        as the operators)."""
        C_SMEFT = wilson.util.smeftutil.wcxf2arrays_symmetrized(C_Warsaw_random)
        C = wilson.match.smeft_loop.match_all_array(C_SMEFT, p, scale=120)
        for k in C:
            if k in C_symm_keys[41] + C_symm_keys[4] + C_symm_keys[6]:
                a = np.einsum('klij', C[k]) # C_ijkl = C_klij
                npt.assert_array_almost_equal(np.array(C[k], complex), np.array(a, complex), err_msg=f"Failed for {k}",
                                              decimal=20)
            if k in C_symm_keys[5] + C_symm_keys[4] + C_symm_keys[6]:
                a = np.einsum('jilk', C[k]).conj() # C_ijkl = C_jilk*
                npt.assert_array_almost_equal(np.array(C[k], complex), np.array(a, complex), err_msg=f"Failed for {k}",
                                              decimal=20)
            if k in C_symm_keys[4] + C_symm_keys[6]:
                a = np.einsum('lkji', C[k]).conj() # C_ijkl = C_lkji*
                npt.assert_array_almost_equal(np.array(C[k], complex), np.array(a, complex), err_msg=f"Failed for {k}",
                                              decimal=20)
            if k in C_symm_keys[6]:
                a = np.einsum('ilkj', C[k]) # C_ijkl = C_ilkj
                npt.assert_array_almost_equal(np.array(C[k], complex), np.array(a, complex), err_msg=f"Failed for {k}",
                                              decimal=20)
            if k in C_symm_keys[71]:
                a = -np.einsum('jikl', C[k]) # C_ijkl = -C_jikl
                npt.assert_array_almost_equal(np.array(C[k], complex), np.array(a, complex), err_msg=f"Failed for {k}",
                                              decimal=20)
