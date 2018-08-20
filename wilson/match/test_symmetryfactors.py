import unittest
import numpy as np
import wcxf
import wilson
from wilson.parameters import p
import ckmutil
from math import pi, log


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
        g = wilson.run.smeft.SMEFT(w.wc).C_in['g']
        # run down
        wc = w.match_run(100, 'SMEFT', 'Warsaw')
        # compare LL to expected value
        sf = 2  # symmetry factor since our 2333 is 2* larger
        self.assertAlmostEqual(wc['ll_2333'],
        sf * 1e-6 / (16 * pi**2) * (-g**2) * log(100 / 1000))
