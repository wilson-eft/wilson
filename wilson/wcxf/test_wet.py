import unittest
from wilson import wcxf
import pkgutil
class TestWET(unittest.TestCase):
    def test_bern(self):
        basis = wcxf.Basis['WET', 'Bern']
        # comparing no. of WCs to table 1 of arXiv:1704.06639
        self.assertEqual(len(basis.sectors['sbsb']), 8)
        self.assertEqual(len(basis.sectors['dbdb']), 8)
        self.assertEqual(len(basis.sectors['ubenu']), 5*3)
        self.assertEqual(len(basis.sectors['ubmunu']), 5*3)
        self.assertEqual(len(basis.sectors['ubtaunu']), 5*3)
        self.assertEqual(len(basis.sectors['cbenu']), 5*3)
        self.assertEqual(len(basis.sectors['cbmunu']), 5*3)
        self.assertEqual(len(basis.sectors['cbtaunu']), 5*3)
        self.assertEqual(len(basis.sectors['sbuc']), 20)
        self.assertEqual(len(basis.sectors['sbcu']), 20)
        self.assertEqual(len(basis.sectors['dbuc']), 20)
        self.assertEqual(len(basis.sectors['dbcu']), 20)
        self.assertEqual(len(basis.sectors['sbsd']), 10)
        self.assertEqual(len(basis.sectors['dbds']), 10)
        self.assertEqual(len(basis.sectors['sb']), 57*2)
        self.assertEqual(len(basis.sectors['sbnunu']), 2*9)
        self.assertEqual(len(basis.sectors['db']), 57*2)
        self.assertEqual(len(basis.sectors['dbnunu']), 2*9)
        for l1 in ['e', 'mu', 'tau']:
            for l2 in ['e', 'mu', 'tau']:
                if l1 != l2:
                    self.assertEqual(len(basis.sectors['sb'+l1+l2]), 10)
                    self.assertEqual(len(basis.sectors['db'+l1+l2]), 10)

    def test_jms(self):
        basis = wcxf.Basis['WET', 'JMS']
        all_wc = {k: v for sk, sv in basis.sectors.items() for k, v in sv.items()}
        def isreal(wc):
            if 'real' in all_wc[wc] and all_wc[wc]['real']:
                return True
            else:
                return False
        def assert_len(s, n):
            L = sum([1 if isreal(k) else 2 for k in all_wc if s in k])
            try:
                self.assertEqual(L, n,
                             msg=f"Failed for {s}")
            except Exception as exc:
                print(exc)
        # compare individual counts to tables 11-17 in arXiv:1709.04486
        assert_len('egamma_', 9*2)
        assert_len('ugamma_', 4*2)
        assert_len('dgamma_', 9*2)
        assert_len('uG_', 4*2)
        assert_len('dG_', 9*2)
        assert_len('VnunuLL_', 36)
        assert_len('VeeLL_', 36)
        assert_len('VnueLL_', 81)
        assert_len('VnuuLL_', 36)
        assert_len('VnudLL_', 81)
        assert_len('VeuLL_', 36)
        assert_len('VedLL_', 81)
        assert_len('VnueduLL_', 2*54)
        assert_len('VuuLL_', 10)
        assert_len('VddLL_', 45)
        assert_len('V1udLL_', 36)
        assert_len('V8udLL_', 36)
        assert_len('VeeRR_', 36)
        assert_len('VeuRR_', 36)
        assert_len('VedRR_', 81)
        assert_len('VuuRR_', 10)
        assert_len('VddRR_', 45)
        assert_len('V1udRR_', 36)
        assert_len('V8udRR_', 36)
        assert_len('VnueLR_', 81)
        assert_len('VeeLR_', 81)
        assert_len('VnuuLR_', 36)
        assert_len('VnudLR_', 81)
        assert_len('VeuLR_', 36)
        assert_len('VedLR_', 81)
        assert_len('VueLR_', 36)
        assert_len('VdeLR_', 81)
        assert_len('VnueduLR_', 2*54)
        assert_len('V1uuLR_', 16)
        assert_len('V8uuLR_', 16)
        assert_len('V1udLR_', 36)
        assert_len('V8udLR_', 36)
        assert_len('V1duLR_', 36)
        assert_len('V8duLR_', 36)
        assert_len('V1ddLR_', 81)
        assert_len('V8ddLR_', 81)
        assert_len('V1udduLR_', 2*36)
        assert_len('V8udduLR_', 2*36)
        assert_len('SeuRL_', 2*36)
        assert_len('SedRL_', 2*81)
        assert_len('SnueduRL_', 2*54)
        assert_len('SeeRR', 2*45)
        assert_len('SeuRR', 2*36)
        assert_len('TeuRR', 2*36)
        assert_len('SedRR', 2*81)
        assert_len('TedRR', 2*81)
        assert_len('SnueduRR', 2*54)
        assert_len('TnueduRR', 2*54)
        assert_len('S1uuRR', 2*10)
        assert_len('S8uuRR', 2*10)
        assert_len('S1udRR', 2*36)
        assert_len('S8udRR', 2*36)
        assert_len('S1ddRR', 2*45)
        assert_len('S8ddRR', 2*45)
        assert_len('S1udduRR', 2*36)
        assert_len('S8udduRR', 2*36)
        # compare total count to table 22
        ntot = sum([1 if isreal(k) else 2 for k in all_wc])
        self.assertEqual(ntot, 1+87+186+76+21+66+76+90+252+266+171+45+342+254+1+66+156+51+15+51+51+72+207+216+171+45+342+254+9+9+26+26+288*2)

#   Unit test for Bern to flavio translator
    def test_wc1(self):
        f = pkgutil.get_data('wilson', 'wcxf/data/test.wcs.bern.yml').decode('utf-8')
        wc0 = wcxf.WC.load(f)
        wc0.validate()
        self.assertEqual(wc0.eft, 'WET')
        self.assertEqual(wc0.basis, 'Bern')
        self.assertEqual(wc0.scale, 4.8)
        wc1= wc0.translate('flavio')
        f = pkgutil.get_data('wilson', 'wcxf/data/test.wcs.flavio.out.yml').decode('utf-8')
        wc2 = wcxf.WC.load(f)
        self.assertEqual(wc2.eft, 'WET')
        self.assertEqual(wc2.basis, 'flavio')
        self.assertEqual(wc2.scale, 4.8)
        wc2.validate()
        self.assertAlmostEqual(wc1.dict['CVLL_sdsd'], wc2.dict['CVLL_sdsd'])
        self.assertAlmostEqual(wc1.dict['CVLL_bsbs'], wc2.dict['CVLL_bsbs'])
        self.assertAlmostEqual(wc1.dict['CVLL_bdbd'], wc2.dict['CVLL_bdbd'])
        self.assertAlmostEqual(wc1.dict['CVL_dumunumu'], wc2.dict['CVL_dumunumu'])
        self.assertAlmostEqual(wc1.dict['CVL_sumunumu'], wc2.dict['CVL_sumunumu'])
        self.assertAlmostEqual(wc1.dict['CVL_bumunumu'], wc2.dict['CVL_bumunumu'])
        self.assertAlmostEqual(wc1.dict['CVL_dcmunumu'], wc2.dict['CVL_dcmunumu'])
        self.assertAlmostEqual(wc1.dict['CVL_scmunumu'], wc2.dict['CVL_scmunumu'])
        self.assertAlmostEqual(wc1.dict['CVL_bcmunumu'], wc2.dict['CVL_bcmunumu'])
        self.assertAlmostEqual(wc1.dict['C9_bsmumu'], wc2.dict['C9_bsmumu'])
        self.assertAlmostEqual(wc1.dict['C9_bdmumu'], wc2.dict['C9_bdmumu'])
        self.assertAlmostEqual(wc1.dict['C7_bs'], wc2.dict['C7_bs'])
        self.assertAlmostEqual(wc1.dict['C8_bs'], wc2.dict['C8_bs'])
        self.assertAlmostEqual(wc1.dict['C7_bd'], wc2.dict['C7_bd'])
        self.assertAlmostEqual(wc1.dict['C8_bd'], wc2.dict['C8_bd'])

#   Unit test for flavio to Bern translator
    def test_wc2(self):
        f = pkgutil.get_data('wilson', 'wcxf/data/test.wcs.flavio.yml').decode('utf-8')
        wc0 = wcxf.WC.load(f)
        wc0.validate()
        self.assertEqual(wc0.eft, 'WET')
        self.assertEqual(wc0.basis, 'flavio')
        self.assertEqual(wc0.scale, 4.8)
        wc1= wc0.translate('Bern')
        f = pkgutil.get_data('wilson', 'wcxf/data/test.wcs.bern.out.yml').decode('utf-8')
        wc2 = wcxf.WC.load(f)
        self.assertEqual(wc2.eft, 'WET')
        self.assertEqual(wc2.basis, 'Bern')
        self.assertEqual(wc2.scale, 4.8)
        wc2.validate()
        self.assertAlmostEqual(wc1.dict['1sbsb'], wc2.dict['1sbsb'])
        self.assertAlmostEqual(wc1.dict['1dbdb'], wc2.dict['1dbdb'])
        self.assertAlmostEqual(wc1.dict['1dsds'], wc2.dict['1dsds'])
        self.assertAlmostEqual(wc1.dict['1udmumu'], wc2.dict['1udmumu'])
        self.assertAlmostEqual(wc1.dict['1usmumu'], wc2.dict['1usmumu'])
        self.assertAlmostEqual(wc1.dict['1ubmumu'], wc2.dict['1ubmumu'])
        self.assertAlmostEqual(wc1.dict['1cdmumu'], wc2.dict['1cdmumu'])
        self.assertAlmostEqual(wc1.dict['1csmumu'], wc2.dict['1csmumu'])
        self.assertAlmostEqual(wc1.dict['1cbmumu'], wc2.dict['1cbmumu'])
        self.assertAlmostEqual(wc1.dict['1sbmumu'], wc2.dict['1sbmumu'])
        self.assertAlmostEqual(wc1.dict['1dbmumu'], wc2.dict['1dbmumu'])
        self.assertAlmostEqual(wc1.dict['7gammasb'], wc2.dict['7gammasb'])
        self.assertAlmostEqual(wc1.dict['8gsb'], wc2.dict['8gsb'])
        self.assertAlmostEqual(wc1.dict['7gammadb'], wc2.dict['7gammadb'])
        self.assertAlmostEqual(wc1.dict['8gdb'], wc2.dict['8gdb'])

#   Unit test for JMS to Bern translator
    def test_wc3(self):
        f = pkgutil.get_data('wilson', 'wcxf/data/test.wcs.jms.yml').decode('utf-8')
        wc0 = wcxf.WC.load(f)
        wc0.validate()
        self.assertEqual(wc0.eft, 'WET')
        self.assertEqual(wc0.basis, 'JMS')
        self.assertEqual(wc0.scale, 4.8)
        wc1= wc0.translate('Bern')
        f = pkgutil.get_data('wilson', 'wcxf/data/test.wcs.bern2.out.yml').decode('utf-8')
        wc2 = wcxf.WC.load(f)
        self.assertEqual(wc2.eft, 'WET')
        self.assertEqual(wc2.basis, 'Bern')
        self.assertEqual(wc2.scale, 4.8)
        wc2.validate()
        self.assertAlmostEqual(wc1.dict['1sbsb'], wc2.dict['1sbsb'])
        self.assertAlmostEqual(wc1.dict['1dbdb'], wc2.dict['1dbdb'])
        self.assertAlmostEqual(wc1.dict['1dsds'], wc2.dict['1dsds'])
        self.assertAlmostEqual(wc1.dict['1udmumu'], wc2.dict['1udmumu'])
        self.assertAlmostEqual(wc1.dict['1usmumu'], wc2.dict['1usmumu'])
        self.assertAlmostEqual(wc1.dict['1ubmumu'], wc2.dict['1ubmumu'])
        self.assertAlmostEqual(wc1.dict['1cdmumu'], wc2.dict['1cdmumu'])
        self.assertAlmostEqual(wc1.dict['1csmumu'], wc2.dict['1csmumu'])
        self.assertAlmostEqual(wc1.dict['1cbmumu'], wc2.dict['1cbmumu'])
        self.assertAlmostEqual(wc1.dict['1sbuc'], wc2.dict['1sbuc'])
        self.assertAlmostEqual(wc1.dict['3sbuc'], wc2.dict['3sbuc'])
        self.assertAlmostEqual(wc1.dict['1sbcu'], wc2.dict['1sbcu'])
        self.assertAlmostEqual(wc1.dict['3sbcu'], wc2.dict['3sbcu'])
        self.assertAlmostEqual(wc1.dict['1dbuc'], wc2.dict['1dbuc'])
        self.assertAlmostEqual(wc1.dict['3dbuc'], wc2.dict['3dbuc'])
        self.assertAlmostEqual(wc1.dict['1dbcu'], wc2.dict['1dbcu'])
        self.assertAlmostEqual(wc1.dict['3dbcu'], wc2.dict['3dbcu'])
        self.assertAlmostEqual(wc1.dict['1dsuc'], wc2.dict['1dsuc'])
        self.assertAlmostEqual(wc1.dict['3dsuc'], wc2.dict['3dsuc'])
        self.assertAlmostEqual(wc1.dict['1dscu'], wc2.dict['1dscu'])
        self.assertAlmostEqual(wc1.dict['3dscu'], wc2.dict['3dscu'])
        self.assertAlmostEqual(wc1.dict['1sbsd'], wc2.dict['1sbsd'])
        self.assertAlmostEqual(wc1.dict['3sbsd'], wc2.dict['3sbsd'])
        self.assertAlmostEqual(wc1.dict['1dbds'], wc2.dict['1dbds'])
        self.assertAlmostEqual(wc1.dict['3dbds'], wc2.dict['3dbds'])
#        self.assertAlmostEqual(wc1.dict['1dbsb'], wc2.dict['1dbsb'])
#        self.assertAlmostEqual(wc1.dict['3dbsb'], wc2.dict['3dbsb'])
        self.assertAlmostEqual(wc1.dict['1sbmumu'], wc2.dict['1sbmumu'])
        self.assertAlmostEqual(wc1.dict['3sbmumu'], wc2.dict['3sbmumu'])
        self.assertAlmostEqual(wc1.dict['1dbmumu'], wc2.dict['1dbmumu'])
        self.assertAlmostEqual(wc1.dict['3dbmumu'], wc2.dict['3dbmumu'])
        self.assertAlmostEqual(wc1.dict['1dsmumu'], wc2.dict['1dsmumu'])
        self.assertAlmostEqual(wc1.dict['3dsmumu'], wc2.dict['3dsmumu'])
        self.assertAlmostEqual(wc1.dict['1sbdd'], wc2.dict['1sbdd'])
        self.assertAlmostEqual(wc1.dict['3sbdd'], wc2.dict['3sbdd'])
        self.assertAlmostEqual(wc1.dict['1dbdd'], wc2.dict['1dbdd'])
        self.assertAlmostEqual(wc1.dict['3dbdd'], wc2.dict['3dbdd'])
        self.assertAlmostEqual(wc1.dict['1dsdd'], wc2.dict['1dsdd'])
        self.assertAlmostEqual(wc1.dict['3dsdd'], wc2.dict['3dsdd'])
        self.assertAlmostEqual(wc1.dict['7gammasb'], wc2.dict['7gammasb'])
        self.assertAlmostEqual(wc1.dict['8gsb'], wc2.dict['8gsb'])

    def test_jms_bnv(self):
        basis = wcxf.Basis['WET', 'JMS']
        bnvsec = ['uddnu', 'udsnu', 'udbnu', 'ussnu', 'usbnu', 'ubbnu', 'cddnu', 'cdsnu', 'cdbnu', 'cssnu', 'csbnu', 'cbbnu', 'uude', 'uudmu', 'uudtau', 'ucde', 'ucdmu', 'ucdtau', 'ccde', 'ccdmu', 'ccdtau', 'uuse', 'uusmu', 'uustau', 'ucse', 'ucsmu', 'ucstau', 'ccse', 'ccsmu', 'ccstau', 'uube', 'uubmu', 'uubtau', 'ucbe', 'ucbmu', 'ucbtau', 'ccbe', 'ccbmu', 'ccbtau']
        Ntot = sum([len(basis.sectors[s]) for s in bnvsec])
        # arXiv:1709.04486, table 20
        self.assertEqual(Ntot, 288)
