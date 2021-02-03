import unittest
from wilson import wcxf
import numpy as np
from wilson.run.smeft import smpar, SMEFT
from math import sqrt, pi
import ckmutil


np.random.seed(110)


def get_random_wc(eft, basis, scale=160, cmax=1e-2):
    """Generate a random Wilson coefficient instance for a given basis."""
    basis_obj = wcxf.Basis[eft, basis]
    _wc = {}
    for s in basis_obj.sectors.values():
        for name, d in s.items():
            _wc[name] = cmax * np.random.rand()
            if 'real' not in d or not d['real']:
                _wc[name] += 1j * cmax * np.random.rand()
    return wcxf.WC(eft, basis, scale, wcxf.WC.dict2values(_wc))



class Testgpbar(unittest.TestCase):
    def test_sm(self):
        gp = smpar.get_gpbar(0.3, 0.6, 246, {'phiWB': 0, 'phiB': 0})
        self.assertEqual(gp, 0.3*0.6/sqrt(-0.3**2+0.6**2))

    def test_general(self):
        C = {'phiWB': np.random.rand() / 200**2, 'phiB': np.random.rand() / 200**2}
        v = 246
        gp = smpar.get_gpbar(0.3, 0.6, v, C)
        gpb = gp / (1 - C['phiB'] * (v**2))
        gb = 0.6
        eps = C['phiWB'] * (v**2)
        eb = (gb * gpb / sqrt(gb**2 + gpb**2) *
              (1 - eps * gb * gpb / (gb**2 + gpb**2)))
        self.assertAlmostEqual(eb, 0.3)


class TestMh2v(unittest.TestCase):
    def test_sm(self):
        v = 246
        Mh2 = 125**2
        C = {k: 0 for k in ['phi', 'phiBox', 'phiD']}
        d = smpar.vMh2_to_m2Lambda(v, Mh2, C)
        d2 = smpar.m2Lambda_to_vMh2(d['m2'], d['Lambda'], C)
        self.assertAlmostEqual(d['m2'], Mh2/2)
        self.assertAlmostEqual(d['Lambda'], Mh2/v**2)
        self.assertAlmostEqual(d2['v'], v)
        self.assertAlmostEqual(d2['Mh2'], Mh2)

    def test_Cphi0(self):
        v = 246
        Mh2 = 125**2
        C = {k: np.random.rand() / 500**2 for k in ['phiBox', 'phiD']}
        C['phi'] = 0
        d = smpar.vMh2_to_m2Lambda(v, Mh2, C)
        d2 = smpar.m2Lambda_to_vMh2(d['m2'], d['Lambda'], C)
        self.assertAlmostEqual(d2['v'], v)
        self.assertAlmostEqual(d2['Mh2'], Mh2)

    def test_general(self):
        v = 246
        Mh2 = 125**2
        C = {k: np.random.rand() / 500**2 for k in ['phi', 'phiBox', 'phiD']}
        d = smpar.vMh2_to_m2Lambda(v, Mh2, C)
        d2 = smpar.m2Lambda_to_vMh2(d['m2'], d['Lambda'], C)
        self.assertAlmostEqual(d2['v'], v, places=6)
        self.assertAlmostEqual(d2['Mh2'], Mh2, places=6)

    def test_vreal(self):
        m2 = 7812
        Lambda = 0.258
        C = {k: 0 for k in ['phiD', 'phiBox']}
        C['phi'] = 1e-6
        with self.assertRaises(ValueError):
            smpar.m2Lambda_to_vMh2(m2, Lambda, C)

    def test_args_positive(self):
        v = 246
        Mh2 = 125**2
        m2 = 7812
        Lambda = 0.258
        C = {k: 0 for k in ['phi', 'phiBox', 'phiD']}
        with self.assertRaises(ValueError):
            smpar.vMh2_to_m2Lambda(-v, Mh2, C)
        with self.assertRaises(ValueError):
            smpar.vMh2_to_m2Lambda(v, -Mh2, C)
        with self.assertRaises(ValueError):
            smpar.m2Lambda_to_vMh2(-m2, Lambda, C)
        with self.assertRaises(ValueError):
            smpar.m2Lambda_to_vMh2(m2, -Lambda, C)

class TestSMpar(unittest.TestCase):
    def test_smeftpar_small(self):
        wc = get_random_wc('SMEFT', 'Warsaw', cmax=1e-24)
        smeft = SMEFT(wc=None)
        smeft.scale_in = 160
        smeft._set_initial_wcxf(wc, get_smpar=False)
        with self.assertRaises(ValueError):
            smpar.smeftpar(smeft.scale_in, smeft.C_in, 'flavio')
        CSM = smpar.smeftpar(smeft.scale_in, smeft.C_in, 'Warsaw')
        p = smpar.p
        self.assertAlmostEqual(CSM['m2'], p['m_h']**2/2)
        v = sqrt(1 / sqrt(2) / p['GF'])
        self.assertAlmostEqual(CSM['Lambda'], p['m_h']**2/v**2)
        self.assertAlmostEqual(CSM['g'], 2*p['m_W']/v)
        # self.assertAlmostEqual(CSM['gp'])
        self.assertAlmostEqual(CSM['gs'], sqrt(4*pi*p['alpha_s']))
        self.assertAlmostEqual(CSM['Gd'][0, 0], p['m_d']/(v/sqrt(2)))
        self.assertAlmostEqual(CSM['Gd'][1, 1], p['m_s']/(v/sqrt(2)))
        self.assertAlmostEqual(CSM['Gd'][2, 2], p['m_b']/(v/sqrt(2)))
        self.assertAlmostEqual(CSM['Ge'][0, 0], p['m_e']/(v/sqrt(2)))
        self.assertAlmostEqual(CSM['Ge'][1, 1], p['m_mu']/(v/sqrt(2)))
        self.assertAlmostEqual(CSM['Ge'][2, 2], p['m_tau']/(v/sqrt(2)))
        UL, S, UR = ckmutil.diag.msvd(CSM['Gu'])
        V = UL.conj().T
        self.assertAlmostEqual(S[0], p['m_u']/(v/sqrt(2)))
        self.assertAlmostEqual(S[1], p['m_c']/(v/sqrt(2)))
        self.assertAlmostEqual(S[2], p['m_t']/(v/sqrt(2)))
        self.assertAlmostEqual(abs(V[0, 2]), p['Vub'])
        self.assertAlmostEqual(abs(V[1, 2]), p['Vcb'])
        self.assertAlmostEqual(abs(V[0, 1]), p['Vus'])

    def test_smpar_small(self):
        wc = get_random_wc('SMEFT', 'Warsaw', cmax=1e-24)
        smeft = SMEFT(wc=None)
        smeft.scale_in = 160
        smeft._set_initial_wcxf(wc, get_smpar=False)
        CSM = smpar.smeftpar(smeft.scale_in, smeft.C_in, 'Warsaw')
        Cboth = CSM.copy()
        Cboth.update(smeft.C_in)
        Cback = smpar.smpar(Cboth)
        for k in smpar.p:
            if k not in ['m_Z', 'delta']:
                self.assertAlmostEqual(smpar.p[k], Cback[k],
                                       msg="Failed for {}".format(k))


    def test_smpar_roundtrip(self):
        wc = get_random_wc('SMEFT', 'Warsaw', cmax=1e-6)
        smeft = SMEFT(wc=None)
        smeft.scale_in = 160
        smeft._set_initial_wcxf(wc, get_smpar=False)
        CSM = smpar.smeftpar(smeft.scale_in, smeft.C_in, 'Warsaw')
        Cboth = CSM.copy()
        Cboth.update(smeft.C_in)
        Cback = smpar.smpar(Cboth)
        for k in smpar.p:
            if k in ['m_Z']:
                self.assertAlmostEqual(smpar.p[k]/Cback[k], 1,
                                       msg="Failed for {}".format(k),
                                       delta=0.05)
            elif k in ['delta']:
                self.assertAlmostEqual(smpar.p[k]/Cback[k], 1,
                                       msg="Failed for {}".format(k),
                                       delta=1e-3)
            else:
                self.assertAlmostEqual(smpar.p[k]/Cback[k], 1,
                                       msg="Failed for {}".format(k),
                                       delta=1e-5)

class TestGetSMpar(unittest.TestCase):
    def test_wcxf_smpar(self):
        wc = get_random_wc('SMEFT', 'Warsaw', 1e5, 1e-11)
        smeft = SMEFT(wc)
        C_out = smeft._rgevolve(91.1876)
        p_out = smpar.smpar(C_out)
        for k in p_out:
            self.assertAlmostEqual(p_out[k] / smpar.p[k], 1,
                                   delta=0.05,
                                   msg="Failed for {}".format(k))

    def test_wcxf_smpar_incomplete(self):
        wc = wcxf.WC('SMEFT', 'Warsaw', 160, {'qd1_1111': {'Im': 1e-6}})
        smeft = SMEFT(wc)
        smeft.run(91.1876)

    def test_method(self):
        wc = wcxf.WC('SMEFT', 'Warsaw', 1000, {'uG_33': 1e-6})
        smeft = SMEFT(wc)
        p_out = smeft.get_smpar()
        p_def = smpar.p
        for k, v in p_out.items():
            self.assertAlmostEqual(v / p_def[k], 1, places=1,
                                   msg="Failed for {}".format(k))
