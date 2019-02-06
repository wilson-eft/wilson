import unittest
from wilson.util.qcd import alpha_s, m_b, m_c, m_s

# All numbers compared to Mathemetica version of RunDec

delta = 1e-8
deltam = 1e-4


class TestMb(unittest.TestCase):
    def test_m_b(self):
        self.assertAlmostEqual(m_b(4.2, 50, 5),
                               3.03526,
                               delta=deltam)
        self.assertAlmostEqual(m_b(4.2, 2, 4),
                               4.98814,
                               delta=deltam)
        self.assertAlmostEqual(m_b(4.2, 1, 3),
                               6.54785,
                               delta=deltam)
        self.assertAlmostEqual(m_b(4.2, 200, 6),
                               2.69632,
                               delta=deltam)
        self.assertAlmostEqual(m_b(4.163, 50, 5, alphasMZ=0.1181, loop=5),
                               3.00704,
                               delta=deltam)
        self.assertAlmostEqual(m_b(4.163, 2, 4, alphasMZ=0.1181, loop=5),
                               4.94031,
                               delta=deltam)
        self.assertAlmostEqual(m_b(4.163, 1, 3, alphasMZ=0.1181, loop=5),
                               6.55283,
                               delta=deltam)
        self.assertAlmostEqual(m_b(4.163, 200, 6, alphasMZ=0.1181, loop=5),
                               2.67213,
                               delta=deltam)


class TestMc(unittest.TestCase):
    def test_m_c(self):
        self.assertAlmostEqual(m_c(1.2, 50, 5),
                               0.612466,
                               delta=deltam)
        self.assertAlmostEqual(m_c(1.2, 2, 4),
                               1.00652,
                               delta=deltam)
        self.assertAlmostEqual(m_c(1.2, 1, 3),
                               1.32125,
                               delta=deltam)
        self.assertAlmostEqual(m_c(1.279, 50, 5, alphasMZ=0.1181, loop=5),
                               0.66840,
                               delta=deltam)
        self.assertAlmostEqual(m_c(1.279, 2, 4, alphasMZ=0.1181, loop=5),
                               1.09811,
                               delta=deltam)
        self.assertAlmostEqual(m_c(1.279, 1, 3, alphasMZ=0.1181, loop=5),
                               1.45654,
                               delta=deltam)


class TestMs(unittest.TestCase):
    def test_m_s(self):
        self.assertAlmostEqual(m_s(0.1, 1, 3),
                               0.131039,
                               delta=deltam)
        self.assertAlmostEqual(m_s(0.1, 2.5, 4),
                               0.094137,
                               delta=deltam)
        self.assertAlmostEqual(m_s(0.1, 50, 5),
                               0.0607461,
                               delta=deltam)
        self.assertAlmostEqual(m_s(0.095, 1, 3, alphasMZ=0.1181, loop=5),
                               0.12527,
                               delta=deltam)
        self.assertAlmostEqual(m_s(0.095, 2.5, 4, alphasMZ=0.1181, loop=5),
                               0.08904,
                               delta=deltam)
        self.assertAlmostEqual(m_s(0.095, 50, 5, alphasMZ=0.1181, loop=5),
                               0.05750,
                               delta=deltam)

class TestAlphas(unittest.TestCase):
    def test_alphas_invalid(self):
        with self.assertRaises(ValueError):
            alpha_s(100, 7)
        with self.assertRaises(ValueError):
            alpha_s(100, 2)
        with self.assertRaises(ValueError):
            alpha_s(100, 7)
        with self.assertRaises(ValueError):
            alpha_s(0, 6)
        with self.assertRaises(ValueError):
            alpha_s(-1, 6)
        with self.assertRaises(TypeError):
            alpha_s("1.0", 6)
        with self.assertRaises(ValueError):
            alpha_s(0.5, 3)

    def test_alphas_5(self):
        self.assertAlmostEqual(alpha_s(100, 5),
                               0.11686431884237730186,
                               delta=delta)
        self.assertAlmostEqual(alpha_s(10, 5),
                               0.17931693160062720703,
                               delta=delta)
        self.assertAlmostEqual(alpha_s(100, 5, alphasMZ=0.1181, loop=5),
                               0.1164747252483566,
                               delta=delta)
        self.assertAlmostEqual(alpha_s(10, 5, alphasMZ=0.1181, loop=5),
                               0.17846838859679,
                               delta=delta)
        # crazy values
        self.assertAlmostEqual(alpha_s(1, 5),
                               0.40957053067188524193,
                               delta=delta)
        self.assertAlmostEqual(alpha_s(1000, 5),
                               0.087076948997751428458,
                               delta=delta)

    def test_alphas_6(self):
        self.assertAlmostEqual(alpha_s(500, 6),
                               0.095517575136454583087,
                               delta=delta)
        self.assertAlmostEqual(alpha_s(500, 6, alphasMZ=0.1181, loop=5),
                               0.095272906877950202894,
                               delta=delta)
        # crazy values
        self.assertAlmostEqual(alpha_s(50, 6),
                               0.12785358110125187370,
                               delta=delta)


    def test_alphas_4(self):
        self.assertAlmostEqual(alpha_s(3, 4),
                               0.25604161478941490576,
                               delta=delta)
        self.assertAlmostEqual(alpha_s(3, 4, alphasMZ=0.1181, loop=5),
                               0.25391329528950183050,
                               delta=delta)
        # crazy values
        self.assertAlmostEqual(alpha_s(1, 4),
                               0.46414770696020787020,
                               delta=delta)
        self.assertAlmostEqual(alpha_s(1000, 4),
                               0.082139482683335368979,
                               delta=delta)


    def test_alphas_3(self):
        self.assertAlmostEqual(alpha_s(0.9, 3),
                               0.527089,
                               delta=1e-5)
        self.assertAlmostEqual(alpha_s(0.9, 3, alphasMZ=0.1181, loop=5),
                               0.517328,
                               delta=1e-5)
        # crazy values
        self.assertAlmostEqual(alpha_s(1000, 3),
                               0.076593079980776995496,
                               delta=delta)
