import unittest
from wilson.util import wet_jms
import numpy as np


class TestWETutil(unittest.TestCase):
    def test_scalar2array(self):
        d = {'bla_123': 3, 'blo': 5j}
        da = wet_jms._scalar2array(d)
        self.assertEqual(da['blo'], 5j)
        self.assertIn('bla', da)
        self.assertEqual(da['bla'].shape, (3, 3, 3))
        self.assertTrue(np.isnan(da['bla'][0, 0, 0]))
        self.assertEqual(da['bla'][0, 1, 2], 3)
