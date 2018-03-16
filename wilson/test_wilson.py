import unittest
import wcxf
import wilson

class TestWilson(unittest.TestCase):
    def test_class(self):
        wc = wcxf.WC('SMEFT', 'Warsaw', 1000, {'qd1_1123': 1})
        wi = wilson.Wilson(wc)
