"""Compare to 1-loop flavio QCD running as of flavio v0.25"""

from wilson.run import wet
import wcxf
import pkgutil
import unittest


class TestFlavio(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _f = pkgutil.get_data('wilson', 'run/wet/tests/data/flavio_wc_random_150.json')
        cls.wc_in = wcxf.WC.load(_f.decode('utf-8'))
        _f = pkgutil.get_data('wilson', 'run/wet/tests/data/flavio_wc_random_5.json')
        cls.wc_out = wcxf.WC.load(_f.decode('utf-8'))

    def test_validate(self):
        self.wc_in.validate()
        self.wc_out.validate()

    def test_run(self):
        wet_in = wet.WETrunner(self.wc_in.translate('Bern'), {'alpha_e': 0})
        wc_out_wet = wet_in.run(5).translate('flavio')
        wc_out_wet.validate()
        for k, v in self.wc_out.dict.items():
            self.assertAlmostEqual(v, wc_out_wet.dict[k],
                                   delta=0.1,
                                   msg="Failed for {}".format(k))
