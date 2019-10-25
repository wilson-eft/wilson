import unittest
import wcxf
from wcxf.converters import dsixtools
import os
import subprocess


my_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(my_path, '..', 'data')


with open(os.path.join(data_path, 'WCsInput-CPV-SMEFT.dat'), 'r') as f:
    wcin_lha = f.read()
with open(os.path.join(data_path, 'Options.dat'), 'r') as f:
    options = f.read()
with open(os.path.join(data_path, 'SMInput-CPV.dat'), 'r') as f:
    smin = f.read()
with open(os.path.join(data_path, 'WCsInput-CPV-SMEFT.json'), 'r') as f:
    wcin_json = f.read()
with open(os.path.join(data_path, 'WCsInput-CPV-SMEFT.yaml'), 'r') as f:
    wcin_yaml = f.read()


class TestDsixTools(unittest.TestCase):
    def test_smeftio(self):
        smeftio = dsixtools.SMEFTio()
        smeftio.load_wcxf(wcin_json)
        smeftio.load_wcxf(wcin_yaml)
        smeftio.load_initial((wcin_lha, options, smin))

    def test_wcxf2dsixtools(self):
        wc = wcxf.WC.load(wcin_json)
        d1 = dsixtools.load(dsixtools.wcxf2dsixtools(wc))
        smeftio = dsixtools.SMEFTio()
        smeftio.load_initial((wcin_lha, options, smin))
        d2 = dsixtools.load(smeftio.dump(smeftio.C_in))
        for key in ['SCALES', 'OPTIONS']:
            d1['BLOCK'].pop(key, None)
            d2['BLOCK'].pop(key, None)
        self.assertDictEqual(d1, d2)

    def test_dsixtools2wcxf(self):
        smeftio = dsixtools.SMEFTio()
        smeftio.load_initial((wcin_lha, options, smin))
        d1 = wcxf.WC.load(dsixtools.dsixtools2wcxf((wcin_lha, options, smin))).dict
        d2 = wcxf.WC.load(wcin_yaml).dict
        self.assertTrue(d1)
        self.assertTrue(d2)
        self.assertEqual(set(d1.keys()), set(d2.keys()))
        for k in d1:
            self.assertEqual(d1[k], d2[k], msg="Failed for {}".format(k))

    def test_cli_wcxf2dsixtools(self):
        smeftio = dsixtools.SMEFTio()
        smeftio.load_initial((wcin_lha, options, smin))
        d1 = dsixtools.load(smeftio.dump(smeftio.C_in))
        res = subprocess.run(['wcxf2dsixtools', '-'],
                             input=wcin_json.encode(),
                             stdout=subprocess.PIPE)
        s = res.stdout.decode('utf-8')
        d2 = dsixtools.load(s)
        self.assertDictEqual(d1, d2)

    def test_cli_dsixtools2wcxf(self):
        d1 = wcxf.WC.load(wcin_yaml).dict
        res = subprocess.run(['dsixtools2wcxf',
                             os.path.join(data_path, 'SMInput-CPV.dat'),
                             os.path.join(data_path, 'WCsInput-CPV-SMEFT.dat'),
                             os.path.join(data_path, 'Options.dat')],
                             stdout=subprocess.PIPE)
        s = res.stdout.decode('utf-8')
        d2 = wcxf.WC.load(s).dict
        self.assertDictEqual(d1, d2)
