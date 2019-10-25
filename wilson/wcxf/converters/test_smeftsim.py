import unittest
import os
from tempfile import mkdtemp
import subprocess
import pylha
from shutil import rmtree
import wcxf


my_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(my_path, '..', 'data')


class TestSMEFTsim(unittest.TestCase):
    def test_wcxf2smeftsim(self):
        for testfile in ['test.Warsaw_mass.yml', 'test.Warsaw_mass_incomplete.yml']:
            tmpdir = mkdtemp()
            # use default settings
            res = subprocess.run(['wcxf2smeftsim',
                                  os.path.join(data_path, testfile)],
                                 cwd=tmpdir)
            # check return code
            self.assertEqual(res.returncode, 0, msg="Command failed")
            # check if file is present
            outf = os.path.join(tmpdir, 'wcxf2smeftsim_param_card.dat')
            self.assertTrue(os.path.isfile, outf)
            # check if can be imported as LHA
            with open(outf, 'r') as f:
                card = pylha.load(f)
            # check dict is not empty
            self.assertTrue(card)
            # remove tmpdir
            rmtree(tmpdir)

    def test_symm_fac(self):
        wc = wcxf.WC('SMEFT', 'Warsaw mass', 120,
                     {'ll_1111': 1e-6, 'll_1221': 4e-6})
        tmpdir = mkdtemp()
        with open(os.path.join(tmpdir, 'my_wcxf.yaml'), 'w') as f:
            wc.dump(f, fmt='yaml')
        res = subprocess.run(['wcxf2smeftsim', 'my_wcxf.yaml'],
                             cwd=tmpdir)
        # check if file is present
        outf = os.path.join(tmpdir, 'wcxf2smeftsim_param_card.dat')
        self.assertTrue(os.path.isfile, outf)
        with open(outf, 'r') as f:
            card = pylha.load(f)
        self.assertEqual(dict(card['BLOCK']['FRBlock']['values'])[691],
                         1e6 * 4e-6 / 2,  # symmetry factor of 1 / 2!
                         msg="Wrong value for ll_1221")
        self.assertEqual(dict(card['BLOCK']['FRBlock']['values'])[689],
                         1e6 * 1e-6,  # no symmetry factor!
                         msg="Wrong value for ll_1111")
        # remove tmpdir
        # rmtree(tmpdir)
