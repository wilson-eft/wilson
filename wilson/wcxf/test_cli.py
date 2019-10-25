import unittest
import numpy as np
import numpy.testing as npt
import yaml
import json
import pkgutil
import wcxf
import subprocess
import tempfile
import os

def _del_files(fs):
    """Delete files if they exist"""
    for f in fs:
        try:
            os.remove(f)
        except OSError:
            pass

class TestCLI(unittest.TestCase):
    def test_convert(self):
        yml1 = pkgutil.get_data('wcxf', 'data/test.basis1.yml').decode('utf-8')
        d_yml1 = yaml.safe_load(yml1)
        # YAML stdin -> JSON stdout
        res = subprocess.run(['wcxf', 'convert', 'json', '-'],
                             input=yml1.encode(),
                             stdout=subprocess.PIPE)
        json1 = res.stdout.decode('utf-8')
        d_json1 = json.loads(json1)
        self.assertDictEqual(d_yml1, d_json1)
        # JSON stdin -> YAML stdout
        res = subprocess.run(['wcxf', 'convert', 'yaml', '-'],
                             input=json1.encode(),
                             stdout=subprocess.PIPE)
        yml2 = res.stdout.decode('utf-8')
        d_yml2 = yaml.safe_load(yml2)
        self.assertDictEqual(d_json1, d_yml2)
        # YAML file -> JSON file
        _, fin = tempfile.mkstemp()
        with open(fin, 'w') as f:
            f.write(yml1)
        _, fout = tempfile.mkstemp()
        res = subprocess.run(['wcxf', 'convert', 'json', fin, '--output', fout])
        with open(fout, 'r') as f:
            d_json3 = json.load(f)
        self.assertDictEqual(d_yml1, d_json3)
        # delete temp files
        _del_files([fin, fout])
        # JSON file -> YAML file
        _, fin = tempfile.mkstemp()
        with open(fin, 'w') as f:
            f.write(json1)
        _, fout = tempfile.mkstemp()
        res = subprocess.run(['wcxf', 'convert', 'yaml', fin, '--output', fout])
        with open(fout, 'r') as f:
            yml3 = f.read()
        d_yml3 = yaml.safe_load(yml3)
        self.assertDictEqual(d_yml3, d_json1)
        # delete temp files
        _del_files([fin, fout])

    def test_validate(self):
        _root = os.path.abspath(os.path.dirname(__file__))
        wet = os.path.join(_root, 'bases', 'wet.eft.json')
        with open(wet, 'rb') as f:
            s = f.read()
        res = subprocess.run(['wcxf', 'validate', 'eft', '-'],
                             input=s,
                             stdout=subprocess.PIPE)
        res = res.stdout.decode('utf-8')
        self.assertEqual(res, "Validation successful.\n")

        _root = os.path.abspath(os.path.dirname(__file__))
        wet = os.path.join(_root, 'bases', 'wet.flavio.basis.json')
        with open(wet, 'rb') as f:
            s = f.read()
        res = subprocess.run(['wcxf', 'validate', 'basis', '-'],
                             input=s,
                             stdout=subprocess.PIPE)
        res = res.stdout.decode('utf-8')
        self.assertEqual(res, "Validation successful.\n")
