import unittest
from collections import OrderedDict, defaultdict
import numpy as np
import numpy.testing as npt
import pylha
from smeftrunner import SMEFT
import pkgutil

smeft = SMEFT()
smeft.scale_in = 1e4
smeft.scale_high = 1e4

f1 = pkgutil.get_data('smeftrunner', 'tests/data/SMInput-CPV.dat').decode('utf-8')
f2 = pkgutil.get_data('smeftrunner', 'tests/data/WCsInput-CPV-SMEFT.dat').decode('utf-8')
smeft.load_initial((f1, f2,))

C_out = smeft.rgevolve(scale_out=160)

# get datas as a dictionary
def BLOCKdict(streams):
    d = {}
    for stream in streams:
        s = pylha.load(stream)
        if 'BLOCK' not in s:
            raise ValueError("No BLOCK found")
        d.update(s['BLOCK'])
    return d

py_results = BLOCKdict((smeft.dump(C_out),))

f = pkgutil.get_data('smeftrunner', 'tests/data/Output_SMEFTrunner.dat').decode('utf-8')
ma_results = BLOCKdict((f,))


class TestSMEFT(unittest.TestCase):
    def test_keys(self):
        # check the keys of blocks
        for k in ma_results:
            # SCALES, OPTIONS blocks not implemented yet
            if k not in py_results and k not in ['SCALES', 'OPTIONS']:
                for iv in ma_results[k]['values']:
                    # if key is not in py_results, values must be zero!
                    self.assertEqual(iv[-1], 0, msg='Entry {} in {} nonzero'.format(iv[:-1], k))


    def test_values(self):
        # check the values of the common blocks
        blocks_co = set(py_results.keys()) & set(ma_results.keys())
        defaultblocks = defaultdict(list)
        for block in blocks_co:
            if block in ['SCALES', 'OPTIONS']:
                continue # SCALES and OPTIONS not implemented yet
            #check dimension
            self.assertEqual(len(py_results[block]['values'][0]),
                             len(ma_results[block]['values'][0]))
            #check values
            # turn into dicts for easier comparison
            py_dict = {tuple(v[:-1]): v[-1] for v in py_results[block]['values']}
            ma_dict = {tuple(v[:-1]): v[-1] for v in ma_results[block]['values']}
            # py_dict should contain LESS (vanishing) OR EQUAL keys compared to ma_dict
            self.assertSetEqual(set(py_dict.keys())-set(ma_dict.keys()), set(), msg=block)
            for k in (set(py_dict.keys()) | set(ma_dict.keys())):
                if k not in py_dict:
                    self.assertAlmostEqual(ma_dict[k], 0, delta=1e-6,
                        msg='Entry {} in {} nonzero'.format(k, block))
                else:
                    self.assertAlmostEqual(ma_dict[k], py_dict[k],
                        delta=max(1e-3, 1e-5*py_dict[k]), # 1e-3 absolute, 1e-5 relative
                        msg='Entry {} in {} does not agree: {} != {}'.format(k, block, ma_dict[k], py_dict[k]))
