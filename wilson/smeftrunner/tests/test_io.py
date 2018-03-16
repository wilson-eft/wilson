import unittest
import numpy as np
import numpy.testing as npt
from wilson.smeftrunner import SMEFT, io
from wilson.util import smeftutil
import pkgutil
import pylha


class TestIO(unittest.TestCase):
    def test_lhamatrix(self):
        M = np.random.rand(2,3,4)
        values = io.matrix2lha(M)
        M2 = io.lha2matrix(values, (2,3,4))
        npt.assert_array_equal(M, M2)

    def test_load(self):
        sm = pkgutil.get_data('wilson', 'smeftrunner/tests/data/SMInput-CPV.dat').decode('utf-8')
        wc = pkgutil.get_data('wilson', 'smeftrunner/tests/data/WCsInput-CPV-SMEFT.dat').decode('utf-8')
        wcout = pkgutil.get_data('wilson', 'smeftrunner/tests/data/Output_SMEFTrunner.dat').decode('utf-8')
        io.sm_lha2dict(pylha.load(sm))
        io.wc_lha2dict(pylha.load(wc))
        CSM = io.sm_lha2dict(pylha.load(wcout))
        C = io.wc_lha2dict(pylha.load(wcout))
        C2 = io.wc_lha2dict(io.wc_dict2lha(C))
        for k in C:
            npt.assert_array_equal(C[k], C2[k])
        smeft = SMEFT()
        smeft.load_initial((wcout,))
        for k in C:
            npt.assert_array_equal(smeftutil.symmetrize(C)[k], smeft.C_in[k], err_msg="Failed for {}".format(k))
        for k in CSM:
            npt.assert_array_equal(smeftutil.symmetrize(CSM)[k], smeft.C_in[k], err_msg="Failed for {}".format(k))
        CSM2 = io.sm_lha2dict(io.sm_dict2lha(CSM))
        for k in CSM:
            npt.assert_array_equal(CSM[k], CSM2[k], err_msg="Failed for {}".format(k))

    def test_dump(self):
        wcout = pkgutil.get_data('wilson', 'smeftrunner/tests/data/Output_SMEFTrunner.dat').decode('utf-8')
        smeft = SMEFT()
        smeft.load_initial((wcout,))
        smeft.scale_in = 1000
        smeft.scale_high = 1000
        C_out = smeft.rgevolve(scale_out=900)
        C_dump = smeft.dump(C_out)
        smeft.load_initial((C_dump,))
        for k in C_out:
            npt.assert_array_almost_equal(C_out[k].real, smeft.C_in[k].real, err_msg="Failed for {}".format(k))


    def test_wcxf(self):
        # load example output file with SM par & WCs
        wcout = pkgutil.get_data('wilson', 'smeftrunner/tests/data/Output_SMEFTrunner.dat').decode('utf-8')
        smeft = SMEFT()
        smeft.load_initial((wcout,))
        smeft.C_in['uphi'] = np.zeros((3,3))
        smeft.C_in['dphi'] = np.zeros((3,3))
        smeft.scale_high = 10000
        # rotate to Md,l diagonal basis
        C_rot = smeft.rotate_defaultbasis(smeft.C_in)
        # re-create the SMEFT instance but now setting input in the Md,l diag basis
        smeft = SMEFT()
        smeft.set_initial(C_rot, scale_in=160, scale_high=10000)
        # dump to WCxf-YAML
        wcyaml = smeft.dump_wcxf(smeft.C_in, 160, default_flow_style=False)
        # new instance: first load SM parameters in Md,l diag basis
        smeft2 = SMEFT()
        C_ini = {k: v for k, v in smeft.C_in.items() if k in smeftutil.SM_keys}
        smeft2.set_initial(C_ini, scale_in=160, scale_high=10000)
        # now load back in the WCxf-YAML
        smeft2.load_wcxf(wcyaml, get_smpar=False)
        # now check that everything is equal to where we started from
        for k in smeft.C_in:
            if np.all(np.round(smeft.C_in[k], 6) == 0):
                continue # zero values are omitted in output
            if k in smeftutil.WC_keys_2f + smeftutil.WC_keys_4f:
                # compare matrices for fermionic op.s
                npt.assert_array_almost_equal(
                                 smeft.C_in[k], smeft2.C_in[k],
                                 err_msg="Failed for {}".format(k),
                                 decimal=1)
            if k in smeftutil.WC_keys_0f:
                # compare numbers for bosonic op.s
                self.assertAlmostEqual(
                                 smeft.C_in[k], smeft2.C_in[k],
                                 msg="Failed for {}".format(k),
                                 places=10)
