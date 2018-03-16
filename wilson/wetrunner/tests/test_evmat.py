"""Compare evolution matrices to v0.1 numerics"""

from wilson import wetrunner
import unittest
from pkg_resources import resource_filename
import numpy as np
import numpy.testing as npt


def getUs_new(classname):
    arg = (0.56, 5, 0.12, 1/127, 0, 0, 0, 1.2, 4.2, 0, 0, 1.8)
    return wetrunner.rge.getUs(classname, *arg)


def getUe_new(classname):
    arg = (0.56, 5, 0.12, 1/127, 0, 0, 0, 1.2, 4.2, 0, 0, 1.8)
    return wetrunner.rge.getUe(classname, *arg)


class TestEvMat(unittest.TestCase):
    def test_evmat(self):
        Usold = dict(np.load(resource_filename('wilson', 'wetrunner/tests/data/evmat_s_old.npz')))
        Ueold = dict(np.load(resource_filename('wilson', 'wetrunner/tests/data/evmat_e_old.npz')))
        Usnew = {k: getUs_new(k) for k in ['I', 'II', 'III', 'IV', 'Vb']}
        Uenew = {k: getUe_new(k) for k in ['I', 'II', 'III', 'IV', 'Vb']}
        Usnew['V'] = getUs_new('Vsb')
        Uenew['V'] = getUe_new('Vsb')
        for k in ['I', 'II', 'III', 'IV', 'V', 'Vb']:
            npt.assert_array_almost_equal(Usold[k], Usnew[k],
                                          err_msg="Failed for {} QCD".format(k))
        for k in ['I', 'II', 'III', 'IV', 'Vb']:  # NB, skipping V!
            npt.assert_array_almost_equal(100*Ueold[k], 100*Uenew[k],
                                          decimal=2,
                                          err_msg="Failed for {} QED".format(k))
