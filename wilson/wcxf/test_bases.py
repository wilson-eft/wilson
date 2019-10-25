import unittest
import wcxf

class TestBases(unittest.TestCase):
    def test_bases(self):
        for basis in wcxf.Basis.instances.values():
            try:
                basis.validate()
            except:
                self.fail("Basis {}-{} failed to validate".format(basis.eft, basis.basis))
