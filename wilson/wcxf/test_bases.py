import unittest
from wilson import wcxf

class TestBases(unittest.TestCase):
    def test_bases(self):
        for basis in wcxf.Basis.instances.values():
            try:
                basis.validate()
            except:
                self.fail(f"Basis {basis.eft}-{basis.basis} failed to validate")
