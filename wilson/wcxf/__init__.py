from .classes import *
from . import matchers
from . import translators
from wilson import util

# read all EFTs and bases from the wcxf-bases submodule

import os
import glob

_root = os.path.abspath(os.path.dirname(__file__))
all_efts = glob.glob(os.path.join(_root, 'bases', '*.eft.json'))
all_bases = glob.glob(os.path.join(_root, 'bases', '*.basis.json'))
child_bases = glob.glob(os.path.join(_root, 'bases', 'child', '*.basis.json'))

for eft in all_efts:
    with open(eft) as f:
        EFT.load(f)

for basis in all_bases + child_bases:
    with open(basis) as f:
        Basis.load(f)
