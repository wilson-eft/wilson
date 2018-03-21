from ._version import __version__
from . import run
from . import util
from . import match
from . import translate
from . import parameters
from .classes import Wilson

# Temporary: modified bases

import os
import glob
import wcxf

_root = os.path.abspath(os.path.dirname(__file__))
all_bases = glob.glob(os.path.join(_root, 'data/bases', '*.basis.json'))


for basis in all_bases:
    with open(basis, 'r') as f:
        wcxf.Basis.load(f)
