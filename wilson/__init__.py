r"""`wilson` is a package for the renormalization group evolution, matching,
and basis translation of Wilson coefficients beyond the Standard Model.

It can deal with the full set of dimension-6 operators in the Standard Model
Effective Field Theory (SMEFT) above the electroweak scale as well as the
weak effective theory (WET) below the W mass.

The package makes heavy use of the [Wilson coefficient exchange format (WCxf)](https://wcxf.github.io).

More information can be found in the publication:

Jason Aeabischer, Jacky Kumar, David M. Straub:
"wilson: a Python package for the running and matching of Wilson coefficients above and below the electroweak scale"
"""


from ._version import __version__
from . import run
from . import util
from . import match
from . import translate
from . import parameters
from .classes import Wilson
