import json
import yaml
import logging
from collections import OrderedDict, Counter
import tempfile
import shutil
import os
import subprocess
from pandas import DataFrame
import random

# the following is necessary to get pretty representations of
# OrderedDict and defaultdict instances in YAML
def _represent_dict_order(self, data):
    return self.represent_mapping('tag:yaml.org,2002:map', data.items())
yaml.add_representer(OrderedDict, _represent_dict_order)

# the following is necessary to have Null values be represented by
# emptyness rather than 'null'
def represent_none(self, _):
    return self.represent_scalar('tag:yaml.org,2002:null', '')
yaml.add_representer(type(None), represent_none)

# the following is necessary to load YAML mappings as OrderedDicts
_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
def _dict_constructor(loader, node):
    return OrderedDict(loader.construct_pairs(node))
yaml.add_constructor(_mapping_tag, _dict_constructor)

def _load_yaml_json(stream, **kwargs):
    """Load a JSON or YAML file from a string or stream."""
    if isinstance(stream, str):
        ss = stream
    else:
        ss = stream.read()
    try:
        return json.loads(ss, **kwargs)
    except ValueError:
        return yaml.safe_load(ss, **kwargs)

def _dump_json(d, stream=None, **kwargs):
    """Dump to a JSON string (if `stream` is None) or stream."""
    if stream is not None:
        return json.dump(d, stream, **kwargs)
    else:
        return json.dumps(d, **kwargs)

def _yaml_to_json(stream_in, stream_out, **kwargs):
    d = yaml.safe_load(stream_in)
    return _dump_json(d, stream_out, **kwargs)

def _json_to_yaml(stream_in, stream_out, **kwargs):
    d = json.load(stream_in)
    return yaml.dump(d, stream_out, **kwargs)


def _testtex(s, delete=True):
    """Function that takes a string and tries to compile a LaTeX file with
    the string as document body. Returns a dictionary with keys 'success'
    (True or False) and 'log', trying to extract the first error message."""
    _preamble = r"""\documentclass{article}
    \usepackage{amsmath,amssymb}
    \begin{document}
    """
    _enddoc = r"""
    \end{document}"""
    doc = _preamble + s + _enddoc
    tmpd = tempfile.mkdtemp()
    tmpf = os.path.join(tmpd, 'textest.tex')
    with open(tmpf, 'w') as f:
        f.write(doc)
    try:
        p = subprocess.run(['latex', '-halt-on-error',
                            '-output-directory', tmpd, tmpf],
                           stdout=subprocess.DEVNULL)
    except FileNotFoundError:
        logging.warn('latex executable not found. Cannot check tex code')
        return {'success': True}
    if p.returncode == 0:
        res = {'success': True}
    else:
        res = {'success': False}
    logf = os.path.join(tmpd, 'textest.log')
    res['log'] = ''
    try:
        with open(logf, 'r') as f:
            logl = f.readlines()
    except FileNotFoundError:
        pass
    fail = False
    for i, l in enumerate(logl):
        if 'Undefined control sequence' in l or 'Emergency stop' in l:
            fail = True
            break
    if fail:
        l = logl[i]
        while l.strip() != '':
            res['log'] += logl[i]
            i += 1
            try:
                l = logl[i]
            except IndexError:
                break
    if delete:
        shutil.rmtree(tmpd)
    return res


class NamedInstanceMetaclass(type):
    # this is just needed to implement the getitem method on NamedInstanceClass
    # to allow the syntax MyClass['instancename'] as shorthand for
    # MyClass.get_instance('instancename'); same for
    # del MyClass['instancename'] instead of MyClass.del_instance('instancename')
    def __getitem__(cls, item):
        return cls.get_instance(item)

    def __delitem__(cls, item):
        return cls.del_instance(item)

class NamedInstanceClass(object, metaclass=NamedInstanceMetaclass):
    """Base class for classes that have named instances that can be accessed
    by their name.

    Parameters
    ----------
     - name: string

    Methods
    -------
     - del_instance(name)
         Delete an instance
     - get_instance(name)
         Get an instance
     - set_description(description)
         Set the description
    """

    def __init__(self, _name):
        if not hasattr(self.__class__, 'instances'):
            self.__class__.instances = OrderedDict()
        self.__class__.instances[_name] = self
        self._name = _name

    @classmethod
    def get_instance(cls, _name):
        return cls.instances[_name]

    @classmethod
    def del_instance(cls, _name):
        del cls.instances[_name]

    @classmethod
    def clear_all(cls):
        """Delete all instances."""
        cls.instances = OrderedDict()


class WCxf(object):
    """Base class for WCxf files (not meant to be used directly)."""

    @classmethod
    def load(cls, stream, **kwargs):
        """Load the object data from a JSON or YAML file."""
        wcxf = _load_yaml_json(stream, **kwargs)
        return cls(**wcxf)

    def dump(self, stream=None, fmt='json', **kwargs):
        """Dump the object data to a JSON or YAML file.

        Optional arguments:

        - `stream`: if None (default), return a string. Otherwise,
          should be a writable file-like object
        - `fmt`: format, should be 'json' (default) or 'yaml'

        Additional keyword arguments will be passed to the `json.dump(s)`
        or `yaml.dump` methods.
        """
        d = {k: v for k,v in self.__dict__.items() if k[0] != '_'}
        if fmt.lower() == 'json':
            # set indent=2 unless specified otherwise
            indent = kwargs.pop('indent', 2)
            return _dump_json(d, stream=stream,
                              indent=indent,
                              **kwargs)
        elif fmt.lower() == 'yaml':
            # set default_flow_style=False unless specified otherwise
            default_flow_style = kwargs.pop('default_flow_style', False)
            return yaml.dump(d, stream,
                             default_flow_style=default_flow_style,
                            **kwargs)
        else:
            raise ValueError("Format {} unknown: use 'json' or 'yaml'.".format(fmt))

class EFT(WCxf, NamedInstanceClass):
    """Class representing EFT files."""
    def __init__(self, eft, sectors, **kwargs):
        """Instantiate the EFT file object."""
        self.eft = eft
        self.sectors = sectors
        super().__init__(eft)
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def known_bases(self):
        """Return a list of known bases for this EFT."""
        return tuple(basis[1] for basis in Basis.instances if basis[0] == self.eft)

    @property
    def known_translators(self):
        """Return a list of known translators between bases of this EFT."""
        return tuple(t for t in Translator.instances if t[0] == self.eft)


class Basis(WCxf, NamedInstanceClass):
    """Class representing basis files."""
    def __init__(self, eft, basis, sectors, **kwargs):
        """Instantiate the basis file object."""
        self.eft = eft
        self.basis = basis
        super().__init__((eft, basis))
        for k, v in kwargs.items():
            setattr(self, k, v)
        if hasattr(self, 'parent'):
            try:
                self.sectors = Basis[self.eft, self.parent].sectors.copy()
                self.sectors.update(sectors)
            except (AttributeError, KeyError):
                raise ValueError("Parent basis {} not found".format(self.parent))
        else:
            self.sectors = sectors
        self._all_wcs = None

    @property
    def known_translators(self):
        """Return a list of known translators to and from this basis."""
        kt = {}
        kt['from'] = tuple(t for t in Translator.instances
                           if t[0] == self.eft and t[1] == self.basis)
        kt['to'] = tuple(t for t in Translator.instances
                         if t[0] == self.eft and t[2] == self.basis)
        return kt

    @property
    def all_wcs(self):
        """Return a list with all Wilson coefficients defined in this basis."""
        if self._all_wcs is None:
            self._all_wcs = [wc for sector, wcs in self.sectors.items() for wc in wcs]
        return self._all_wcs

    def validate(self):
        """Validate the basis file."""
        try:
            eft_instance = EFT[self.eft]
        except (AttributeError, KeyError):
            raise ValueError("EFT {} not defined".format(self.eft))
        unknown_sectors = set(self.sectors.keys()) - set(eft_instance.sectors.keys())
        if unknown_sectors:
            raise ValueError("Unknown sectors: {}".format(unknown_sectors))
        all_keys = [k for s in self.sectors.values() for k, v in s.items()]
        if len(all_keys) != len(set(all_keys)):  # we have duplicate keys!
            cnt = Counter(all_keys)
            dupes = [k for k, v in cnt.items() if v > 1]
            raise ValueError("Duplicate coefficients in different sectors:"
                             " {}".format(dupes))
        # check for LaTeX errors
        # string with all tex values
        alltex = '${}$'.format('$, $'.join([d.get('tex', '')
                                            for c in self.sectors.values()
                                            for d in c.values()
                                            if d is not None]))
        res = _testtex(alltex)
        if not res['success']:
            raise ValueError("Validation of basis {}/{}: "
                             .format(self.eft, self.basis)
                             + "LaTeX compilation errors encountered:\n"
                             + "{}".format(res['log']))

    def __repr__(self):
        return "wcxf.Basis('{}', '{}', {{...}})".format(self.eft, self.basis)

    def _repr_markdown_(self):
        md = "# Basis `{}` (EFT `{}`)\n\n".format(self.basis, self.eft)
        if hasattr(self, 'metadata') and 'description' in self.metadata:
            md += self.metadata['description'] + "\n\n"
        return md

    def _markdown_tables(self):
        md = "## Sectors\n\n"
        md += "The effective Lagrangian is defined as\n"
        # general definition of Leff according to WCxf papaer
        md += (r"$$\mathcal L_\text{eff} = -\mathcal H_\text{eff} ="
               r"\sum_{O_i= O_i^\dagger} C_i \, O_i + \sum_{O_i\neq O_i^\dagger}"
               r"\left( C_i \, O_i + C^*_i \, O^\dagger_i\right).$$")
        md += "\n\n"
        for s, wcs in self.sectors.items():
            md += "### `{}`\n\n".format(s)
            if wcs:
                md += "| WC name | Operator | Type |\n"
                # NB: this is meant for pandoc; it computes column widths
                # in latex by counting the number of "-" separators as
                # fractions of line width (default: 72)
                md += "|" + 18 * "-" + "|" + 48 * "-" + "|" + 6 * "-" + "|\n"
                for name, d in wcs.items():
                    if d is None:
                        d = {}
                    if 'real' not in d or not d['real']:
                        t = 'C'
                    else:
                        t = 'R'
                    md += "| `{}` | ${}$ | {} |\n".format(name,
                                                          d.get('tex', '~'), t)
        return md

    def __str__(self):
        md = self._repr_markdown_()
        md += self._markdown_tables()
        return md


class WC(WCxf):
    """Class representing Wilson coefficient files."""
    def __init__(self, eft, basis, scale, values, **kwargs):
        """Instantiate the Wilson coefficient file object."""
        self.eft = eft
        self.basis = basis
        self.scale = float(scale)
        self.values = values
        self._dict = None
        self._df = None
        self._hash = hash(random.random())
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __hash__(self):
        return self._hash

    @staticmethod
    def _to_number(v):
        """Turn a Wilson coefficient value - that could be a number or a Re/Im
        dict - into a number."""
        if isinstance(v, dict):
            return float(v.get('Re', 0)) + 1j*float(v.get('Im', 0))
        else:
            return float(v)

    @staticmethod
    def _to_complex_dict(v):
        """Turn a numeric Wilson coefficient value into a Re/Im dict if it is
        complex."""
        if v.imag != 0:
            return {'Re': float(v.real), 'Im': float(v.imag)}
        else:
            return float(v.real)

    @classmethod
    def dict2values(cls, d):
        return {k: cls._to_complex_dict(v) for k, v in d.items()}

    def __getitem__(self, key):
        try:
            return self.dict[key]
        except KeyError:
            return 0

    def validate(self):
        """Validate the Wilson coefficient file."""
        try:
            eft_instance = EFT[self.eft]
        except (AttributeError, KeyError):
            raise ValueError("EFT {} not defined".format(self.eft))
        try:
            basis_instance = Basis[self.eft, self.basis]
        except (AttributeError, KeyError):
            raise ValueError("Basis {} not defined for EFT {}".format(self.basis, self.eft))
        unknown_keys = set(self.values.keys()) - set(basis_instance.all_wcs)
        assert unknown_keys == set(), \
            "Wilson coefficients do not exist in this basis: " + str(unknown_keys)

    @property
    def dict(self):
        """Return a dictionary with the Wilson coefficient values.
        The dictionary will be cached when called for the first time."""
        if self._dict is None:
            self._dict = {k: self._to_number(v) for k, v in self.values.items()}
        return self._dict

    @property
    def df(self):
        """Return a pandas.DataFrame with the Wilson coefficient values
        split by real and imaginary part.
        The DataFrame will be cached when called for the first time."""
        if self._df is None:
            re = [self.dict[k].real for k in self.values]
            im = [self.dict[k].imag for k in self.values]
            self._df = DataFrame({'Re': re, 'Im': im},
                                 index=self.values,
                                 columns=('Re', 'Im'))
        return self._df

    def __repr__(self):
        return ("wcxf.WC(eft='{}', basis='{}', scale='{}', values={{...}})"
                .format(self.eft, self.basis, self.scale))

    def __str__(self):
        return self._repr_markdown_()

    def _repr_markdown_(self):
        md = "## WCxf Wilson coefficients\n\n"
        md += "**EFT:** `{}`\n\n".format(self.eft)
        md += "**Basis:** `{}`\n\n".format(self.basis)
        md += "**Scale:** {} GeV\n\n".format(self.scale)
        md += "### Values\n\n"
        md += "| WC name | Value |\n"
        # NB: this is meant for pandoc; it computes column widths
        # in latex by counting the number of "-" separators as
        # fractions of line width (default: 72)
        md += "|" + 20 * "-" + "|" + 52 * "-" + "|\n"
        for name, v in self.dict.items():
            md += "| `{}` | {} |\n".format(name, v)
        return md

    def _repr_html_(self):
        html = "<h3>WCxf Wilson coefficients</h3>\n\n"
        html += """<table>
  <thead>
    <tr>
      <th>EFT</th>
      <th>Basis</th>
      <th>scale</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>{}</code></td>
      <td><code>{}</code></td>
      <td>{} GeV</td>
    </tr>
  </tbody>
</table>
""".format(self.eft, self.basis, self.scale)
        html += "<h4>Values</h4>\n\n"
        html += self.df._repr_html_()
        return html

    def translate(self, to_basis, parameters=None, sectors=None):
        """Translate the Wilson coefficients to a different basis.
        Returns a WC instance.

        Parameters:
        - `to_basis`: name of output basis
        - parameters: an optional dictionary of parameters specific to the
          translation function
        - sectors: an optional iterable of sector names of interest that the
          translator function may choose (but is not obliged) to limit itself
          to in the output."""
        if to_basis == self.basis:
            return self  # nothing to do
        try:
            translator = Translator[self.eft, self.basis, to_basis]
        except (KeyError, AttributeError):
            raise ValueError("No translator from basis {} to {} found.".format(self.basis, to_basis))
        return translator.translate(self, parameters=parameters, sectors=sectors)

    def match(self, to_eft, to_basis, parameters=None):
        """Match the Wilson coefficients to a different EFT.
        Returns a WC instance."""
        if to_eft == self.eft and to_basis == self.basis:
            return self  # nothing to do
        try:
            matcher = Matcher[self.eft, self.basis, to_eft, to_basis]
        except (KeyError, AttributeError):
            raise ValueError("No matcher from EFT {} in basis {} to EFT {} in basis {} found.".format(self.eft, self.basis, to_eft, to_basis))
        return matcher.match(self, parameters=parameters)


class Translator(NamedInstanceClass):
    """Class for translating between different bases of the same EFT."""
    def __init__(self, eft, from_basis, to_basis, function):
        """Initialize the Translator instance."""
        super().__init__((eft, from_basis, to_basis))
        self.eft = eft
        self.from_basis = from_basis
        self.to_basis = to_basis
        self.function = function

    def translate(self, WC_in, parameters=None, sectors=None):
        r"""Translate a WC object from `from_basis` to `to_basis`.

        Parameters:
        - `WC_in`: the input `WC` instance
        - parameters: an optional dictionary of parameters specific to the
          translation function
        - sectors: an optional iterable of sector names of interest that the
          translator function may choose (but is not obliged) to limit itself
          to in the output."""
        if sectors is None:
            dict_out = self.function(WC_in.dict, WC_in.scale, parameters)
        else:
            dict_out = self.function(WC_in.dict, WC_in.scale, parameters, sectors=sectors)
        # filter out zero values
        dict_out = {k: v for k, v in dict_out.items() if v != 0}
        values = WC.dict2values(dict_out)
        WC_out = WC(self.eft, self.to_basis, WC_in.scale, values)
        return WC_out


class Matcher(NamedInstanceClass):
    """Class for matching from a UV to an IR EFT."""
    def __init__(self, from_eft, from_basis, to_eft, to_basis, function):
        """Initialize the Matcher instance."""
        super().__init__((from_eft, from_basis, to_eft, to_basis))
        self.from_eft = from_eft
        self.from_basis = from_basis
        self.to_eft = to_eft
        self.to_basis = to_basis
        self.function = function

    def match(self, WC_in, parameters=None):
        """Translate a WC object in EFT `from_eft` and basis `from_basis`
        to EFT `to_eft` and basis `to_basis`."""
        dict_out = self.function(WC_in.dict, WC_in.scale, parameters)
        # filter out zero values
        dict_out = {k: v for k, v in dict_out.items() if v != 0}
        values = WC.dict2values(dict_out)
        WC_out = WC(self.to_eft, self.to_basis, WC_in.scale, values)
        return WC_out

def parametrized(dec):
    """Decorator for a decorator allowing it to take arguments.
    See https://stackoverflow.com/a/26151604."""
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer

@parametrized
def translator(func, eft, from_basis, to_basis):
    """Decorator for basis translation functions.

    Usage:

    ```python
    @translator('myEFT', 'myBasis_from', 'myBasis_to')
    def myFunction(wc_dict_from):
        ... # do something
        return wc_dict_to
    ```
    """
    Translator(eft, from_basis, to_basis, func)
    return func


@parametrized
def matcher(func, from_eft, from_basis, to_eft, to_basis):
    """Decorator for matching functions.

    Usage:

    ```python
    @matcher('myEFT_from', 'myBasis_from', 'myEFT_to', 'myBasis_to')
    def myFunction(wc_dict_from):
        ... # do something
        return wc_dict_to
    ```
    """
    Matcher(from_eft, from_basis, to_eft, to_basis, func)
    return func
