import numpy as np
from collections import OrderedDict, defaultdict
import pylha
import json
import yaml
from math import sqrt
import ckmutil
from wilson.util import smeftutil, smeft_warsaw
from . import dsixtools_definitions as definitions
import wilson


def load(stream, fmt='lha'):
    """Load a parameter file in DSixTools SLHA-like format or its JSON or
    YAML representation."""
    if fmt == 'lha':
        return pylha.load(stream)
    elif fmt == 'json':
        if isinstance(stream, str):
            return json.loads(stream)
        else:
            return json.load(stream)
    elif fmt == 'yaml':
        return yaml.safe_load(stream)

def lha2matrix(values, shape):
    """Return a matrix given a list of values of the form
    [[1, 1, float], [1, 2, float], ...]
    referring to the (1,1)-element etc.
    `shape` is the shape of the final matrix. All elements not provided
    will be assumed to be zero. Also works for higher-rank tensors."""
    M = np.zeros(shape)
    for v in values:
        M[tuple([int(i-1) for i in v[:-1]])] = v[-1]
    return M

def matrix2lha(M):
    """Inverse function to lha2matrix: return a LHA-like list given a tensor."""
    l = []
    ind = np.indices(M.shape).reshape(M.ndim, M.size).T
    for i in ind:
        l.append([j+1 for j in i] + [M[tuple(i)]])
    return l

def sm_lha2dict(lha):
    """Convert a dictionary returned by pylha from a DSixTools SM input file
    into a dictionary of SM values."""
    d = OrderedDict()
    v = dict(lha['BLOCK']['GAUGE']['values'])
    d['g'] = v[1]
    d['gp'] = v[2]
    d['gs'] = v[3]
    v = dict(lha['BLOCK']['SCALAR']['values'])
    d['Lambda'] = v[1]
    d['m2'] = v[2]
    d['Gu'] = lha2matrix(lha['BLOCK']['GU']['values'], (3,3))
    if 'IMGU' in lha['BLOCK']:
        d['Gu'] = d['Gu'] + 1j*lha2matrix(lha['BLOCK']['IMGU']['values'], (3,3))
    d['Gd'] = lha2matrix(lha['BLOCK']['GD']['values'], (3,3))
    if 'IMGD' in lha['BLOCK']:
        d['Gd'] = d['Gd'] + 1j*lha2matrix(lha['BLOCK']['IMGD']['values'], (3,3))
    d['Ge'] = lha2matrix(lha['BLOCK']['GE']['values'], (3,3))
    if 'IMGE' in lha['BLOCK']:
        d['Ge'] = d['Ge'] + 1j*lha2matrix(lha['BLOCK']['IMGE']['values'], (3,3))
    # thetas default to 0
    if 'THETA' in lha['BLOCK']:
        v = dict(lha['BLOCK']['THETA']['values'])
        d['Theta'] = v.get(1, 0)
        d['Thetap'] = v.get(2, 0)
        d['Thetas'] = v.get(3, 0)
    else:
        d['Theta'] = 0
        d['Thetap'] = 0
        d['Thetas'] = 0
    return d

def sm_dict2lha(d):
    """Convert a a dictionary of SM parameters into
    a dictionary that pylha can convert into a DSixTools SM output file."""
    blocks = OrderedDict([
        ('GAUGE', {'values': [[1, d['g'].real], [2, d['gp'].real], [3, d['gs'].real]]}),
        ('SCALAR', {'values': [[1, d['Lambda'].real], [2, d['m2'].real]]}),
        ('GU', {'values': matrix2lha(d['Gu'].real)}),
        ('IMGU', {'values': matrix2lha(d['Gu'].imag)}),
        ('GD', {'values': matrix2lha(d['Gd'].real)}),
        ('IMGD', {'values': matrix2lha(d['Gd'].imag)}),
        ('GE', {'values': matrix2lha(d['Ge'].real)}),
        ('IMGE', {'values': matrix2lha(d['Ge'].imag)}),
        ('THETA', {'values': [[1, d['Theta'].real], [2, d['Thetap'].real], [3, d['Thetas'].real]]}),
        ])
    return {'BLOCK': blocks}

# dictionary necessary for translating to DSixTools IO format
WC_dict_0f = OrderedDict([
('G', ('WC1', 1)),
('Gtilde', ('WC1', 2)),
('W', ('WC1', 3)),
('Wtilde', ('WC1', 4)),
('phi', ('WC2', 1)),
('phiBox', ('WC3', 1)),
('phiD', ('WC3', 2)),
('phiG', ('WC4', 1)),
('phiB', ('WC4', 2)),
('phiW', ('WC4', 3)),
('phiWB', ('WC4', 4)),
('phiGtilde', ('WC4', 5)),
('phiBtilde', ('WC4', 6)),
('phiWtilde', ('WC4', 7)),
('phiWtildeB', ('WC4', 8)),
])

def lha2scale(lha):
    """Extract the high scale from a dictionary eturned by pylha from
    a DSixTools options file."""
    return dict(lha['BLOCK']['SCALES']['values'])[1]

def wc_lha2dict(lha):
    """Convert a dictionary returned by pylha from a DSixTools WC input file
    into a dictionary of Wilson coefficients."""
    C = OrderedDict()
    # try to read all WCs with 0, 2, or 4 fermions; if not found, set to zero
    for k, (block, i) in WC_dict_0f.items():
        try:
            C[k] = dict(lha['BLOCK'][block]['values'])[i]
        except KeyError:
            C[k] = 0
    for k in smeft_warsaw.WC_keys_2f:
        try:
            C[k] = lha2matrix(lha['BLOCK']['WC' + k.upper()]['values'], (3,3)).real
        except KeyError:
            C[k] = np.zeros((3,3))
        try: # try to add imaginary part
            C[k] = C[k] + 1j*lha2matrix(lha['BLOCK']['IMWC' + k.upper()]['values'], (3,3))
        except KeyError:
            pass
    for k in smeft_warsaw.WC_keys_4f:
        try:
            C[k] = lha2matrix(lha['BLOCK']['WC' + k.upper()]['values'], (3,3,3,3))
        except KeyError:
            C[k] = np.zeros((3,3,3,3))
        try: # try to add imaginary part
            C[k] = C[k] + 1j*lha2matrix(lha['BLOCK']['IMWC' + k.upper()]['values'], (3,3,3,3))
        except KeyError:
            pass
    return C

def wc_dict2lha(wc, skip_redundant=True, skip_zero=True):
    """Convert a a dictionary of Wilson coefficients into
    a dictionary that pylha can convert into a DSixTools WC output file."""
    d = OrderedDict()
    for name, (block, i) in WC_dict_0f.items():
        if block not in d:
            d[block] = defaultdict(list)
        if wc[name] != 0:
            d[block]['values'].append([i, wc[name].real])
    for name in smeft_warsaw.WC_keys_2f:
        reblock = 'WC'+name.upper()
        imblock = 'IMWC'+name.upper()
        if reblock not in d:
            d[reblock] = defaultdict(list)
        if imblock not in d:
            d[imblock] = defaultdict(list)
        for i in range(3):
            for j in range(3):
                if (i, j) in definitions.redundant_elements[name] and skip_redundant:
                    # skip redundant elements
                    continue
                if wc[name][i, j].real != 0 or not skip_zero:
                    d[reblock]['values'].append([i+1, j+1, float(wc[name][i, j].real)])
                if wc[name][i, j].imag != 0 or not skip_zero:
                    # omit Im parts that have to vanish by symmetry
                    if (i, j) not in definitions.vanishing_im_parts[name]:
                        d[imblock]['values'].append([i+1, j+1, float(wc[name][i, j].imag)])
    for name in smeft_warsaw.WC_keys_4f:
        reblock = 'WC'+name.upper()
        imblock = 'IMWC'+name.upper()
        if reblock not in d:
            d[reblock] = defaultdict(list)
        if imblock not in d:
            d[imblock] = defaultdict(list)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        if (i, j, k, l) in definitions.redundant_elements[name] and skip_redundant:
                            # skip redundant elements
                            continue
                        if wc[name][i, j, k, l].real != 0 or not skip_zero:
                            d[reblock]['values'].append([i+1, j+1, k+1, l+1, float(wc[name][i, j, k, l].real)])
                        if wc[name][i, j, k, l].imag != 0 or not skip_zero:
                            # omit Im parts that have to vanish by symmetry
                            if (i, j, k, l) not in definitions.vanishing_im_parts[name]:
                                d[imblock]['values'].append([i+1, j+1, k+1, l+1, float(wc[name][i, j, k, l].imag)])
    # remove empty blocks
    empty = []
    for block in d:
        if d[block] == {}:
            empty.append(block)
    for block in empty:
        del d[block]
    return {'BLOCK': d}


class SMEFTio:

    def __init__(self):
        """Initialize the SMEFTio instance."""
        self.C_in = None
        self.scale_in = None
        self.scale_high = None

    def set_initial(self, C_in, scale_in, scale_high):
        r"""Set the initial values for parameters and Wilson coefficients at
        the scale `scale_in`, setting the new physics scale $\Lambda$ to
        `scale_high`."""
        self.C_in = C_in
        self.scale_in = scale_in
        self.scale_high = scale_high

    def load_initial(self, streams):
        """Load the initial values for parameters and Wilson coefficients from
        one or several files.

        `streams` should be a tuple of file-like objects strings."""
        d = {}
        for stream in streams:
            s = load(stream)
            if 'BLOCK' not in s:
                raise ValueError("No BLOCK found")
            d.update(s['BLOCK'])
        d = {'BLOCK': d}
        C = wc_lha2dict(d)
        sm = sm_lha2dict(d)
        self.scale_high = lha2scale(d)
        self.scale_in = lha2scale(d)
        C.update(sm)
        C = smeftutil.symmetrize(C)
        self.C_in = C

    def set_initial_wcxf(self, wc, scale_high=None):
        """Load the initial values for Wilson coefficients from a
        wcxf.WC instance.

        Parameters:

        - `scale_high`: since Wilson coefficients are dimensionless in
          DsixTools but not in WCxf, the high scale in GeV has to be provided.
          If this parameter is None (default), either a previously defined
          value will be used, or the scale attribute of the WC instance will
          be used.
        """
        from wilson import wcxf
        if wc.eft != 'SMEFT':
            raise ValueError("Wilson coefficients use wrong EFT.")
        if wc.basis != 'Warsaw':
            raise ValueError("Wilson coefficients use wrong basis.")
        if scale_high is not None:
            self.scale_high = scale_high
        elif self.scale_high is None:
            self.scale_high = wc.scale
        C = smeftutil.wcxf2arrays(wc.dict)
        keys_dim5 = ['llphiphi']
        keys_dim6 = list(set(smeft_warsaw.WC_keys_0f + smeft_warsaw.WC_keys_2f + smeft_warsaw.WC_keys_4f) - set(keys_dim5))
        self.scale_in = wc.scale
        for k in keys_dim5:
            if k in C:
                C[k] = C[k]*self.scale_high
        for k in keys_dim6:
            if k in C:
                C[k] = C[k]*self.scale_high**2
        C = smeftutil.symmetrize(C)
        # fill in zeros for missing WCs
        for k, s in smeftutil.C_keys_shape.items():
            if k not in C and k not in smeft_warsaw.SM_keys:
                if s == 1:
                    C[k] = 0
                else:
                    C[k] = np.zeros(s)
        if self.C_in is None:
            self.C_in = C
        else:
            self.C_in.update(C)

    def load_wcxf(self, stream):
        """Load the initial values for Wilson coefficients from
        a file-like object or a string in WCxf format.

        Note that Standard Model parameters have to be provided separately
        and are assumed to be in the weak basis used for the Warsaw basis as
        defined in WCxf, i.e. in the basis where the down-type and charged
        lepton mass matrices are diagonal."""
        from wilson import wcxf
        wc = wcxf.WC.load(stream)
        self.set_initial_wcxf(wc)

    def dump(self, C_out, scale_out=None, stream=None, fmt='lha', skip_redundant=True):
        """Return a string representation of the parameters and Wilson
        coefficients `C_out` in DSixTools output format. If `stream` is
        specified, export it to a file. `fmt` defaults to `lha` (the SLHA-like
        DSixTools format), but can also be `json` or `yaml` (see the
        pylha documentation)."""
        C = OrderedDict()
        if scale_out is not None:
            C['SCALES'] = {'values': [[1, self.scale_high], [2, scale_out]]}
        else:
            C['SCALES'] = {'values': [[1, self.scale_high]]}
        # sm = sm_dict2lha(C_out)['BLOCK']
        # C.update(sm)
        wc = wc_dict2lha(C_out, skip_redundant=skip_redundant)['BLOCK']
        C.update(wc)
        return pylha.dump({'BLOCK': C}, fmt=fmt, stream=stream)

    def get_wcxf(self, C_out, scale_out):
        """Return the Wilson coefficients `C_out` as a wcxf.WC instance.

        Note that the Wilson coefficients are rotated into the Warsaw basis
        as defined in WCxf, i.e. to the basis where the down-type and charged
        lepton mass matrices are diagonal."""
        from wilson import wcxf
        # C = self.rotate_defaultbasis(C_out)
        C = C_out.copy()  # FIXME
        d = smeftutil.arrays2wcxf(C)
        basis = wcxf.Basis['SMEFT', 'Warsaw']
        d = {k: v for k, v in d.items() if k in basis.all_wcs and v != 0}
        keys_dim5 = ['llphiphi']
        keys_dim6 = list(set(smeft_warsaw.WC_keys_0f + smeft_warsaw.WC_keys_2f
                             + smeft_warsaw.WC_keys_4f) - set(keys_dim5))
        for k in d:
            if k.split('_')[0] in keys_dim5:
                d[k] = d[k] / self.scale_high
        for k in d:
            if k.split('_')[0] in keys_dim6:
                d[k] = d[k] / self.scale_high**2
        # d = {k: v for k, v in d.items() if v != 0}
        d = wcxf.WC.dict2values(d)
        wc = wcxf.WC('SMEFT', 'Warsaw', scale_out, d)
        return wc

    def dump_wcxf(self, C_out, scale_out, fmt='yaml', stream=None, **kwargs):
        """Return a string representation of the Wilson coefficients `C_out`
        in WCxf format. If `stream` is specified, export it to a file.
        `fmt` defaults to `yaml`, but can also be `json`.

        Note that the Wilson coefficients are rotated into the Warsaw basis
        as defined in WCxf, i.e. to the basis where the down-type and charged
        lepton mass matrices are diagonal."""
        wc = self.get_wcxf(C_out, scale_out)
        return wc.dump(fmt=fmt, stream=stream, **kwargs)

    def rotate_defaultbasis(self, C):
        """Rotate all parameters to the basis where the running down-type quark
        and charged lepton mass matrices are diagonal and where the running
        up-type quark mass matrix has the form V.S, with V unitary and S real
        diagonal, and where the CKM and PMNS matrices have the standard
        phase convention."""
        v = 246.22
        Mep = v/sqrt(2) * (C['Ge'] - C['ephi'] * v**2/self.scale_high**2/2)
        Mup = v/sqrt(2) * (C['Gu'] - C['uphi'] * v**2/self.scale_high**2/2)
        Mdp = v/sqrt(2) * (C['Gd'] - C['dphi'] * v**2/self.scale_high**2/2)
        Mnup = -v**2 * C['llphiphi']
        UeL, Me, UeR = ckmutil.diag.msvd(Mep)
        UuL, Mu, UuR = ckmutil.diag.msvd(Mup)
        UdL, Md, UdR = ckmutil.diag.msvd(Mdp)
        Unu, Mnu = ckmutil.diag.mtakfac(Mnup)
        UuL, UdL, UuR, UdR = ckmutil.phases.rephase_standard(UuL, UdL, UuR, UdR)
        Unu, UeL, UeR = ckmutil.phases.rephase_pmns_standard(Unu, UeL, UeR)
        return smeft_warsaw.flavor_rotation(C, Uq=UdL, Uu=UuR, Ud=UdR, Ul=UeL, Ue=UeR)


def wcxf2dsixtools(wc, stream=None):
    smeftio = SMEFTio()
    smeftio.set_initial_wcxf(wc)
    return smeftio.dump(smeftio.C_in, stream=stream)

def dsixtools2wcxf(streams, stream=None):
    smeftio = SMEFTio()
    smeftio.load_initial(streams)
    return smeftio.dump_wcxf(smeftio.C_in, smeftio.scale_in, stream=stream)
