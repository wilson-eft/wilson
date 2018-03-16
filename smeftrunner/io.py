import numpy as np
from smeftrunner import beta, definitions
from collections import OrderedDict, defaultdict
import pylha
import json
import yaml

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
        return yaml.load(stream)

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
    for k in definitions.WC_keys_2f:
        try:
            C[k] = lha2matrix(lha['BLOCK']['WC' + k.upper()]['values'], (3,3)).real
        except KeyError:
            C[k] = np.zeros((3,3))
        try: # try to add imaginary part
            C[k] = C[k] + 1j*lha2matrix(lha['BLOCK']['IMWC' + k.upper()]['values'], (3,3))
        except KeyError:
            pass
    for k in definitions.WC_keys_4f:
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
    for name in definitions.WC_keys_2f:
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
    for name in definitions.WC_keys_4f:
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
