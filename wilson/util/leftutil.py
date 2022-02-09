import numpy as np

SM_keys = ['e', 'g', 'Md', 'Mu', 'Me', 'Mnu'] 

WC_keys_0f= ['G', 'Gtilde']

WC_keys_2f = ['egamma', 'ugamma', 'dgamma', 'nugamma', 'uG', 'dG', 'delta']

WC_keys_4f= ['S1udRR', 'S1udduRR', 'S8udRR', 'S8udduRR', 'SedRL', 'SedRR', 'SeuRL', 'SeuRR', \
'SnueduRL', 'SnueduRR', 'TedRR', 'TeuRR', 'TnueduRR', 'V1udduLR', 'V8udduLR', 'VnueduLL', 'VnueduLR', \
'SuddLL', 'SduuLL', 'SduuLR', 'SduuRL', 'SdudRL', 'SduuRR', 'VuuRR', 'VddRR', 'VuuLL', 'VddLL', 'S1ddRR',\
 'S1uuRR', 'S8uuRR', 'SeeRR', 'S8ddRR', 'VnueLL', 'VnuuLL', 'VnudLL', 'VeuLL', 'VedLL', 'V1udLL', 'V8udLL',\
 'VeuRR', 'VedRR', 'V1udRR', 'V8udRR', 'VnueLR', 'VeeLR', 'VnuuLR', 'VnudLR', 'VeuLR', 'VedLR', 'VueLR',\
 'VdeLR', 'V1uuLR', 'V8uuLR', 'V1udLR', 'V8udLR', 'V1duLR', 'V8duLR', 'V1ddLR', 'V8ddLR', 'VeeLL', 'VeeRR',\
 'VnunuLL', 'SuudLR', 'SuudRL', 'SdduRL', 'SnudLL', 'SnudLR', 'SnueLL', 'SnueLR', 'SnuuLL', 'SnuuLR', \
'SnunuLL', 'TnueLL', 'TnuuLL', 'TnudLL', 'SnueduLL', 'TnueduLL', 'SnueduLR', 'VnueduRL', 'VnueduRR',\
 'SdddLL', 'SuddLR', 'SdduLR', 'SdddLR', 'SdddRL', 'SuddRR', 'SdddRR']

C_keys = SM_keys + WC_keys_0f + WC_keys_2f + WC_keys_4f
WC_keys =  WC_keys_0f + WC_keys_2f + WC_keys_4f

#FIXME remove third flavour
C_keys_shape = {
    'delta': (3,3),
    'G': 1,
    'Gtilde': 1,
    'egamma': (3, 3),
    'uG': (3, 3),
    'dG': (3, 3),
    'ugamma': (3, 3),
    'dgamma': (3, 3),
    'S1udRR': (3, 3, 3, 3),
    'S1udduRR': (3, 3, 3, 3),
    'S8udRR': (3, 3, 3, 3),
    'S8udduRR': (3, 3, 3, 3),
    'SedRL': (3, 3, 3, 3),
    'SedRR': (3, 3, 3, 3),
    'SeuRL': (3, 3, 3, 3),
    'SeuRR': (3, 3, 3, 3),
    'SnueduRL': (3, 3, 3, 3),
    'SnueduRR': (3, 3, 3, 3),
    'TedRR': (3, 3, 3, 3),
    'TeuRR': (3, 3, 3, 3),
    'TnueduRR': (3, 3, 3, 3),
    'V1udduLR': (3, 3, 3, 3),
    'V8udduLR': (3, 3, 3, 3),
    'VnueduLL': (3, 3, 3, 3),
    'VnueduLR': (3, 3, 3, 3),
    'SuddLL': (3, 3, 3, 3),
    'SduuLL': (3, 3, 3, 3),
    'SduuLR': (3, 3, 3, 3),
    'SduuRL': (3, 3, 3, 3),
    'SdudRL': (3, 3, 3, 3),
    'SduuRR': (3, 3, 3, 3),
    'VuuRR': (3, 3, 3, 3),
    'VddRR': (3, 3, 3, 3),
    'VuuLL': (3, 3, 3, 3),
    'VddLL': (3, 3, 3, 3),
    'S1ddRR': (3, 3, 3, 3),
    'S1uuRR': (3, 3, 3, 3),
    'S8uuRR': (3, 3, 3, 3),
    'SeeRR': (3, 3, 3, 3),
    'S8ddRR': (3, 3, 3, 3),
    'VnueLL': (3, 3, 3, 3),
    'VnuuLL': (3, 3, 3, 3),
    'VnudLL': (3, 3, 3, 3),
    'VeuLL': (3, 3, 3, 3),
    'VedLL': (3, 3, 3, 3),
    'V1udLL': (3, 3, 3, 3),
    'V8udLL': (3, 3, 3, 3),
    'VeuRR': (3, 3, 3, 3),
    'VedRR': (3, 3, 3, 3),
    'V1udRR': (3, 3, 3, 3),
    'V8udRR': (3, 3, 3, 3),
    'VnueLR': (3, 3, 3, 3),
    'VeeLR': (3, 3, 3, 3),
    'VnuuLR': (3, 3, 3, 3),
    'VnudLR': (3, 3, 3, 3),
    'VeuLR': (3, 3, 3, 3),
    'VedLR': (3, 3, 3, 3),
    'VueLR': (3, 3, 3, 3),
    'VdeLR': (3, 3, 3, 3),
    'V1uuLR': (3, 3, 3, 3),
    'V8uuLR': (3, 3, 3, 3),
    'V1udLR': (3, 3, 3, 3),
    'V8udLR': (3, 3, 3, 3),
    'V1duLR': (3, 3, 3, 3),
    'V8duLR': (3, 3, 3, 3),
    'V1ddLR': (3, 3, 3, 3),
    'V8ddLR': (3, 3, 3, 3),
    'VeeLL': (3, 3, 3, 3),
    'VeeRR': (3, 3, 3, 3),
    'VnunuLL': (3, 3, 3, 3),
    'SuudLR': (3, 3, 3, 3),
    'SuudRL': (3, 3, 3, 3),
    'SdduRL': (3, 3, 3, 3),
}

C_keys_shape_missing = {
    'e': 1,
    'g': 1,
    'Md': (3,3),
    'Me': (3,3),
    'Mu': (3,3),
    'Mnu': (3,3),
    'nugamma': (3,3),
    'SnudLL': (3,3,3,3),
    'SnudLR': (3,3,3,3),
    'SnueLL': (3,3,3,3),
    'SnueLR': (3,3,3,3),
    'SnuuLL': (3,3,3,3),
    'SnuuLR': (3,3,3,3),
    'SnunuLL': (3,3,3,3),
    'TnueLL': (3,3,3,3),
    'TnuuLL': (3,3,3,3),
    'TnudLL': (3,3,3,3),
    'SnueduLL': (3,3,3,3),
    'TnueduLL': (3,3,3,3),
    'SnueduLR': (3,3,3,3),
    'VnueduRL': (3,3,3,3),
    'VnueduRR': (3,3,3,3),
    'SdddLL': (3,3,3,3),
    'SuddLR': (3,3,3,3),
    'SdduLR': (3,3,3,3),
    'SdddLR': (3,3,3,3),
    'SdddRL': (3,3,3,3),
    'SuddRR': (3,3,3,3),
    'SdddRR': (3,3,3,3),
}

C_keys_shape.update(C_keys_shape_missing)


def wcxf2arrays(d):
    """Convert a dictionary with a Wilson coefficient
    name followed by underscore and numeric indices as keys and numbers as
    values to a dictionary with Wilson coefficient names as keys and
    numbers or numpy arrays as values. This is needed for the parsing
    of input in WCxf format."""
    C = {}
    for k, v in d.items():
        name = k.split('_')[0]
        s = C_keys_shape[name]
        if s == 1:
            C[k] = v
        else:
            ind = k.split('_')[-1]
            if name not in C:
                C[name] = np.zeros(s, dtype=complex)
            C[name][tuple([int(i) - 1 for i in ind])] = v
    return C

def add_missing(C):
    """Add arrays with zeros for missing Wilson coefficient keys"""
    C_out = C.copy()
    for k in (set(C_keys) - set(C.keys())): # FIXME
        s = C_keys_shape[k]
        if s == 1:
            C_out[k] = 0
        else:
            C_out[k] = np.zeros(C_keys_shape[k])
    return C_out

def wcxf2arrays_symmetrized(d):
    """Convert a LEFT dictionary with a Wilson coefficient
    name followed by underscore and numeric indices as keys and numbers as
    values to a dictionary with Wilson coefficient names as keys and
    numbers or numpy arrays as values.


    In contrast to `wcxf2arrays`, here the numpy arrays fulfill the same
    symmetry relations as the operators (i.e. they contain redundant entries)
    and they do not contain undefined indices.

    Zero arrays are added for missing coefficients."""
    C = wcxf2arrays(d)
#    C = symmetrize_nonred(C)
    C = add_missing(C)
    return C


def arrays2wcxf(C):
    """Convert a dictionary with Wilson coefficient names as keys and
    numbers or numpy arrays as values to a dictionary with a Wilson coefficient
    name followed by underscore and numeric indices as keys and numbers as
    values. This is needed for the output in WCxf format."""
    d = {}
    for k, v in C.items():
        if np.shape(v) == () or np.shape(v) == (1,):
            d[k] = v
        else:
            ind = np.indices(v.shape).reshape(v.ndim, v.size).T
            for i in ind:
                name = k + '_' + ''.join([str(int(j) + 1) for j in i])
                d[name] = v[tuple(i)]
    return d









