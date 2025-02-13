"""Functions for translating to and from the SMEFTsim bases.
"""

from wilson import wcxf
from .smeft import warsaw_to_warsaw_up, warsaw_up_to_warsaw

names_permuted_indices =  {
    'cdd', 'cee', 'cll', 'cqq1', 'cqq3', 'cuu'
}
name_mapping = {
    'cHBox': 'cHbox',
    'cHD': 'cHDD',
    'cHWtilB': 'cHWBtil'
}
warsaw_up = wcxf.Basis['SMEFT', 'Warsaw up']
smeftsim_general = wcxf.Basis['SMEFT', 'SMEFTsim_general']
all_wcs_smeftsim_general = set(smeftsim_general.all_wcs)

def wc_name_warsaw_to_SMEFTsim(wc_name_in):
    """Convert a Warsaw up Wilson coefficient name to the SMEFTsim naming convention.
    """
    wc_name_out = f'c{wc_name_in}'.replace('tilde', 'til').replace('phi', 'H').split('_')
    if wc_name_out[0] in names_permuted_indices:
        if wc_name_out[1] == '2333':
            wc_name_out[1] = '3323'
        elif wc_name_out[0] == 'cee' and wc_name_out[1] == '1223':
            wc_name_out[1] = '1322'
    if wc_name_out[0] in name_mapping:
        wc_name_out[0] = name_mapping[wc_name_out[0]]
    return wc_name_out


def warsaw_up_to_SMEFTsim_general(C, parameters=None, sectors=None):
    """Translate from the Warsaw up basis to the SMEFTsim_general basis.
    """
    wc_out = {}
    for wc_name_in in warsaw_up.sectors['dB=dL=0'].keys():
        wc_name_out = wc_name_warsaw_to_SMEFTsim(wc_name_in)
        if warsaw_up.sectors['dB=dL=0'][wc_name_in].get('real'):
            if 'Re'.join(wc_name_out) in all_wcs_smeftsim_general:
                wc_out['Re'.join(wc_name_out)] = C.get(wc_name_in, 0).real * 1e6
            else:
                wc_out[''.join(wc_name_out)] = C.get(wc_name_in, 0).real * 1e6
        else:
            wc_out['Re'.join(wc_name_out)] = C.get(wc_name_in, 0).real * 1e6
            wc_out['Im'.join(wc_name_out)] = C.get(wc_name_in, 0).imag * 1e6
    return wc_out

def SMEFTsim_general_to_warsaw_up(C, parameters=None, sectors=None):
    """Translate from the SMEFTsim_general basis to the Warsaw up basis.
    """
    wc_out = {}
    for wc_name_out in warsaw_up.sectors['dB=dL=0'].keys():
        wc_name_in = wc_name_warsaw_to_SMEFTsim(wc_name_out)
        if warsaw_up.sectors['dB=dL=0'][wc_name_out].get('real'):
            if 'Re'.join(wc_name_in) in all_wcs_smeftsim_general:
                wc_out[wc_name_out] = C.get('Re'.join(wc_name_in), 0) / 1e6
            else:
                wc_out[wc_name_out] = C.get(''.join(wc_name_in), 0) / 1e6
        else:
            wc_out[wc_name_out] = (
                C.get('Re'.join(wc_name_in), 0) / 1e6
                + 1j * C.get('Im'.join(wc_name_in), 0) / 1e6
            )
    return wc_out
