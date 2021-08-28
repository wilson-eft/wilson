"""Functions to convert WCxf files to EOS parameter files."""

from collections import OrderedDict
import yaml
import glob
import os
import re


def get_sm_wcs(eos_parameter_dir):
    """Read SM contributions to EOS Wilson coefficients from the EOS
    installation parameter directory and return them as a dictionary."""
    all_wcs = {}
    yamlfiles = glob.glob(os.path.join(eos_parameter_dir, '*.yaml'))
    for yamlfile in yamlfiles:
        with open(yamlfile) as f:
            wcs = yaml.safe_load(f)
        meta = wcs.get('@metadata@', {})
        if 'wcxf-relevant' in meta and meta['wcxf-relevant']:
            del wcs['@metadata@']
            all_wcs.update(wcs)
    return all_wcs


def get_eos_np(sm_key, wc):
    """Get the NP contribution to an EOS Wilson coefficient identified
    by the name `sm_key` from a wcxf.WC instance `wc`"""
    if sm_key in wc.dict:
        return wc.dict[sm_key].real
    elif r'Re{' in sm_key: # if the EOS name contains "Re"
        k2 = re.sub(r'Re\{([^\}]+)\}', r'\1', sm_key) # name of the coeff in the basis file
        if k2 in wc.dict:
            return wc.dict[k2].real
        elif k2.replace('lnu', 'enue') in wc.dict or k2.replace('lnu', 'munumu') in wc.dict:
            # for charged-current WCs, check if the WCxf file is LFU, else raise an error
            if wc.dict.get(k2.replace('lnu', 'enue'), 0) != wc.dict.get(k2.replace('lnu', 'munumu'), 0):
                raise ValueError("Found lepton flavour non-universal charged-current coefficients")
            return wc.dict.get(k2.replace('lnu', 'enue'), 0).real
    elif r'Im{' in sm_key: # if the EOS name contains "Im"
        k2 = re.sub(r'Im\{([^\}]+)\}', r'\1', sm_key) # name of the coeff in the basis file
        if k2 in wc.dict:
            return wc.dict[k2].imag
        elif k2.replace('lnu', 'enue') in wc.dict or k2.replace('lnu', 'munumu') in wc.dict:
            # for charged-current WCs, check if the WCxf file is LFU, else raise an error
            if wc.dict.get(k2.replace('lnu', 'enue'), 0) != wc.dict.get(k2.replace('lnu', 'munumu'), 0):
                raise ValueError("Found lepton flavour non-universal charged-current coefficients")
            return wc.dict.get(k2.replace('lnu', 'enue'), 0).imag
    return 0


def wcxf2eos(wc, sm_wc_dict):
    """From a wcxf.WC instance wc and a dictionary of EOS Wilson coefficient
    SM contributions, return a dictionary of EOS Wilson coefficient parameter
    values including SM and NP contributions."""
    eosd = OrderedDict()
    for k, v in sm_wc_dict.items():
        sm = v['central']
        np = get_eos_np(k, wc)
        eosd[k] = OrderedDict()
        for q in ['central', 'min', 'max']:
            eosd[k][q] = sm + np
    return eosd
