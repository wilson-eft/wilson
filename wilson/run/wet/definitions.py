"""Dictionaries assigning Wilson coefficients to classes and sectors,
needed to construct appropriate vectors."""

import json
import pkgutil


classes = {
'I': ['sbsb', 'dbdb', 'sdsd'],

'Ie': ['tauetaue', 'muemue', 'taumutaumu'],
'Iu': ['cucu'],

'II': ['ubenu', 'ubmunu', 'ubtaunu', 'cbenu', 'cbmunu', 'cbtaunu', 'usenu', 'usmunu', 'ustaunu', 'csenu', 'csmunu', 'cstaunu', 'udenu', 'udmunu', 'udtaunu', 'cdenu', 'cdmunu', 'cdtaunu'],
'III': ['sbuc', 'sbcu', 'dbuc', 'dbcu', 'sduc', 'sdcu'],
'IV': ['sbsd', 'dbsb', 'dbds'],

'Vb': ['sbemu', 'sbmue', 'sbetau', 'sbtaue', 'sbmutau', 'sbtaumu', 'dbemu', 'dbmue', 'dbetau', 'dbtaue', 'dbmutau', 'dbtaumu', 'sdemu', 'sdmue', 'sdetau', 'sdtaue', 'sdmutau', 'sdtaumu'],

'Vb_c': ['cuemu', 'cumue', 'cuetau', 'cutaue', 'cumutau', 'cutaumu'],

'IVe': ['muemutau', 'etauemu', 'tauetaumu'],

'inv': ['sbnunu', 'dbnunu', 'sdnunu', 'cununu', 'nunumue', 'nunumutau', 'nunutaue', 'nunununu'],

 'cu': ['cu'],
 'db': ['db'],
 'sb': ['sb'],
 'sd': ['sd'],
 'mue': ['mue'],
 'mutau': ['mutau'],
 'taue': ['taue'],
 'dF0': ['dF=0'],
 'ffnunu': ['ffnunu'],
}


sectors = {}
for c, ss in classes.items():
    for s in ss:
        sectors[s] = c

coeffs = json.loads(pkgutil.get_data('wilson', 'data/run_wet_definitions_coeffs.json').decode('utf-8'))
