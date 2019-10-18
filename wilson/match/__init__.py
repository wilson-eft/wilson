"""Provides matchers from SMEFT to WET, WET to WET-4, and WET-4 to WET-3 that
can be used with the `wcxf` Python package."""


import wcxf
import wilson
from . import smeft


@wcxf.matcher('SMEFT', 'Warsaw up', 'WET', 'JMS')
def warsaw_up_to_jms(C, scale, parameters):
    return smeft.match_all(C, scale, parameters)


@wcxf.matcher('SMEFT', 'Warsaw', 'WET', 'JMS')
def warsaw_to_jms(C, scale, parameters):
    C_warsawup = wilson.translate.smeft.warsaw_to_warsaw_up(C, parameters)
    return smeft.match_all(C_warsawup, scale, parameters)


@wcxf.matcher('SMEFT', 'Warsaw', 'WET', 'flavio')
def warsaw_to_flavio(C, scale, parameters):
    C_warsawup = wilson.translate.smeft.warsaw_to_warsaw_up(C, parameters)
    C_JMS = smeft.match_all(C_warsawup, scale, parameters)
    return wilson.translate.JMS_to_flavio(C_JMS, scale, parameters)


@wcxf.matcher('SMEFT', 'Warsaw up', 'WET', 'flavio')
def warsaw_up_to_flavio(C, scale, parameters):
    C_JMS = smeft.match_all(C, scale, parameters)
    return wilson.translate.JMS_to_flavio(C_JMS, scale, parameters)


@wcxf.matcher('SMEFT', 'Warsaw', 'WET', 'EOS')
def warsaw_to_eos(C, scale, parameters):
    C_warsawup = wilson.translate.smeft.warsaw_to_warsaw_up(C, parameters)
    C_JMS = smeft.match_all(C_warsawup, scale, parameters)
    return wilson.translate.JMS_to_EOS(C_JMS, scale, parameters)


@wcxf.matcher('SMEFT', 'Warsaw', 'WET', 'Bern')
def warsaw_to_bern(C, scale, parameters):
    C_warsawup = wilson.translate.smeft.warsaw_to_warsaw_up(C, parameters)
    C_JMS = smeft.match_all(C_warsawup, scale, parameters)
    return wilson.translate.JMS_to_Bern(C_JMS, scale, parameters)


@wcxf.matcher('WET', 'flavio', 'WET-4', 'flavio')
def wet_wet4_flavio(C, scale, parameters):
    keys = set(wcxf.Basis['WET-4', 'flavio'].all_wcs)
    return {k: v for k, v in C.items() if k in keys}


@wcxf.matcher('WET-4', 'flavio', 'WET-3', 'flavio')
def wet4_wet3_flavio(C, scale, parameters):
    keys = set(wcxf.Basis['WET-3', 'flavio'].all_wcs)
    return {k: v for k, v in C.items() if k in keys}


@wcxf.matcher('WET', 'Bern', 'WET-4', 'Bern')
def wet_wet4_bern(C, scale, parameters):
    keys = set(wcxf.Basis['WET-4', 'Bern'].all_wcs)
    return {k: v for k, v in C.items() if k in keys}


@wcxf.matcher('WET-4', 'Bern', 'WET-3', 'Bern')
def wet4_wet3_bern(C, scale, parameters):
    keys = set(wcxf.Basis['WET-3', 'Bern'].all_wcs)
    return {k: v for k, v in C.items() if k in keys}


@wcxf.matcher('WET', 'JMS', 'WET-4', 'JMS')
def wet_wet4_jms(C, scale, parameters):
    keys = set(wcxf.Basis['WET-4', 'JMS'].all_wcs)
    return {k: v for k, v in C.items() if k in keys}


@wcxf.matcher('WET-4', 'JMS', 'WET-3', 'JMS')
def wet4_wet3_jms(C, scale, parameters):
    keys = set(wcxf.Basis['WET-3', 'JMS'].all_wcs)
    return {k: v for k, v in C.items() if k in keys}


@wcxf.matcher('WET-3', 'JMS', 'WET-2', 'JMS')
def wet3_wet2_jms(C, scale, parameters):
    keys = set(wcxf.Basis['WET-2', 'JMS'].all_wcs)
    return {k: v for k, v in C.items() if k in keys}
