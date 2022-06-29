"""Provides basis translators for SMEFT and and WET that can be used with the
`wcxf` Python package."""


from . import smeft, smeft_higgs
from . import wet
from wilson import wcxf


@wcxf.translator('SMEFT', 'Higgs-Warsaw up', 'Warsaw up')
def higgs_up_to_warsaw_up(C, scale, parameters, sectors=None):
    return smeft_higgs.higgslike_to_warsaw_up(C=C, parameters=parameters, sectors=sectors)


@wcxf.translator('SMEFT', 'Higgs-Warsaw up', 'Warsaw')
def higgs_up_to_warsaw(C, scale, parameters, sectors=None):
    C = smeft_higgs.higgslike_to_warsaw_up(C=C, parameters=parameters, sectors=sectors)
    return smeft.warsaw_up_to_warsaw(C=C, parameters=parameters, sectors=sectors)


@wcxf.translator('SMEFT', 'Warsaw up', 'Higgs-Warsaw up')
def warsaw_up_to_higgs_up(C, scale, parameters, sectors=None):
    return smeft_higgs.warsaw_up_to_higgslike(C=C, parameters=parameters, sectors=sectors)

@wcxf.translator('SMEFT', 'Warsaw', 'Higgs-Warsaw up')
def warsaw_up_to_higgs_up(C, scale, parameters, sectors=None):
    C = smeft.warsaw_to_warsaw_up(C=C, parameters=parameters, sectors=sectors)
    return smeft_higgs.warsaw_up_to_higgslike(C=C, parameters=parameters, sectors=sectors)


@wcxf.translator('SMEFT', 'Warsaw', 'Warsaw mass')
def warsaw_to_warsawmass(C, scale, parameters, sectors=None):
    return smeft.warsaw_to_warsawmass(C=C, parameters=parameters, sectors=sectors)


@wcxf.translator('SMEFT', 'Warsaw', 'Warsaw up')
def warsaw_to_warsaw_up(C, scale, parameters, sectors=None):
    return smeft.warsaw_to_warsaw_up(C=C, parameters=parameters, sectors=sectors)


@wcxf.translator('SMEFT', 'Warsaw up', 'Warsaw')
def warsaw_up_to_warsaw(C, scale, parameters, sectors=None):
    return smeft.warsaw_up_to_warsaw(C=C, parameters=parameters, sectors=sectors)


@wcxf.translator('WET', 'flavio', 'JMS')
def flavio_to_JMS(C, scale, parameters, sectors=None):
    return wet.flavio_to_JMS(C_incomplete=C, scale=scale, parameters=parameters, sectors=sectors)


@wcxf.translator('WET-4', 'flavio', 'JMS')
def flavio_to_JMS_wet4(C, scale, parameters, sectors=None):
    return wet.flavio_to_JMS(C_incomplete=C, scale=scale, parameters=parameters, sectors=sectors)


@wcxf.translator('WET-3', 'flavio', 'JMS')
def flavio_to_JMS_wet3(C, scale, parameters, sectors=None):
    return wet.flavio_to_JMS(C_incomplete=C, scale=scale, parameters=parameters, sectors=sectors)


@wcxf.translator('WET', 'JMS', 'flavio')
def JMS_to_flavio(C, scale, parameters, sectors=None):
    return wet.JMS_to_flavio(Cflat=C, scale=scale, parameters=parameters, sectors=sectors)


@wcxf.translator('WET-4', 'JMS', 'flavio')
def JMS_to_flavio_wet4(C, scale, parameters, sectors=None):
    return wet.JMS_to_flavio(Cflat=C, scale=scale, parameters=parameters, sectors=sectors)


@wcxf.translator('WET-3', 'JMS', 'flavio')
def JMS_to_flavio_wet3(C, scale, parameters, sectors=None):
    return wet.JMS_to_flavio(Cflat=C, scale=scale, parameters=parameters, sectors=sectors)


@wcxf.translator('WET', 'Bern', 'flavio')
def Bern_to_flavio(C, scale, parameters, sectors=None):
    return wet.Bern_to_flavio(C_incomplete=C, scale=scale, parameters=parameters, sectors=sectors)


@wcxf.translator('WET', 'flavio', 'Bern')
def flavio_to_Bern(C, scale, parameters, sectors=None):
    return wet.flavio_to_Bern(C_incomplete=C, scale=scale, parameters=parameters, sectors=sectors)


@wcxf.translator('WET-4', 'Bern', 'flavio')
def Bern_to_flavio_wet4(C, scale, parameters, sectors=None):
    return wet.Bern_to_flavio(C_incomplete=C, scale=scale, parameters=parameters, sectors=sectors)


@wcxf.translator('WET-4', 'flavio', 'Bern')
def flavio_to_Bern_wet4(C, scale, parameters, sectors=None):
    return wet.flavio_to_Bern(C_incomplete=C, scale=scale, parameters=parameters, sectors=sectors)


@wcxf.translator('WET-3', 'Bern', 'flavio')
def Bern_to_flavio_wet3(C, scale, parameters, sectors=None):
    return wet.Bern_to_flavio(C_incomplete=C, scale=scale, parameters=parameters, sectors=sectors)


@wcxf.translator('WET-3', 'flavio', 'Bern')
def flavio_to_Bern_wet3(C, scale, parameters, sectors=None):
    return wet.flavio_to_Bern(C_incomplete=C, scale=scale, parameters=parameters, sectors=sectors)


@wcxf.translator('WET', 'JMS', 'EOS')
def JMS_to_EOS(C, scale, parameters, sectors=None):
    return wet.JMS_to_EOS(Cflat=C, scale=scale, parameters=parameters, sectors=sectors)


@wcxf.translator('WET', 'JMS', 'Bern')
def JMS_to_Bern(C, scale, parameters, sectors=None):
    return wet.JMS_to_Bern(Cflat=C, scale=scale, parameters=parameters, sectors=sectors)


@wcxf.translator('WET-4', 'JMS', 'Bern')
def JMS_to_Bern_wet4(C, scale, parameters, sectors=None):
    return wet.JMS_to_Bern(Cflat=C, scale=scale, parameters=parameters, sectors=sectors)


@wcxf.translator('WET-3', 'JMS', 'Bern')
def JMS_to_Bern_wet3(C, scale, parameters, sectors=None):
    return wet.JMS_to_Bern(Cflat=C, scale=scale, parameters=parameters, sectors=sectors)


@wcxf.translator('WET', 'Bern', 'JMS')
def Bern_to_JMS(C, scale, parameters, sectors=None):
    return wet.Bern_to_JMS(C_incomplete=C, scale=scale, parameters=parameters, sectors=sectors)

@wcxf.translator('WET-4', 'Bern', 'JMS')
def Bern_to_JMS_wet4(C, scale, parameters, sectors=None):
    return wet.Bern_to_JMS(C_incomplete=C, scale=scale, parameters=parameters, sectors=sectors)

@wcxf.translator('WET-3', 'Bern', 'JMS')
def Bern_to_JMS_wet3(C, scale, parameters, sectors=None):
    return wet.Bern_to_JMS(C_incomplete=C, scale=scale, parameters=parameters, sectors=sectors)


@wcxf.translator('WET', 'JMS', 'formflavor')
def JMS_to_FormFlavor(C, scale, parameters, sectors=None):
    return wet.JMS_to_FormFlavor(Cflat=C, scale=scale, parameters=parameters, sectors=sectors)


@wcxf.translator('WET', 'FlavorKit', 'JMS')
def FlavorKit_to_JMS(C, scale, parameters, sectors=None):
    return wet.FlavorKit_to_JMS(C=C, scale=scale, parameters=parameters, sectors=sectors)


@wcxf.translator('WET', 'JMS', 'FlavorKit')
def JMS_to_FlavorKit(C, scale, parameters, sectors=None):
    return wet.JMS_to_FlavorKit(C=C, scale=scale, parameters=parameters, sectors=sectors)


@wcxf.translator('WET', 'FlavorKit', 'flavio')
def FlavorKit_to_flavio(C, scale, parameters, sectors=None):
    C_JMS = wet.FlavorKit_to_JMS(C=C, scale=scale, parameters=parameters, sectors=sectors)
    return wet.JMS_to_flavio(Cflat=C_JMS, scale=scale, parameters=parameters, sectors=sectors)
