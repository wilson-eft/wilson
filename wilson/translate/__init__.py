from . import smeft
from . import wet
import wcxf


@wcxf.translator('SMEFT', 'Warsaw', 'Warsaw mass')
def warsaw_to_warsawmass(C, scale, parameters):
    return smeft.warsaw_to_warsawmass(C)


@wcxf.translator('SMEFT', 'Warsaw', 'Warsaw up')
def warsaw_to_warsaw_up(C, scale, parameters):
    return smeft.warsaw_to_warsaw_up(C)


@wcxf.translator('SMEFT', 'Warsaw up', 'Warsaw')
def warsaw_up_to_warsaw(C, scale, parameters):
    return smeft.warsaw_up_to_warsaw(C)


@wcxf.translator('WET', 'flavio', 'JMS')
def flavio_to_JMS(C, scale, parameters):
    return wet.flavio_to_JMS(C, scale, parameters)


@wcxf.translator('WET-4', 'flavio', 'JMS')
def flavio_to_JMS_wet4(C, scale, parameters):
    return wet.flavio_to_JMS(C, scale, parameters)


@wcxf.translator('WET-3', 'flavio', 'JMS')
def flavio_to_JMS_wet3(C, scale, parameters):
    return wet.flavio_to_JMS(C, scale, parameters)


@wcxf.translator('WET', 'JMS', 'flavio')
def JMS_to_flavio(C, scale, parameters):
    return wet.JMS_to_flavio(C, scale, parameters)


@wcxf.translator('WET-4', 'JMS', 'flavio')
def JMS_to_flavio_wet4(C, scale, parameters):
    return wet.JMS_to_flavio(C, scale, parameters)


@wcxf.translator('WET-3', 'JMS', 'flavio')
def JMS_to_flavio_wet3(C, scale, parameters):
    return wet.JMS_to_flavio(C, scale, parameters)


@wcxf.translator('WET', 'Bern', 'flavio')
def Bern_to_flavio(C, scale, parameters):
    return wet.Bern_to_flavio(C, scale, parameters)


@wcxf.translator('WET', 'flavio', 'Bern')
def flavio_to_Bern(C, scale, parameters):
    return wet.flavio_to_Bern(C, scale, parameters)


@wcxf.translator('WET-4', 'Bern', 'flavio')
def Bern_to_flavio_wet4(C, scale, parameters):
    return wet.Bern_to_flavio(C, scale, parameters)


@wcxf.translator('WET-4', 'flavio', 'Bern')
def flavio_to_Bern_wet4(C, scale, parameters):
    return wet.flavio_to_Bern(C, scale, parameters)


@wcxf.translator('WET-3', 'Bern', 'flavio')
def Bern_to_flavio_wet3(C, scale, parameters):
    return wet.Bern_to_flavio(C, scale, parameters)


@wcxf.translator('WET-3', 'flavio', 'Bern')
def flavio_to_Bern_wet3(C, scale, parameters):
    return wet.flavio_to_Bern(C, scale, parameters)


@wcxf.translator('WET', 'JMS', 'EOS')
def JMS_to_EOS(C, scale, parameters):
    return wet.JMS_to_EOS(C, scale, parameters)


@wcxf.translator('WET', 'JMS', 'Bern')
def JMS_to_Bern(C, scale, parameters):
    return wet.JMS_to_Bern(C, scale, parameters)


@wcxf.translator('WET-4', 'JMS', 'Bern')
def JMS_to_Bern_wet4(C, scale, parameters):
    return wet.JMS_to_Bern(C, scale, parameters)


@wcxf.translator('WET-3', 'JMS', 'Bern')
def JMS_to_Bern_wet3(C, scale, parameters):
    return wet.JMS_to_Bern(C, scale, parameters)


@wcxf.translator('WET', 'Bern', 'JMS')
def Bern_to_JMS(C, scale, parameters):
    return wet.Bern_to_JMS(C, scale, parameters)

@wcxf.translator('WET-4', 'Bern', 'JMS')
def Bern_to_JMS_wet4(C, scale, parameters):
    return wet.Bern_to_JMS(C, scale, parameters)

@wcxf.translator('WET-3', 'Bern', 'JMS')
def Bern_to_JMS_wet3(C, scale, parameters):
    return wet.Bern_to_JMS(C, scale, parameters)


@wcxf.translator('WET', 'JMS', 'formflavor')
def JMS_to_FormFlavor(C, scale, parameters):
    return wet.JMS_to_FormFlavor(C, scale, parameters)


@wcxf.translator('WET', 'FlavorKit', 'JMS')
def FlavorKit_to_JMS(C, scale, parameters):
    return wet.FlavorKit_to_JMS(C, scale, parameters)


@wcxf.translator('WET', 'JMS', 'FlavorKit')
def JMS_to_FlavorKit(C, scale, parameters):
    return wet.JMS_to_FlavorKit(C, scale, parameters)


@wcxf.translator('WET', 'FlavorKit', 'flavio')
def FlavorKit_to_flavio(C, scale, parameters):
    C_JMS = wet.FlavorKit_to_JMS(C, scale, parameters)
    return wet.JMS_to_flavio(C_JMS, scale, parameters)
