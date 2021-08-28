preamble_A = '''
######################################################################
## PARAM_CARD FOR SMEFTSIM SET A v2.0 - FLAVOR_GENERAL ALPHA_INPUTS  #####
######################################################################

###################################
## INFORMATION FOR SMINPUTS
###################################
Block SMINPUTS
    1 7.815553e-03 # aEW
    2 1.166379e-05 # Gf
    3 1.181000e-01 # aS

###################################
## INFORMATION FOR MASS
###################################
Block MASS
    1 4.700000e-03 # MD
    2 2.200000e-03 # MU
    3 9.600000e-02 # MS
    4 1.280000e+00 # MC
    5 4.180000e+00 # MB
    6 1.732000e+02 # MT
   11 5.110000e-04 # Me
   13 1.056600e-01 # MMU
   15 1.777000e+00 # MTA
   23 9.118760e+01 # MZ
   25 1.250900e+02 # MH
##  Not dependent paramater.
## Those values should be edited following analytical the
## analytical expression. Some generator could simply ignore
## those values and use the analytical expression
  22 0.000000 # a : 0.0
  24 73.187688 # W+ : dMW + MW0
  21 0.000000 # g : 0.0
  9000001 0.000000 # ghA : 0.0
  9000003 73.187688 # ghWp : dMW + MW0
  9000004 73.187688 # ghWm : dMW + MW0
  82 0.000000 # ghG : 0.0
  12 0.000000 # ve : 0.0
  14 0.000000 # vm : 0.0
  16 0.000000 # vt : 0.0
  251 73.187688 # G+ : dMW + MW0
  9000002 91.187600 # ghZ : MZ
  250 91.187600 # G0 : MZ

###################################
## INFORMATION FOR DECAY
###################################
DECAY   6 1.508336e+00
DECAY  23 2.495200e+00
DECAY  24 2.085000e+00
DECAY  25 4.070000e-03
##  Not dependent paramater.
## Those values should be edited following analytical the
## analytical expression. Some generator could simply ignore
## those values and use the analytical expression
DECAY  22 0.000000 # a : 0.0
DECAY  21 0.000000 # g : 0.0
DECAY  9000001 0.000000 # ghA : 0.0
DECAY  82 0.000000 # ghG : 0.0
DECAY  12 0.000000 # ve : 0.0
DECAY  14 0.000000 # vm : 0.0
DECAY  16 0.000000 # vt : 0.0
DECAY  11 0.000000 # e- : 0.0
DECAY  13 0.000000 # mu- : 0.0
DECAY  15 0.000000 # ta- : 0.0
DECAY  2 0.000000 # u : 0.0
DECAY  4 0.000000 # c : 0.0
DECAY  1 0.000000 # d : 0.0
DECAY  3 0.000000 # s : 0.0
DECAY  5 0.000000 # b : 0.0
DECAY  9000002 2.495200 # ghZ : WZ
DECAY  9000003 2.085000 # ghWp : WW
DECAY  9000004 2.085000 # ghWm : WW
DECAY  250 2.495200 # G0 : WZ
DECAY  251 2.085000 # G+ : WW

###################################
## INFORMATION FOR CKMBLOCK
###################################
Block CKMBLOCK
    1 2.277360e-01 # cabi
    2 2.250600e-01 # CKMlambda
    3 8.110000e-01 # CKMA
    4 1.240000e-01 # CKMrho
    5 3.560000e-01 # CKMeta

'''

postamble_A = '''

###################################
## INFORMATION FOR YUKAWA
###################################
Block YUKAWA
    1 4.700000e-03 # ymdo
    2 2.200000e-03 # ymup
    3 9.600000e-02 # yms
    4 1.280000e+00 # ymc
    5 4.180000e+00 # ymb
    6 1.732000e+02 # ymt
   11 5.110000e-04 # yme
   13 1.056600e-01 # ymm
   15 1.777000e+00 # ymtau
#===========================================================
# QUANTUM NUMBERS OF NEW STATE(S) (NON SM PDG CODE)
#===========================================================

Block QNUMBERS 9000001  # ghA
        1 0  # 3 times electric charge
        2 -1  # number of spin states (2S+1)
        3 1  # colour rep (1: singlet, 3: triplet, 8: octet)
        4 1  # Particle/Antiparticle distinction (0=own anti)
Block QNUMBERS 9000002  # ghZ
        1 0  # 3 times electric charge
        2 -1  # number of spin states (2S+1)
        3 1  # colour rep (1: singlet, 3: triplet, 8: octet)
        4 1  # Particle/Antiparticle distinction (0=own anti)
Block QNUMBERS 9000003  # ghWp
        1 3  # 3 times electric charge
        2 -1  # number of spin states (2S+1)
        3 1  # colour rep (1: singlet, 3: triplet, 8: octet)
        4 1  # Particle/Antiparticle distinction (0=own anti)
Block QNUMBERS 9000004  # ghWm
        1 -3  # 3 times electric charge
        2 -1  # number of spin states (2S+1)
        3 1  # colour rep (1: singlet, 3: triplet, 8: octet)
        4 1  # Particle/Antiparticle distinction (0=own anti)
Block QNUMBERS 82  # ghG
        1 0  # 3 times electric charge
        2 -1  # number of spin states (2S+1)
        3 8  # colour rep (1: singlet, 3: triplet, 8: octet)
        4 1  # Particle/Antiparticle distinction (0=own anti)
Block QNUMBERS 250  # G0
        1 0  # 3 times electric charge
        2 1  # number of spin states (2S+1)
        3 1  # colour rep (1: singlet, 3: triplet, 8: octet)
        4 0  # Particle/Antiparticle distinction (0=own anti)
Block QNUMBERS 251  # G+
        1 3  # 3 times electric charge
        2 1  # number of spin states (2S+1)
        3 1  # colour rep (1: singlet, 3: triplet, 8: octet)
        4 1  # Particle/Antiparticle distinction (0=own anti)
'''



preamble_B = '''
######################################################################
## PARAM_CARD FOR SMEFTSIM SET B  - FLAVOR_GENERAL ALPHA_INPUTS  #####
######################################################################

###################################
## INFORMATION FOR SMINPUTS
###################################
Block SMINPUTS
    1 1.279000e+02 # aEWM1
    2 1.166370e-05 # Gf
    3 1.181000e-01 # aS

###################################
## INFORMATION FOR MASS
###################################
Block MASS
    1 4.700000e-03 # MD
    2 2.200000e-03 # MU
    3 9.600000e-02 # MS
    4 1.280000e+00 # MC
    5 4.180000e+00 # MB
    6 1.731000e+02 # MT
   11 5.110000e-04 # Me
   13 1.056600e-01 # MMU
   15 1.777000e+00 # MTA
   23 9.118750e+01 # MZ
   25 1.250900e+02 # MH
##  Not dependent paramater.
## Those values should be edited following analytical the
## analytical expression. Some generator could simply ignore
## those values and use the analytical expression
  22 0.000000 # a : 0.0
  24 79.824234 # W+ : dMW + MW0
  21 0.000000 # g : 0.0
  9000001 0.000000 # ghA : 0.0
  9000003 79.824234 # ghWp : dMW + MW0
  9000004 79.824234 # ghWm : dMW + MW0
  82 0.000000 # ghG : 0.0
  12 0.000000 # ve : 0.0
  14 0.000000 # vm : 0.0
  16 0.000000 # vt : 0.0
  251 79.824234 # G+ : dMW + MW0
  9000002 91.187500 # ghZ : MZ
  250 91.187500 # G0 : MZ

###################################
## INFORMATION FOR DECAY
###################################
DECAY   6 1.508336e+00
DECAY  23 2.495200e+00
DECAY  24 2.085000e+00
DECAY  25 4.070000e-03
##  Not dependent paramater.
## Those values should be edited following analytical the
## analytical expression. Some generator could simply ignore
## those values and use the analytical expression
DECAY  22 0.000000 # a : 0.0
DECAY  21 0.000000 # g : 0.0
DECAY  9000001 0.000000 # ghA : 0.0
DECAY  82 0.000000 # ghG : 0.0
DECAY  12 0.000000 # ve : 0.0
DECAY  14 0.000000 # vm : 0.0
DECAY  16 0.000000 # vt : 0.0
DECAY  11 0.000000 # e- : 0.0
DECAY  13 0.000000 # mu- : 0.0
DECAY  15 0.000000 # ta- : 0.0
DECAY  2 0.000000 # u : 0.0
DECAY  4 0.000000 # c : 0.0
DECAY  1 0.000000 # d : 0.0
DECAY  3 0.000000 # s : 0.0
DECAY  5 0.000000 # b : 0.0
DECAY  9000002 2.495200 # ghZ : WZ
DECAY  9000003 2.085000 # ghWp : WW
DECAY  9000004 2.085000 # ghWm : WW
DECAY  250 2.495200 # G0 : WZ
DECAY  251 2.085000 # G+ : WW

###################################
## INFORMATION FOR CKMBLOCK
###################################
Block CKMBLOCK
    1 2.277360e-01 # cabi
    2 9.743400e-01 # CKM11
    3 2.250600e-01 # CKM12
    4 3.570000e-03 # CKM13
    5 2.249200e-01 # CKM21
    6 9.735100e-01 # CKM22
    7 4.110000e-02 # CKM23
    8 8.750000e-03 # CKM31
    9 4.030000e-02 # CKM32
   10 9.991500e-01 # CKM33

'''

postamble_B = '''

###################################
## INFORMATION FOR NEWCOUPAUX
###################################
Block NEWCOUPaux
    0 1.000000e+00 # WC

###################################
## INFORMATION FOR YUKAWA
###################################
Block YUKAWA
    1 4.700000e-03 # ymdo
    2 2.200000e-03 # ymup
    3 9.600000e-02 # yms
    4 1.280000e+00 # ymc
    5 4.180000e+00 # ymb
    6 1.731000e+02 # ymt
   11 5.110000e-04 # yme
   13 1.056600e-01 # ymm
   15 1.777000e+00 # ymtau
#===========================================================
# QUANTUM NUMBERS OF NEW STATE(S) (NON SM PDG CODE)
#===========================================================

Block QNUMBERS 9000001  # ghA
        1 0  # 3 times electric charge
        2 -1  # number of spin states (2S+1)
        3 1  # colour rep (1: singlet, 3: triplet, 8: octet)
        4 1  # Particle/Antiparticle distinction (0=own anti)
Block QNUMBERS 9000002  # ghZ
        1 0  # 3 times electric charge
        2 -1  # number of spin states (2S+1)
        3 1  # colour rep (1: singlet, 3: triplet, 8: octet)
        4 1  # Particle/Antiparticle distinction (0=own anti)
Block QNUMBERS 9000003  # ghWp
        1 3  # 3 times electric charge
        2 -1  # number of spin states (2S+1)
        3 1  # colour rep (1: singlet, 3: triplet, 8: octet)
        4 1  # Particle/Antiparticle distinction (0=own anti)
Block QNUMBERS 9000004  # ghWm
        1 -3  # 3 times electric charge
        2 -1  # number of spin states (2S+1)
        3 1  # colour rep (1: singlet, 3: triplet, 8: octet)
        4 1  # Particle/Antiparticle distinction (0=own anti)
Block QNUMBERS 82  # ghG
        1 0  # 3 times electric charge
        2 -1  # number of spin states (2S+1)
        3 8  # colour rep (1: singlet, 3: triplet, 8: octet)
        4 1  # Particle/Antiparticle distinction (0=own anti)
Block QNUMBERS 250  # G0
        1 0  # 3 times electric charge
        2 1  # number of spin states (2S+1)
        3 1  # colour rep (1: singlet, 3: triplet, 8: octet)
        4 0  # Particle/Antiparticle distinction (0=own anti)
Block QNUMBERS 251  # G+
        1 3  # 3 times electric charge
        2 1  # number of spin states (2S+1)
        3 1  # colour rep (1: singlet, 3: triplet, 8: octet)
        4 1  # Particle/Antiparticle distinction (0=own anti)
'''
