#beta functions for nuSMEFT

from wilson.run.smeft.beta import *
from wilson.util import nusmeftutil
import numpy as np

def nubeta(C, HIGHSCALE=1, newphys=True):
    
    HIGHSCALE=1
    g = C["g"]
    gp = C["gp"]
    gs = C["gs"]
    m2 = C["m2"]
    yn = 0
    Lambda = C["Lambda"]

    Gu = C["Gu"]
    Gd = C["Gd"]
    Ge = C["Ge"]
    Gn = C["Gn"]

#    Gu = np.zeros((3,3))
#    Gd = np.zeros((3,3))
#    Ge = np.zeros((3,3))
#    Gn = np.zeros((3,3))

    yn = 0
    ye = -1
    yl = -1/2
    yd = -1/3
    yu = 2/3
    yq = 1/6
    yh = 1/2
    Nc = 3

    CF2 = 3/4
    b02 = 19/6
    b01 = -41/6
    
   

    GammaH = np.trace(3*Gu @ Gu.conj().T + 3*Gd @ Gd.conj().T + Ge @ Ge.conj().T)
    Gammaq = 1/2*(Gu @ Gu.conj().T + Gd @ Gd.conj().T)
    Gammau = Gu.conj().T @ Gu
    Gammad = Gd.conj().T @ Gd
    Gammal = 1/2*Ge @ Ge.conj().T
    Gammae = Ge.conj().T @ Ge
    Gamman = Gn.conj().T @ Gn #


    Xie = 2*my_einsum("prst,rs", C["le"], Ge) \
    - 3*my_einsum("ptsr,rs", C["ledq"], Gd) \
    + 3*my_einsum("ptsr,sr", C["lequ1"], np.conj(Gu)) \
    - my_einsum("vwpr,vw", C["lnle"], np.conj(Gn))

    Xid = 2*(my_einsum("prst,rs", C["qd1"], Gd) \
    + 4/3*my_einsum("prst,rs", C["qd8"], Gd)) \
    - (3*my_einsum("srpt,sr", C["quqd1"], np.conj(Gu)) \
    + 1/2*(my_einsum("prst,sr", C["quqd1"], np.conj(Gu)) \
    + 4/3*my_einsum("prst,sr", C["quqd8"], np.conj(Gu)))) \
    - my_einsum("srtp,sr", np.conj(C["ledq"]), Ge) \
    - my_einsum("vwpr,vw", C["lnqd1"], np.conj(Gn))
    
    Xiu = 2*(my_einsum("prst,rs", C["qu1"], Gu) \
    + 4/3*my_einsum("prst,rs", C["qu8"], Gu)) \
    - (3*my_einsum("ptsr,sr", C["quqd1"], np.conj(Gd)) \
    + 1/2*(my_einsum("stpr,sr", C["quqd1"], np.conj(Gd)) \
    + 4/3*my_einsum("stpr,sr", C["quqd8"], np.conj(Gd)))) \
    + my_einsum("srpt,sr", C["lequ1"], np.conj(Ge)) \
    - my_einsum("vwrp,vw", np.conj(C["lnuq"]), Gn)
    
    Xin = 2*my_einsum("pwvr,wv", C["ln"], Gn) \
    - 3*my_einsum("prvw,vw", C["lnqd1"], np.conj(Gd)) \
    - 3*my_einsum("prvw,wv", C["lnuq"], Gu) \
    - my_einsum("prvw,vw", C["lnle"], np.conj(Ge))
  
    nuBeta = beta(C, HIGHSCALE, newphys)
    
    if not newphys:
        # if there is no new physics, generate a dictionary with zero
        # Wilson coefficients (i.e. zero beta functions)
        BetaSM = nusmeftutil.C_array2dict(np.zeros(5000))
        BetaSM.update(nuBeta)
        return BetaSM

    nuBeta["Gn"] = np.zeros((3,3)) #FIXME?

    nuBeta["ll"] = nuBeta["ll"] + (1/2)*my_einsum("pr,st", Gn @ Gn.conj().T, C["phil1"]) \
    + (1/2)*my_einsum("pr,st", Gn @ Gn.conj().T, C["phil3"]) \
    + (1/2)*my_einsum("st,pr", Gn @ Gn.conj().T, C["phil1"]) \
    + (1/2)*my_einsum("st,pr", Gn @ Gn.conj().T, C["phil3"]) \
    - (1/2)*my_einsum("vs,tw,prvw", np.conj(Gn), np.conj(Gn), C["ln"]) \
    - (1/2)*my_einsum("vp,rw,stvw", np.conj(Gn), np.conj(Gn), C["ln"] ) \
    - (1/2)*my_einsum("rv,tw,pvsw", np.conj(Gn), np.conj(Ge), C["lnle"]) \
    - (1/2)*my_einsum("tw,rv,swpv", np.conj(Gn), np.conj(Ge), C["lnle"]) \
    - (1/2)*my_einsum("pv,sw,rvtw", Gn, Ge, np.conj(C["lnle"])) \
    - (1/2)*my_einsum("sw,pv,twrv", Gn, Ge, np.conj(C["lnle"])) 

    nuBeta["lq1"] = nuBeta["lq1"] + my_einsum("pr,st", Gn @ Gn.conj().T, C["phiq1"])\
    - my_einsum("pv,rw,stvw", Gn, np.conj(Gn), C["qn"]) \
    + (1/4)*my_einsum("rv,ws,pvwt", np.conj(Gn), np.conj(Gu), C["lnuq"]) \
    + (1/4)*my_einsum("pv,wt,rvws", Gn, Gu, np.conj(C["lnuq"])) \
    - (1/4)*my_einsum("rv,tw,pvsw", np.conj(Gn), np.conj(Gd), C["lnqd1"]) \
    - (1/4)*my_einsum("pv,sw,rvtw", Gn, Gd, np.conj(C["lnqd1"])) \
    + 3*my_einsum("rv,tw,pvsw", np.conj(Gn), np.conj(Gd), C["lnqd3"]) \
    + 3*my_einsum("pv,sw,rvtw", Gn, Gd, np.conj(C["lnqd3"]))
    

    nuBeta["lq3"] = nuBeta["lq3"] + my_einsum("pr,st", Gn @ Gn.conj().T, C["phiq3"]) \
    + (1/4)*my_einsum("rv,ws,pvwt", np.conj(Gn), np.conj(Gu), C["lnuq"]) \
    + (1/4)*my_einsum("pv,wt,rvws", Gn, Gu, np.conj(C["lnuq"])) \
    + (1/4)*my_einsum("rv,tw,pvsw", np.conj(Gn), Gd, C["lnqd1"]) \
    + (1/4)*my_einsum("pv,sw,rvtw", Gn, Gd, np.conj(C["lnqd1"])) \
    - 3*my_einsum("rv,tw,pvsw", np.conj(Gn), np.conj(Gd), C["lnqd3"]) \
    - 3*my_einsum("pv,sw,rvtw", Gn, Gd, C["lnqd3"])

    nuBeta["ld"] = nuBeta["ld"] + my_einsum("pr,st", Gn @ Gn.conj().T, C["phid"]) \
    - my_einsum("vp,wr,vwst", np.conj(Gn), Gn, C["nd"]) \
    + (1/2)*my_einsum("ws,rv,pvwt", np.conj(Gd), np.conj(Gn), C["lnqd1"]) \
    + (1/2)*my_einsum("wt,pv,rvws", Gd, Gn, np.conj(C["lnqd1"])) \
    + 6*my_einsum("ws,rv,pvwt", np.conj(Gd), np.conj(Gn), C["lnqd3"]) \
    + 6*my_einsum("wt,pv,rvws", Gd, Gn, np.conj(C["lnqd3"]))

    nuBeta["lu"] = nuBeta["lu"] + my_einsum("pr,st", Gn @ Gn.conj().T, C["phiu"]) 
    - my_einsum("vp,wr,vwst", np.conj(Gn), Gn, C["nu"]) \
    - (1/2)*my_einsum("wt,rv,pvsw", Gu, np.conj(Gn), C["lnuq"]) \
    - (1/2)*my_einsum("ws,pv,rvtw", np.conj(Gu), Gn, np.conj(C["lnuq"])) 

    nuBeta["le"] = nuBeta["le"] + my_einsum("rs,pt",np.conj(Ge), Xie) \
    + my_einsum("pt,rs", Ge, np.conj(Xie)) \
    + my_einsum("pr,st", Gn @ Gn.conj().T, C["phie"]) \
    - my_einsum("vp,wr,vwst", np.conj(Gn), Gn, C["ne"]) \
    + (1/2)*my_einsum("ws,rv,pvwt", np.conj(Ge), np.conj(Gn), C["lnle"]) \
    + (1/2)*my_einsum("wt,pv,rvws", Ge, Gn, np.conj(C["lnle"]))

    nuBeta["qu1"] = nuBeta["qu1"] + (1/Nc)*my_einsum("rs,pt", np.conj(Gu), Xiu) \
    + (1/Nc)*my_einsum("pt,rs", Gu, np.conj(Xiu))

    nuBeta["qu8"] = nuBeta["qu8"] + 2*my_einsum("rs,pt", np.conj(Gu), Xiu) \
    + 2*my_einsum("pt,rs", Gu, np.conj(Xiu))

    nuBeta["qd1"] = nuBeta["qd1"] + (1/Nc)*my_einsum("rs,pt", np.conj(Gd), Xid) \
    + (1/Nc)*my_einsum("pt,rs", Gd, np.conj(Xid))

    nuBeta["qd8"] = nuBeta["qd8"] + 2*my_einsum("rs,pt", np.conj(Gd), Xid) \
    + 2*my_einsum("pt,rs", Gd, np.conj(Xid))

    nuBeta["ledq"] = nuBeta["ledq"] - 2*my_einsum("ts,pr", np.conj(Gd), Xie) \
    - 2*my_einsum("pr,ts", Ge, np.conj(Xid)) \
    + 2*my_einsum("pv,wr,wvst", Gn, Ge, np.conj(C["lnqd1"])) \
    + 2*my_einsum("pv,wt,vrsw", Gn, np.conj(Gu), C["nedu"])

    nuBeta["lequ1"] = nuBeta["lequ1"] + 2*my_einsum("st,pr", Gu, Xie) \
    + 2*my_einsum("pr,st", Ge, Xiu) \
    + 2*my_einsum("pv,wr,wvts", Gn, Ge, np.conj(C["lnuq"])) \
    - 2*my_einsum("pv,sw,vrwt", Gn, Gd, C["nedu"])

    nuBeta["lequ3"] = nuBeta["lequ3"] \
    + (1/2)*my_einsum("pv,sw,vrwt", Gn, Gd, C["nedu"])

    nuBeta["quqd1"] = nuBeta["quqd1"] - 2*my_einsum("pr,st", Gu, Xid) \
    - 2*my_einsum("st,pr", Gd, Xiu)

    #SMNEFT operators 

    #RRRR operators
    nuBeta["nd"] = (4/3)*Nc*yd**2*gp**2*my_einsum("prww,st", C["nd"], I3)\
    +(4/3)*Nc*yd*yu*gp**2*my_einsum("prww,st", C["nu"], I3) \
    +(4/3)*yd*ye*gp**2*my_einsum("prww,st", C["ne"], I3) \
    +(8/3)*Nc*yd*yq*gp**2*my_einsum("wwpr,st", C["qn"], I3) \
    +(8/3)*yd*yl*gp**2*my_einsum("wwpr,st", C["ln"], I3) \
    +(4/3)*yd*yh*gp**2*my_einsum("pr,st", C["phin"], I3) \
    - 2*my_einsum("pr,st", Gn.conj().T @ Gn, C["phid"]) \
    + 2*my_einsum("st,pr", Gd.conj().T @ Gd, C["phin"]) \
    - 2*my_einsum("vp,wr,vwst", np.conj(Gn), Gn, C["ld"]) \
    - 2*my_einsum("vs,wt,vwpr", np.conj(Gd), Gd, C["qn"]) \
    - 1*my_einsum("vp,ws,vrwt", np.conj(Gn), np.conj(Gd), C["lnqd1"]) \
    - 1*my_einsum("vr,wt,vpws", Gn, Gd, np.conj(C["lnqd1"])) \
    + 12*my_einsum("vp,ws,vrwt", np.conj(Gn), np.conj(Gd), C["lnqd3"]) \
    + 12*my_einsum("vr,wt,vpws", Gn, Gd, np.conj(C["lnqd3"])) \
    + my_einsum("pv,vrst", Gamman, C["nd"]) \
    + my_einsum("sv,prvt", Gammad, C["nd"]) \
    + my_einsum("pvst,vr", C["nd"], Gamman) \
    + my_einsum("prsv,vt", C["nd"], Gammad)


    nuBeta["nu"] = (4/3)*Nc*yu*yd*gp**2*my_einsum("prww,st", C["nd"], I3) \
    +(4/3)*Nc*yu**2*gp**2*my_einsum("prww,st", C["nu"], I3) \
    +(4/3)*yu*ye*gp**2*my_einsum("prww,st", C["ne"], I3) \
    +(8/3)*Nc*yu*yq*gp**2*my_einsum("wwpr,st", C["qn"], I3) \
    +(8/3)*yu*yl*gp**2*my_einsum("wwpr,st", C["ln"], I3) \
    +(4/3)*yu*yh*gp**2*my_einsum("pr,st", C["phin"], I3) \
    - 2*my_einsum("pr,st", Gn.conj().T @ Gn, C["phiu"]) \
    - 2*my_einsum("st,pr", Gu.conj().T @ Gu, C["phie"]) \
    - 2*my_einsum("vp,wr,vwst", np.conj(Gn), Gn, C["lu"]) \
    - 2*my_einsum("vs,wt,vwpr", np.conj(Gu), Gu, C["qn"]) \
    + my_einsum("vp,wt,vrsw", np.conj(Gn), Gu, C["lnuq"]) \
    + my_einsum("vr,ws,vptw", Gn, np.conj(Gu), np.conj(C["lnuq"])) \
    + my_einsum("pv,vrst", Gamman, C["nu"]) \
    + my_einsum("sv,prvt", Gammau, C["nu"]) \
    + my_einsum("pvst,vr", C["nu"], Gamman) \
    + my_einsum("prsv,vt", C["nu"], Gammau)
    
    nuBeta["ne"] = (4/3)*Nc*ye*yd*gp**2*my_einsum("prww,st", C["nd"], I3) \
    +(4/3)*Nc*ye*yu*gp**2*my_einsum("prww,st", C["nu"], I3) \
    +(4/3)*ye**2*gp**2*my_einsum("prww,st", C["ne"], I3) \
    +(8/3)*Nc*ye*yq*gp**2*my_einsum("wwpr,st", C["qn"], I3) \
    +(8/3)*ye*yl*gp**2*my_einsum("wwpr,st", C["ln"], I3) \
    +(4/3)*ye*yh*gp**2*my_einsum("pr,st", C["phin"], I3) \
    -2*my_einsum("pr,st",Gn.conj().T @ Gn, C["phie"]) \
    +2*my_einsum("st,pr",Ge.conj().T @ Ge, C["phin"]) \
    +2*my_einsum("sr,pt",Ge.conj().T @ Gn, C["phine"]) \
    +2*my_einsum("pt,rs",Gn.conj().T @ Ge, np.conj(C["phine"])) \
    -2*my_einsum("vp,wr,vwst",np.conj(Gn), Gn, C["le"]) \
    -1*my_einsum("vp,ws,vrwt",np.conj(Gn), np.conj(Ge), C["lnle"]) \
    -1*my_einsum("vr,wt,vpws",Gn, Ge, np.conj(C["lnle"]) ) \
    +1*my_einsum("wp,vs,vrwt",np.conj(Gn), np.conj(Ge), C["lnle"]) \
    +1*my_einsum("wr,vt,vpws",Gn, Ge, np.conj(C["lnle"]) ) \
    -2*my_einsum("vs,wt,vwpr",np.conj(Ge), Ge, C["ln"]) \
    + my_einsum("pv,vrst", Gamman, C["ne"]) \
    + my_einsum("sv,prvt", Gammae, C["ne"]) \
    + my_einsum("pvst,vr", C["ne"], Gamman) \
    + my_einsum("prsv,vt", C["ne"], Gammae)
    
    nuBeta["nn"] = - my_einsum("pr,st", Gn.conj().T @ Gn, C["phin"]) \
    - my_einsum("st,pr", Gn.conj().T @ Gn, C["phin"]) \
    - my_einsum("vp,wr,vwst", np.conj(Gn), Gn, C["ln"]) \
    - my_einsum("vs,wt,vwpr", np.conj(Gn), Gn, C["ln"]) \
    + my_einsum("pv,vrst", Gamman, C["nn"]) \
    + my_einsum("sv,prvt", Gamman, C["nn"]) \
    + my_einsum("pvst,vr", C["nn"], Gamman) \
    + my_einsum("pvsv,vt", C["nn"], Gamman) 
     
    nuBeta["nedu"] =((yd-yu)**2 + ye*(ye + 8*yu - 2*yd))*gp**2*my_einsum("prst",C["nedu"])\
    + 2*my_einsum("st,pr", Gd.conj().T @ Gu, C["phine"]) \
    + 2*my_einsum("pr,ts", Gn.conj().T @ Ge, C["phiud"]) \
    - 1*my_einsum("vp,ws,vrwt", np.conj(Gn), np.conj(Gd), C["lequ1"]) \
    + 12*my_einsum("vp,ws,vrwt", np.conj(Gn), np.conj(Gd), C["lequ3"]) \
    + 1*my_einsum("vp,wt,vrsw", np.conj(Gn), Gu, C["ledq"]) \
    + 1*my_einsum("vr,wt,vpws", Ge, Gu, np.conj(C["lnqd1"]) ) \
    - 12*my_einsum("vr,wt,vpws", Ge, Gu, np.conj(C["lnqd3"]) ) \
    + 1*my_einsum("vr,ws,vptw", Ge, np.conj(Gd), C["lnuq"]) \
    + my_einsum("pv,vrst", Gamman, C["nedu"]) \
    + my_einsum("sv,prvt", Gammad, C["nedu"]) \
    + my_einsum("pvst,vr", C["nedu"], Gammae) \
    + my_einsum("prsv,vt", C["nedu"], Gammau)

    #LLRR Operators

    nuBeta["qn"] =(4/3)*Nc*yq*yd*gp**2*my_einsum("stww,pr", C["nd"], I3) \
    +(4/3)*Nc*yq*yu*gp**2*my_einsum("stww,pr", C["nu"], I3) \
    +(4/3)*yq*ye*gp**2*my_einsum("stww,pr", C["ne"], I3) \
    +(8/3)*Nc*yq**2*gp**2*my_einsum("wwst,pr", C["qn"], I3) \
    +(8/3)*yq*yl*gp**2*my_einsum("wwst,pr", C["ln"], I3) \
    +(4/3)*yq*yh*gp**2*my_einsum("st,pr", C["phin"], I3) \
    + my_einsum("pv,vrst", Gammaq, C["qn"]) \
    + my_einsum("sv,prvt", Gamman, C["qn"]) \
    + my_einsum("pvst,vr", C["qn"], Gammaq) \
    + my_einsum("prsv,vt", C["qn"], Gamman)
   
    nuBeta["ln"] = (4/3)*Nc*yl*yd*gp**2*my_einsum("stww,pr", C["nd"], I3) \
    +(4/3)*Nc*yl*yu*gp**2*my_einsum("stww,pr", C["nu"], I3) \
    +(4/3)*yl*ye*gp**2*my_einsum("stww,pr", C["ne"], I3) \
    +(8/3)*Nc*yl*yq*gp**2*my_einsum("wwst,pr", C["qn"], I3) \
    +(8/3)*yl**2*gp**2*my_einsum("wwst,pr", C["ln"], I3) \
    +(4/3)*yl*yh*gp**2*my_einsum("st,pr", C["phin"], I3) \
    + my_einsum("pv,vrst", Gammal, C["ln"]) \
    + my_einsum("sv,prvt", Gamman, C["ln"]) \
    + my_einsum("pvst,vr", C["ln"], Gammal) \
    + my_einsum("prsv,vt", C["ln"], Gamman)

    #LRRL and LRLR operators 
   
    nuBeta["lnle"] = ((ye**2 - 8*ye*yl + 6*yl**2)*gp**2 - (3/2)*g**2)*my_einsum("prst", C["lnle"]) \
    -(4*yl*(ye+yl)*gp**2 - 3*g**2)*my_einsum("srpt",C["lnle"])\
    -4*my_einsum("vr,wt,pvsw", Gn, Ge, C["ll"])\
    +4*my_einsum("vr,wt,svpw", Gn, Ge, C["ll"])\
    +4*my_einsum("wr,vt,pvsw", Gn, Ge, C["ll"])\
    -4*my_einsum("wt,vt,svpw", Gn, Ge, C["ll"])\
    -4*my_einsum("pv,sw,vrwt", Gn, Ge, C["ne"])\
    +4*my_einsum("sv,pw,vrwt", Gn, Ge, C["ne"])\
    +4*my_einsum("sw,vt,pvwr", Gn, Ge, C["ln"])\
    -4*my_einsum("vr,pw,svwt", Gn, Ge, C["le"])\
    +4*gp*(ye+yl)*my_einsum("pr,st", C["nB"], Ge)\
    -8*gp*(ye+yl)*my_einsum("sr,pt", C["nB"], Ge)\
    -6*g*my_einsum("pr,st", C["nB"], Ge)\
    +12*g*my_einsum("sr,pt", C["nW"], Ge)\
    +4*gp*(yn+yl)*my_einsum("st,pr", C["eB"], Gn)\
    -8*gp*(yn+yl)*my_einsum("pt,sr", C["eB"], Gn)\
    -6*g*my_einsum("st,pr", C["eW"], Gn)\
    +12*g*my_einsum("pt,sr", C["eW"], Gn)\
    -2*my_einsum("pr,st", Xin, Ge) \
    -2*my_einsum("st,pr", Xie, Gn) \
    + my_einsum("pv,vrst", Gammal, C["lnle"]) \
    + my_einsum("sv,prvt", Gammal, C["lnle"]) \
    + my_einsum("pvst,vr", C["lnle"], Gamman) \
    + my_einsum("prsv,vt", C["lnle"], Gammae)
   
    nuBeta["lnqd1"] = ((yd**2 - 2*yd*(yl + 4*yq) + (yl + yq)**2)*gp**2 - 8*gs**2)*my_einsum("prst", C["lnqd1"]) \
    + (-24*yl*(yd + yq)*gp**2 + 18*g**2)*my_einsum("prst", C["lnqd3"])\
    -2*my_einsum("vr,pw,vwts", Gn, Ge, np.conj(C["ledq"]))\
    +2*my_einsum("pw,vt,svwr", Gn, Gd, C["qn"])\
    +2*my_einsum("pw,sv,rwtv", Ge, Gu, np.conj(C["nedu"]))\
    +2*my_einsum("vr,sw,pvwt", Gn, Gd, C["ld"])\
    -2*my_einsum("wt,sv,prvw", Gd, Gu, np.conj(C["lnuq"]))\
    -2*my_einsum("pw,sv,wrvt", Gn, Gd, C["nd"])\
    -2*my_einsum("vr,wt,pvsw", Gn, Gd, np.conj(C["lq1"]))\
    +6*my_einsum("vr,wt,pvsw", Gn, Gd, C["lq3"])\
    -2*my_einsum("pr,st", Xin, Gd) \
    -2*my_einsum("st,pr", Xid, Gn) \
    + my_einsum("pv,vrst", Gammal, C["lnqd1"]) \
    + my_einsum("sv,prvt", Gammaq, C["lnqd1"]) \
    + my_einsum("pvst,vr", C["lnqd1"], Gamman) \
    + my_einsum("prsv,vt", C["lnqd1"], Gammad)
    

    nuBeta["lnqd3"] = (-(1/2)*yl*(yd+yq)*gp**2 + (3/8)*g**2)*my_einsum("prst",C["lnqd1"])\
    +((yd**2 - 6*yd*yl + yl**2 + 6*yl*yq + yq**2)*gp**2 - 3*g**2 + (8/3)*gs**2)*my_einsum("prst",C["lnqd3"])\
    -(1/2)*my_einsum("pw,sv,rwtv", Ge, Gu, np.conj(C["nedu"]))\
    +(1/2)*my_einsum("vr,wt,pvsw", Gn, Gd, C["lq1"])\
    -(3/2)*my_einsum("vr,wt,pvsw", Gn, Gd, C["lq3"])\
    +(1/2)*my_einsum("vr,sw,pvwt", Gn, Gd, C["ld"])\
    +(1/2)*my_einsum("pw,vt,svwr", Gn, Gd, C["qn"])\
    +(1/2)*my_einsum("pw,sv,wrvt", Gn, Gd, C["nd"])\
    -gp*(yd+yq)*my_einsum("pr,st", C["nB"], Gd)\
    -gp*(yn+yl)*my_einsum("st,pr", C["dB"], Gn)\
    +(3/2)*g*my_einsum("pr,st", C["nW"], Gd)\
    +(3/2)*g*my_einsum("st,pr", C["dW"], Gn)\
    + my_einsum("pv,vrst", Gammal, C["lnqd3"]) \
    + my_einsum("sv,prvt", Gammaq, C["lnqd3"]) \
    + my_einsum("pvst,vr", C["lnqd3"], Gamman) \
    + my_einsum("prsv,vt", C["lnqd3"], Gammad)



    nuBeta["lnuq"] = (((yl+yu)**2 + yq*(yq - 2*yl -8*yu))*gp**2 - 8*gs**2)*my_einsum("prst",C["lnuq"]) \
    +2*my_einsum("wr,pv,wvts", Gn, Ge, np.conj(C["lequ1"])) \
    -2*my_einsum("pw,vs,vtwr", Gn, np.conj(Gu), C["qn"])\
    +2*my_einsum("pw,tv,wrsv", Gn, np.conj(Gu), C["nu"])\
    +2*my_einsum("pw,tv,rwvs", Ge, np.conj(Gd), np.conj(C["nedu"]))\
    +2*my_einsum("vr,tw,pvwt", Gn, np.conj(Gu), C["lq1"])\
    +6*my_einsum("vr,tw,pvwt", Gn, np.conj(Gu), C["lq3"])\
    -2*my_einsum("vr,tw,pvsw", Gn, np.conj(Gu), C["lu"])\
    -2*my_einsum("sv,wt,prvw", Gu, np.conj(Gd), C["lnqd1"])\
    -2*my_einsum("pr,ts", Xin, np.conj(Gu)) \
    -2*my_einsum("st,pr", np.conj(Xiu), Gn) \
    + my_einsum("pv,vrst", Gammal, C["lnuq"]) \
    + my_einsum("sv,prvt", Gammau, C["lnuq"]) \
    + my_einsum("pvst,vr", C["lnuq"], Gamman) \
    + my_einsum("prsv,vt", C["lnuq"], Gammaq)



    #psi2phi3 tems
    nuBeta["nphi"] = -(9*yl**2*gp**2 + (27/4)*g**2)*my_einsum("pr",C["nphi"]) \
    -6*(4*yh**2*yl*gp**3 - yh*gp*g**2)*my_einsum("pr",C["nB"]) \
    +3*(4*yh*yl*gp**2*g - 3*g**3)*my_einsum("pr",C["nW"])
    

    #psi2phi2D terms

    nuBeta["phin"] = (4/3)*yh**2*gp**2*my_einsum("pr",C["phin"])\
    +(4/3)*Nc*yd*yh*gp**2*my_einsum("prww",C["nd"])\
    +(4/3)*Nc*yu*yh*gp**2*my_einsum("prww", C["nu"]) \
    +(4/3)*ye*yh*gp**2*my_einsum("prww", C["ne"]) \
    +(8/3)*Nc*yq*yh*gp**2*my_einsum("wwpr", C["qn"]) \
    +(8/3)*yl*yh*gp**2*my_einsum("wwpr", C["ln"]) 
    
    #psi2Xphi
    nuBeta["nW"] = ((3*CF2 - b02)*g**2 - 3*yl**2*gp**2 )*my_einsum("pr",C["nW"])\
    +3*yl*gp*g*my_einsum("pr",C["nB"]) + (g/4)*my_einsum("srpt,ts", C["lnle"], Ge) + 2*g*Nc*my_einsum("prst,st", C["lnqd3"], Gd)

    nuBeta["nB"] = (-3*CF2*g**2 + (3*yl**2 - b01)*gp**2 )*my_einsum("pr",C["nB"]) \
    +12*CF2*yl*gp*g*my_einsum("pr",C["nW"]) 


    nuBeta["phine"] = -3*ye**2*my_einsum("pr", C["phine"])*gp**2


    return nuBeta
