import rundec
from functools import lru_cache


def _sane(scale, f):
    """Check if scale and no. of flavours are sane"""
    if not isinstance(scale, (float, int)) or scale <= 0:
        raise ValueError("Scale must be a positive number")
    if not isinstance(f, int) and 3 <= f <= 6:
        raise ValueError("f must be an integer between 3 and 6")


MZ = 91.1876


@lru_cache(32)
def alpha_s(scale, f, alphasMZ=0.1185):
    """3-loop cumputation of alpha_s for f flavours
    with initial condition alpha_s(MZ) = 0.1185"""
    if scale == MZ and f == 5:
        return alphasMZ  # nothing to do
    _sane(scale, f)
    loop = 3
    crd = rundec.CRunDec()
    if f == 5:
        return crd.AlphasExact(alphasMZ, MZ, scale, f, loop)
    elif f == 6:
        crd.nfMmu.Mth = 170
        crd.nfMmu.muth = 170
        crd.nfMmu.nf = 6
        return crd.AlL2AlH(alphasMZ, MZ, crd.nfMmu, scale, loop)
    elif f == 4:
        crd.nfMmu.Mth = 4.8
        crd.nfMmu.muth = 4.8
        crd.nfMmu.nf = 5
        return crd.AlH2AlL(alphasMZ, MZ, crd.nfMmu, scale, loop)
    elif f == 3:
        crd.nfMmu.Mth = 4.8
        crd.nfMmu.muth = 4.8
        crd.nfMmu.nf = 5
        mc = 1.3
        asmc =  crd.AlH2AlL(alphasMZ, MZ, crd.nfMmu, mc, loop)
        crd.nfMmu.Mth = mc
        crd.nfMmu.muth = mc
        crd.nfMmu.nf = 4
        return crd.AlH2AlL(asmc, mc, crd.nfMmu, scale, loop)
    else:
        raise ValueError("Invalid input: f={}, scale={}".format(f, scale))


@lru_cache(32)
def m_b(mbmb, scale, f, alphasMZ=0.1185):
    r"""Get running b quark mass in the MSbar scheme at the scale `scale`
    in the theory with `f` dynamical quark flavours starting from $m_b(m_b)$"""
    if scale == mbmb and f == 5:
        return mbmb  # nothing to do
    _sane(scale, f)
    loop = 3
    alphas_mb = alpha_s(mbmb, 5, alphasMZ=alphasMZ)
    crd = rundec.CRunDec()
    if f == 5:
        alphas_scale = alpha_s(scale, f, alphasMZ=alphasMZ)
        return crd.mMS2mMS(mbmb, alphas_mb, alphas_scale, f, loop)
    elif f == 4:
        crd.nfMmu.Mth = 4.8
        crd.nfMmu.muth = 4.8
        crd.nfMmu.nf = 5
        return crd.mH2mL(mbmb, alphas_mb, mbmb, crd.nfMmu, scale, loop)
    elif f == 3:
        mc = 1.3
        crd.nfMmu.Mth = 4.8
        crd.nfMmu.muth = 4.8
        crd.nfMmu.nf = 5
        mbmc = crd.mH2mL(mbmb, alphas_mb, mbmb, crd.nfMmu, mc, loop)
        crd.nfMmu.Mth = mc
        crd.nfMmu.muth = mc
        crd.nfMmu.nf = 4
        alphas_mc = alpha_s(mc, 4, alphasMZ=alphasMZ)
        return crd.mH2mL(mbmc, alphas_mc, mc, crd.nfMmu, scale, loop)
    elif f == 6:
        crd.nfMmu.Mth = 170
        crd.nfMmu.muth = 170
        crd.nfMmu.nf = 6
        return crd.mL2mH(mbmb, alphas_mb, mbmb, crd.nfMmu, scale, loop)
    else:
        raise ValueError("Invalid input: f={}, scale={}".format(f, scale))


@lru_cache(32)
def m_c(mcmc, scale, f, alphasMZ=0.1185):
    r"""Get running c quark mass in the MSbar scheme at the scale `scale`
    in the theory with `f` dynamical quark flavours starting from $m_c(m_c)$"""
    if scale == mcmc:
        return mcmc  # nothing to do
    _sane(scale, f)
    loop = 3
    crd = rundec.CRunDec()
    alphas_mc = alpha_s(mcmc, 4, alphasMZ=alphasMZ)
    if f == 4:
        alphas_scale = alpha_s(scale, f, alphasMZ=alphasMZ)
        return crd.mMS2mMS(mcmc, alphas_mc, alphas_scale, f, loop)
    elif f == 3:
        crd.nfMmu.Mth = 1.3
        crd.nfMmu.muth = 1.3
        crd.nfMmu.nf = 4
        return crd.mH2mL(mcmc, alphas_mc, mcmc, crd.nfMmu, scale, loop)
    elif f == 5:
        crd.nfMmu.Mth = 4.8
        crd.nfMmu.muth = 4.8
        crd.nfMmu.nf = 5
        return crd.mL2mH(mcmc, alphas_mc, mcmc, crd.nfMmu, scale, loop)
    else:
        raise ValueError("Invalid input: f={}, scale={}".format(f, scale))


@lru_cache(32)
def m_s(ms2, scale, f, alphasMZ=0.1185, loop=3):
    r"""Get running s quark mass in the MSbar scheme at the scale `scale`
    in the theory with `f` dynamical quark flavours starting from $m_s(2 \,\text{GeV})$"""
    if scale == 2 and f == 3:
        return ms2  # nothing to do
    _sane(scale, f)
    crd = rundec.CRunDec()
    alphas_2 = alpha_s(2, 3, alphasMZ=alphasMZ)
    if f == 3:
        alphas_scale = alpha_s(scale, f, alphasMZ=alphasMZ)
        return crd.mMS2mMS(ms2, alphas_2, alphas_scale, f, loop)
    elif f == 4:
        crd.nfMmu.Mth = 1.3
        crd.nfMmu.muth = 1.3
        crd.nfMmu.nf = 4
        return crd.mL2mH(ms2, alphas_2, 2, crd.nfMmu, scale, loop)
    elif f == 5:
        mc = 1.3
        crd.nfMmu.Mth = mc
        crd.nfMmu.muth = mc
        crd.nfMmu.nf = 4
        msmc = crd.mL2mH(ms2, alphas_2, 2, crd.nfMmu, mc, loop)
        crd.nfMmu.Mth = 4.8
        crd.nfMmu.muth = 4.8
        crd.nfMmu.nf = 5
        alphas_mc = alpha_s(mc, 4, alphasMZ=alphasMZ)
        return crd.mL2mH(msmc, alphas_mc, mc, crd.nfMmu, scale, loop)
    else:
        raise ValueError("Invalid input: f={}, scale={}".format(f, scale))
