"""Defines the EFTevolve class that provides the main API to smeft and nusmeft."""
from . import smpar
from wilson.run.smeft import definitions
import numpy as np
import ckmutil.phases, ckmutil.diag
import wilson
from wilson.util import smeftutil, nusmeftutil, wetutil
from wilson import wcxf
from copy import deepcopy
from math import pi, log, sqrt
from scipy.integrate import solve_ivp
from collections import OrderedDict

eftutil = {'SMEFT': smeftutil, 'nuSMEFT': nusmeftutil, 'WET': wetutil}

class EFTevolve:
    """?? Class representing a parameter point in the Standard Model Effective
       Field Theory and other EFTs such as nuSMEFT  and allowing the evolution 
       of the Wilson Coefficients.

    Methods:

    - __init__: Initialize, given a wcxf.WC instance
    - run: solve the RGE and return a wcxf.WC instance
    """

    def __init__(self, wc, beta, get_smpar=True):
        """Initialize the SMEFT instance.

        Parameters:

        - `wc`: the Wilson coefficients as `wcxf.WC` instance.
           `beta`: is a function of WCs which returns beta-fucntion dictionary 
            for SMEFT (beta.beta) or nuSMEFT (beta_numseft.nubeta)
        """
        self.wc = wc
#       self.scale_in = wc.scale
        self.beta = beta

        self.scale_in = None
        self.C_in = None

        if wc is not None:
            self.basis = wc.basis
            self.eft = wc.eft
            self._set_initial_wcxf(wc, get_smpar=get_smpar)

    def ext_par_scale_in(self,par):
        if par is not None:
            self.C_in.update(par)

    def _set_initial(self, C_in, scale_in):
        r"""Set the initial values for parameters and Wilson coefficients at
        the scale `scale_in`."""
        self.C_in = C_in
        self.scale_in = scale_in

    def _set_initial_wcxf(self, wc, get_smpar=True):
        """Load the initial values for Wilson coefficients from a
        wcxf.WC instance.

        Parameters:

        - `get_smpar`: boolean, optional, defaults to True. If True, an attempt
          is made to determine the SM parameters from the requirement of
          reproducing the correct SM masses and mixings at the electroweak
          scale. As approximations are involved, the result might or might not
          be reliable, depending on the size of the Wilson coefficients
          affecting the SM masses and mixings. If False, Standard Model
          parameters have to be provided separately and are assumed to be in
          the weak basis used for the Warsaw basis as defined in WCxf,
          i.e. in the basis where the down-type and charged lepton mass
          matrices are diagonal.
        """
        if wc.eft not in ('SMEFT','nuSMEFT'):
            raise ValueError("Wilson coefficients use wrong or unknown EFT.")
        if wc.basis not in ('Warsaw'):
            raise ValueError("Wilson coefficients use wrong or unknown basis.")
        self.scale_in = wc.scale 
        C = eftutil[self.eft].wcxf2arrays_symmetrized(wc.dict)
        # fill in zeros for missing WCs
        for k, s in eftutil[self.eft].C_keys_shape.items():
            if k not in C and k not in eftutil[self.eft].dim4_keys:
                if s == 1:
                    C[k] = 0
                else:
                    C[k] = np.zeros(s)
        if self.C_in is None:
            self.C_in = C
        else:
            self.C_in.update(C)
        if get_smpar:
            self.C_in.update(self._get_sm_scale_in())


    def _to_wcxf(self, C_out, scale_out):
        """Return the Wilson coefficients `C_out` as a wcxf.WC instance.

        Note that the Wilson coefficients are rotated into the Warsaw basis
        as defined in WCxf, i.e. to the basis where the down-type and charged
        lepton mass matrices are diagonal."""
        C = self._rotate_defaultbasis(C_out)
        d = eftutil[self.eft].arrays2wcxf_nonred(C)
        d = wcxf.WC.dict2values(d)
        wc = wcxf.WC(self.eft, self.basis, scale_out, d) 
        return wc
    
    def beta_reordered(self, C, HIGHSCALE=1, *args, **kwargs):
        """Returns a beta dictionary but changes its order to match that is obtained by 
           EFTutil class. It is essential, because when flattening the array the 
           order is lost and it is assumed that keys from beta and keys from EFTutil are in order. 
           Subsequently, they are used in evolution of WCs, where a mismatch  of flattened arrays 
           could lead to incorrect solutions, if not reordered"""
        ordered_beta_dict = {}
        _keys = eftutil[self.eft].C_keys
        _beta = self.beta(C, HIGHSCALE, *args, **kwargs)
        ordered_beta_dict = OrderedDict({k: _beta[k] for k in _keys})
        return ordered_beta_dict

    def beta_array_reordered(self, C, HIGHSCALE=1, *args, **kwargs):
        """Return the beta functions of all SM parameters and SMEFT/nuSMEFT Wilson
        coefficients as a 1D numpy array."""
        return np.hstack([np.asarray(b).ravel() for b in self.beta_reordered(C, HIGHSCALE=1, *args, **kwargs).values()])
    
    def eft_evolve_leadinglog(self, scale_out, newphys=True):
        """Solve the EFT (SMEFT and nuSMEFT) RGEs in the leading log approximation.
        Input C_in and output C_out are dictionaries of arrays."""
        self._check_initial()
        C_out = deepcopy(self.C_in)
        b = self.beta_reordered(self.C_in, newphys=newphys) 
        for k, C in C_out.items():
            C_out[k] = C + b[k] / (16 * pi**2) * log(scale_out / self.scale_in)
        return C_out
    
    def _eft_evolve(self, scale_out, newphys=True, **kwargs):
        """Axuliary function used in `eft_evolve` and `eft_evolve_continuous`"""
        def fun(t0, y):
            return self.beta_array_reordered(C=eftutil[self.eft].C_array2dict(y.view(complex)),
                               newphys=newphys).view(float) / (16 * pi**2)
        y0 = eftutil[self.eft].C_dict2array(self.C_in).view(float)
        sol = solve_ivp(fun=fun,
                    t_span=(log(self.scale_in), log(scale_out)),
                    y0=y0, **kwargs)
        return sol
    
    def eft_evolve(self, scale_out, newphys=True, **kwargs):
        """Solve the SMEFT/nuSMEFT RGEs by numeric integration.

        Input C_in and output C_out are dictionaries of arrays."""
        self._check_initial()
        sol = self._eft_evolve(scale_out, newphys=newphys, **kwargs)
        C_out= eftutil[self.eft].C_array2dict(sol.y[:, -1].view(complex))
        return C_out

    def run(self, scale, accuracy='integrate', **kwargs):
        """Return the Wilson coefficients  (as wcxf.WC instance) evolved to the
        scale `scale`.

        Parameters:

        - `scale`: scale in GeV
        - accuracy: whether to use the numerical solution to the RGE
        ('integrate', the default, slow but precise) or the leading logarithmic
        approximation ('leadinglog', approximate but much faster).
        """
        if accuracy == 'integrate':
            C_out = self.eft_evolve(scale, **kwargs)
        elif accuracy == 'leadinglog':
            C_out = self.eft_evolve_leadinglog(scale)
        else:
            raise ValueError(f"'{accuracy}' is not a valid value of 'accuracy' (must be either 'integrate' or 'leadinglog').")
        return self._to_wcxf(C_out, scale)


    def _run_sm_scale_in(self, C_out, scale_sm=91.1876):
        """Get the dim4 parameters at the EW scale according to EFT, using an estimate `C_out`
        of the Wilson coefficients at that scale, and run them to the input scale."""
        """New: Before, this function ran only SM parameters. But now it makes more sense to 
        call them dim4 parameters, as for nusmeft,it will contain Gn, which technically is not 
        a SM parameter"""

        # set the SM parameters to the values obtained from smpar.eftpar 
        eftpar = {'SMEFT': smpar.smeftpar, 'nuSMEFT': smpar.nusmeftpar} #FIXME Correct Gn in smpar file
        C_SM = eftpar[self.eft](scale_sm, C_out, basis='Warsaw')
        SM_keys = set(eftutil[self.eft].dim4_keys)  # to speed up lookup
        C_SM = {k: v for k, v in C_SM.items() if k in SM_keys}

        # set the Wilson coefficients at the EW scale to C_out
          ##First defining C_in_sm - a dictionary with all SMpar and WCs with zeros
        C_in_sm = eftutil[self.eft].C_array2dict(np.zeros(9999))
        C_in_sm.update(C_out) 
          ##Then we update the value of SM parameters which we extracted above
        C_in_sm.update(C_SM)

        # initialize an empty EFT instance
        eft_sm = EFTevolve(None, self.beta)
        # update C_in in eft_sm
        eft_sm.C_in = C_in_sm 
        eft_sm.scale_in = scale_sm
        eft_sm.eft = self.eft 

        # run up (with 1% relative precision, ignore running of Wilson coefficients)
        # putting newphys false makes beta function of new physics WC zero, but it doesnt 
        # change the beta functions of SM. Therefore, the SM function evolves as usual -
        # and it evolves using the C_out. newphys = False doesn't make C_out zero, but it 
        # makes its corresponding beta function zero. 
        C_SM_high = eft_sm.eft_evolve(self.scale_in, newphys=False, rtol=0.001, atol=1)
        C_SM_high = self._rotate_defaultbasis(C_SM_high) 
        return {k: v for k, v in C_SM_high.items() if k in SM_keys}

    def _get_sm_scale_in(self, scale_sm=91.1876):
        """Get an estimate of the SM parameters at the input scale by running
        them from the EW scale using constant values for the Wilson coefficients
        (corresponding to their leading log approximated values at the EW
        scale).

        Note that this is not guaranteed to work and will fail if some of the
        Wilson coefficients (the ones affecting the extraction of SM parameters)
        are large."""
        # intialize a copy of ourselves without SM parameters
        _smeft = EFTevolve(self.wc, self.beta, get_smpar=False)
        # Step 1: run the SM up, using the WCs at scale_input as (constant) estimate
        _smeft.C_in.update(self._run_sm_scale_in(self.C_in, scale_sm=scale_sm))
        # Step 2: run the WCs down in LL approximation
        C_out = _smeft.eft_evolve_leadinglog(scale_sm)
        # Step 3: run the SM up again, this time using the WCs at scale_sm as (constant) estimate
        return self._run_sm_scale_in(C_out, scale_sm=scale_sm)
  
    
            


#    @staticmethod
    def _rotate_defaultbasis(self, C):
        """Rotate all parameters to the basis where the running down-type quark
        and charged lepton mass matrices are diagonal and where the running
        up-type quark mass matrix has the form V.S, with V unitary and S real
        diagonal, and where the CKM and PMNS matrices have the standard
        phase convention."""
        "Equation 3.27 1704.03888"
        "Need to implement neutrino sector" #FIXME
        v = 246.22
        Mep = v/sqrt(2) * (C['Ge'] - C['ephi'] * v**2/2)
        Mup = v/sqrt(2) * (C['Gu'] - C['uphi'] * v**2/2)
        Mdp = v/sqrt(2) * (C['Gd'] - C['dphi'] * v**2/2)
        Mnup = -v**2 * C['llphiphi']
        UeL, Me, UeR = ckmutil.diag.msvd(Mep)
        UuL, Mu, UuR = ckmutil.diag.msvd(Mup)
        UdL, Md, UdR = ckmutil.diag.msvd(Mdp)
        Unu, Mnu = ckmutil.diag.mtakfac(Mnup)
        UuL, UdL, UuR, UdR = ckmutil.phases.rephase_standard(UuL, UdL, UuR, UdR)
        Unu, UeL, UeR = ckmutil.phases.rephase_pmns_standard(Unu, UeL, UeR)
        if self.eft == 'SMEFT':
            C_out = definitions.flavor_rotation_smeft(C, Uq=UdL, Uu=UuR, Ud=UdR, Ul=UeL, Ue=UeR)
        elif self.eft == 'nuSMEFT': 
            C_out = definitions.flavor_rotation_nusmeft(C, Uq=UdL, Uu=UuR, Ud=UdR, Ul=UeL, Ue=UeR, Un=np.eye(3))
        return C_out

    # If we don't input scale_in or input wilson coefficient, the wcxf.WC function will automatically 
    # not run, so this _check_initial function seems useles. 
    # Sill will see later if its used. I am just thinking, if the original author put it, 
    # so mt must have some use

    def get_smpar(self, accuracy='integrate', scale_sm=91.1876):
        """Compute the SM MS-bar parameters at the electroweak scale.

        This method can be used to validate the accuracy of the iterative
        extraction of SM parameters. If successful, the values returned by this
        method should agree with the values in the dictionary
        `wilson.run.smeft.smpar.p`."""
        if accuracy == 'integrate':
            C_out = self.eft_evolve(scale_sm)
        elif accuracy == 'leadinglog':
            C_out = self.eft_evolve_leadinglog(scale_sm)
        else:
            raise ValueError(f"'{accuracy}' is not a valid value of 'accuracy' (must be either 'integrate' or 'leadinglog').")
        return smpar.smpar(C_out)

    def eft_evolve_continuous(self, C_in, scale_in, scale_out, newphys=True, **kwargs):
        """Solve the SMEFT RGEs by numeric integration, returning a function that
        allows to compute an interpolated solution at arbitrary intermediate
        scales."""
        sol = self._eft_evolve(scale_out, newphys=newphys, dense_output=True, **kwargs)
        @np.vectorize
        def _rge_solution(scale):
            t = log(scale)
            y = sol.sol(t).view(complex)
            yd = eftutil[self.eft].C_array2dict(y)
            yw = eftutil[self.eft].arrays2wcxf_nonred(yd)
            return yw
        def rge_solution(scale):
            # this is to return a scalar if the input is scalar
            return _rge_solution(scale)[()]
        return rge_solution

    def run_continuous(self, scale):
        """Return a continuous solution to the RGE as `RGsolution` instance."""
        if scale == self.scale_in:
            raise ValueError("The scale must be different from the input scale")
        elif scale < self.scale_in:
            scale_min = scale
            scale_max = self.scale_in
        elif scale > self.scale_in:
            scale_max = scale
            scale_min = self.scale_in
        fun = self.eft_evolve_continuous(C_in=self.C_in,
                                          scale_in=self.scale_in,
                                          scale_out=scale)
        return wilson.classes.RGsolution(fun, scale_min, scale_max)

    def _check_initial(self):
        """Check if initial values and scale as well as the new physics scale
        have been set."""
        if self.C_in is None:
            raise Exception("You have to specify the initial conditions first.")
        if self.scale_in is None:
            raise Exception("You have to specify the initial scale first.")

