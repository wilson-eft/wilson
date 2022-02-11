"""Defines the SMEFT class that provides the main API to smeft."""

from . import rge
from . import definitions
from . import smpar
from math import sqrt
import numpy as np
import ckmutil.phases, ckmutil.diag
import wilson
from wilson.util import smeftutil
from wilson import wcxf
from . import rgeleft
from wilson.util import leftutil 


class EFT:
   """Class representing a parameter point in the Effective
   Field Theory.

   Methods:
   - __init__: Initialize, given a wcxf.WC instance
   """
   def __init__(self, wc):
       """Initialize the EFT instance.
       Parameters:  'wc' : Wilson coefficients as `wcxf.WC` instance.
       """
       self.wc=wc

class LEFT(EFT):
    """Class representing a parameter point in the Low Energy Effective
    Field Theory and allowing the evolution of the Wilson Coefficients.

    Methods:

    - __init__: Initialize, given a wcxf.WC instance
    - run: solve the RGE and return a wcxf.WC instance
    """

    def __init__(self, wc, get_smpar=True):
       super().__init__(wc)
       self.eft= wc.eft
       self.wc = wc
       self.scale_in = wc.scale
       self.C_in = None

       C = wilson.util.leftutil.wcxf2arrays_symmetrized(wc.dict)

       for k, s in leftutil.C_keys_shape.items():
           if k not in C and k not in leftutil.SM_keys:
               if s == 1:
                   C[k] = 0
               else:
                   C[k] = np.zeros(s)
       if self.C_in is None:
           self.C_in = C
       else:
           self.C_in.update(C)
    def _leftevolve_leadinglog(self, scale_out):
        """Compute the leading logarithmic approximation to the solution
        of the LEFT RGEs from the initial scale to `scale_out`.
        Returns a dictionary with parameters and Wilson coefficients.
        """
        return rgeleft.left_evolve_leadinglog(C_in=self.C_in,
                            scale_in=self.scale_in,
                            scale_out=scale_out)

    def _to_wcxf(self, C_out, scale_out):
        """Return the Wilson coefficients `C_out` as a wcxf.WC instance.
        """
        C = C_out
        d = wilson.util.leftutil.arrays2wcxf(C)
        d = wcxf.WC.dict2values(d)
        wc = wcxf.WC('WET', 'JMS', scale_out, d)
        return wc

    def run(self, scale):
        """Return the Wilson coefficients  (as wcxf.WC instance) evolved to the
        scale `scale`.
        Parameters:
        - `scale`: scale in GeV
        """
        C_out = self._leftevolve_leadinglog(scale)
        return self._to_wcxf(C_out, scale)


class SMEFT(EFT):
    """Class representing a parameter point in the Standard Model Effective
    Field Theory and allowing the evolution of the Wilson Coefficients.

    Methods:

    - __init__: Initialize, given a wcxf.WC instance
    - run: solve the RGE and return a wcxf.WC instance
    """
    def __init__(self, wc, get_smpar=True):
        super().__init__(wc)
        self.scale_in=None
        self.C_in= None

        if wc is not None:
            self._set_initial_wcxf(wc, get_smpar=get_smpar)

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
        if wc.eft != 'SMEFT':
            raise ValueError("Wilson coefficients use wrong EFT.")
        if wc.basis != 'Warsaw':
            raise ValueError("Wilson coefficients use wrong basis.")
        self.scale_in = wc.scale
        C = wilson.util.smeftutil.wcxf2arrays_symmetrized(wc.dict)
        # fill in zeros for missing WCs
        for k, s in smeftutil.C_keys_shape.items():
            if k not in C and k not in smeftutil.dim4_keys:
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
        d = wilson.util.smeftutil.arrays2wcxf_nonred(C)
        d = wcxf.WC.dict2values(d)
        wc = wcxf.WC('SMEFT', 'Warsaw', scale_out, d)
        return wc

    def _rgevolve(self, scale_out, **kwargs):
        """Solve the SMEFT RGEs from the initial scale to `scale_out`.
        Returns a dictionary with parameters and Wilson coefficients at
        `scale_out`. Additional keyword arguments will be passed to
        the ODE solver `scipy.integrate.odeint`."""
        self._check_initial()
        return rge.smeft_evolve(C_in=self.C_in,
                            scale_in=self.scale_in,
                            scale_out=scale_out,
                            **kwargs)

    def _rgevolve_leadinglog(self, scale_out):
        """Compute the leading logarithmic approximation to the solution
        of the SMEFT RGEs from the initial scale to `scale_out`.
        Returns a dictionary with parameters and Wilson coefficients.
        Much faster but less precise that `rgevolve`.
        """
        self._check_initial()
        return rge.smeft_evolve_leadinglog(C_in=self.C_in,
                            scale_in=self.scale_in,
                            scale_out=scale_out)

    def _check_initial(self):
        """Check if initial values and scale as well as the new physics scale
        have been set."""
        if self.C_in is None:
            raise Exception("You have to specify the initial conditions first.")
        if self.scale_in is None:
            raise Exception("You have to specify the initial scale first.")

    def _rotate_defaultbasis(self, C):
        """Rotate all parameters to the basis where the running down-type quark
        and charged lepton mass matrices are diagonal and where the running
        up-type quark mass matrix has the form V.S, with V unitary and S real
        diagonal, and where the CKM and PMNS matrices have the standard
        phase convention."""
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
        return definitions.flavor_rotation(C, Uq=UdL, Uu=UuR, Ud=UdR, Ul=UeL, Ue=UeR)

    def _run_sm_scale_in(self, C_out, scale_sm=91.1876):
        """Get the SM parameters at the EW scale, using an estimate `C_out`
        of the Wilson coefficients at that scale, and run them to the
        input scale."""
        # initialize an empty SMEFT instance
        smeft_sm = SMEFT(wc=None)
        C_in_sm = smeftutil.C_array2dict(np.zeros(9999))
        # set the SM parameters to the values obtained from smpar.smeftpar
        C_SM = smpar.smeftpar(scale_sm, C_out, basis='Warsaw')
        SM_keys = set(smeftutil.dim4_keys)  # to speed up lookup
        C_SM = {k: v for k, v in C_SM.items() if k in SM_keys}
        # set the Wilson coefficients at the EW scale to C_out
        C_in_sm.update(C_out)
        C_in_sm.update(C_SM)
        smeft_sm._set_initial(C_in_sm, scale_sm)
        # run up (with 1% relative precision, ignore running of Wilson coefficients)
        C_SM_high = smeft_sm._rgevolve(self.scale_in, newphys=False, rtol=0.001, atol=1)
        C_SM_high = self._rotate_defaultbasis(C_SM_high)
        return {k: v for k, v in C_SM_high.items() if k in SM_keys}

    def get_smpar(self, accuracy='integrate', scale_sm=91.1876):
        """Compute the SM MS-bar parameters at the electroweak scale.

        This method can be used to validate the accuracy of the iterative
        extraction of SM parameters. If successful, the values returned by this
        method should agree with the values in the dictionary
        `wilson.run.smeft.smpar.p`."""
        if accuracy == 'integrate':
            C_out = self._rgevolve(scale_sm)
        elif accuracy == 'leadinglog':
            C_out = self._rgevolve_leadinglog(scale_sm)
        else:
            raise ValueError(f"'{accuracy}' is not a valid value of 'accuracy' (must be either 'integrate' or 'leadinglog').")
        return smpar.smpar(C_out)

    def _get_sm_scale_in(self, scale_sm=91.1876):
        """Get an estimate of the SM parameters at the input scale by running
        them from the EW scale using constant values for the Wilson coefficients
        (corresponding to their leading log approximated values at the EW
        scale).

        Note that this is not guaranteed to work and will fail if some of the
        Wilson coefficients (the ones affecting the extraction of SM parameters)
        are large."""
        # intialize a copy of ourselves
        _smeft = SMEFT(self.wc, get_smpar=False)
        # Step 1: run the SM up, using the WCs at scale_input as (constant) estimate
        _smeft.C_in.update(self._run_sm_scale_in(self.C_in, scale_sm=scale_sm))
        # Step 2: run the WCs down in LL approximation
        C_out = _smeft._rgevolve_leadinglog(scale_sm)
        # Step 3: run the SM up again, this time using the WCs at scale_sm as (constant) estimate
        return self._run_sm_scale_in(C_out, scale_sm=scale_sm)

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
            C_out = self._rgevolve(scale, **kwargs)
        elif accuracy == 'leadinglog':
            C_out = self._rgevolve_leadinglog(scale)
        else:
            raise ValueError(f"'{accuracy}' is not a valid value of 'accuracy' (must be either 'integrate' or 'leadinglog').")
        return self._to_wcxf(C_out, scale)

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
        fun = rge.smeft_evolve_continuous(C_in=self.C_in,
                                          scale_in=self.scale_in,
                                          scale_out=scale)
        return wilson.classes.RGsolution(fun, scale_min, scale_max)
