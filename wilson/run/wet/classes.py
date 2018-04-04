"""Defines the `WET` class that provides the main interface to the run.wet
package."""


import wcxf
import wilson
from wilson.util import qcd
from wilson.run.wet import rge, definitions
from wilson.run.wet.parameters import p as default_parameters
from collections import OrderedDict
import numpy as np


class WETrunner(object):
    """Class representing a point in Wilson coefficient space.

    Methods:

    - run: Evolve the Wilson coefficients to a different scale
    """

    def __init__(self, wc, parameters=None):
        """Initialize the instance.

        Parameters:

        - wc: instance of `wcxf.WC` representing Wilson coefficient values
          at a given (input) scale. The EFT must be one of `WET`, `WET-4`,
          or `WET-3`; the basis must be `JMS`.
        - parameters: optional. If provided, must be a dictionary containing
          values for the input parameters as defined in `run.wet.parameters`.
          Default values are used for all parameters not provided.
        """
        assert isinstance(wc, wcxf.WC)
        assert wc.basis == 'JMS', \
            "Wilson coefficients must be given in the 'JMS' basis"
        self.eft = wc.eft
        # number of quark flavours
        if self.eft == 'WET':
            self.f = 5
        elif self.eft == 'WET-4':
            self.f = 4
        elif self.eft == 'WET-3':
            self.f = 3
        self.scale_in = wc.scale
        self.C_in = wc.dict
        self.parameters = default_parameters
        if parameters is not None:
            self.parameters.update(parameters)

    def _get_running_parameters(self, scale, f, loop=3):
        """Get the running parameters (e.g. quark masses and the strong
        coupling at a given scale."""
        p = {}
        p['alpha_s'] = qcd.alpha_s(scale, self.f, self.parameters['alpha_s'], loop=loop)
        p['m_b'] = qcd.m_b(self.parameters['m_b'], scale, self.f, self.parameters['alpha_s'], loop=loop)
        p['m_c'] = qcd.m_c(self.parameters['m_c'], scale, self.f, self.parameters['alpha_s'], loop=loop)
        p['m_s'] = qcd.m_s(self.parameters['m_s'], scale, self.f, self.parameters['alpha_s'], loop=loop)
        p['m_u'] = qcd.m_s(self.parameters['m_u'], scale, self.f, self.parameters['alpha_s'], loop=loop)
        p['m_d'] = qcd.m_s(self.parameters['m_d'], scale, self.f, self.parameters['alpha_s'], loop=loop)
        # running ignored for alpha_e and lepton mass
        p['alpha_e'] = self.parameters['alpha_e']
        p['m_e'] = self.parameters['m_e']
        p['m_mu'] = self.parameters['m_mu']
        p['m_tau'] = self.parameters['m_tau']
        return p

    def _run_dict(self, scale_out, sectors='all'):
        p_i = self._get_running_parameters(self.scale_in, self.f)
        p_o = self._get_running_parameters(scale_out, self.f)
        Etas = (p_i['alpha_s'] / p_o['alpha_s'])
        C_out = OrderedDict()
        for sector in wcxf.EFT[self.eft].sectors:
            if sector in definitions.sectors:
                if sectors == 'all' or sector in sectors:
                    C_out.update(rge.run_sector(sector, self.C_in,
                                                Etas, self.f, p_i, p_o))
        return C_out

    def run(self, scale_out, sectors='all'):
        """Evolve the Wilson coefficients to the scale `scale_out`.

        Parameters:

        - scale_out: output scale
        - sectors: optional. If provided, must be a tuple of strings
          corresponding to WCxf sector names. Only Wilson coefficients
          belonging to these sectors will be present in the output.

        Returns an instance of `wcxf.WC`.
        """
        C_out = self._run_dict(scale_out, sectors=sectors)
        all_wcs = set(wcxf.Basis[self.eft, 'JMS'].all_wcs)  # to speed up lookup
        C_out = {k: v for k, v in C_out.items()
                 if v != 0 and k in all_wcs}
        return wcxf.WC(eft=self.eft, basis='JMS',
                       scale=scale_out,
                       values=wcxf.WC.dict2values(C_out))

    def run_continuous(self, scale, sectors='all'):
        if scale == self.scale_in:
            raise ValueError("The scale must be different from the input scale")
        elif scale < self.scale_in:
            scale_min = scale
            scale_max = self.scale_in
        elif scale > self.scale_in:
            scale_max = scale
            scale_min = self.scale_in
        @np.vectorize
        def _f(scale):
            return self._run_dict(scale, sectors=sectors)
        def f(scale):
            # this is to return a scalar if the input is scalar
            return _f(scale)[()]
        return wilson.classes.RGsolution(f, scale_min, scale_max)
