"""Main classes used at the top level of the wilson package:

`Wilson`: main interface to the wilson package, providing automatic running
and matching in SMEFT and WET

`RGsolution`: Class representing a continuous solution to the
SMEFT and WET RGEs to be used for plotting.
"""


from wilson.run.smeft import SMEFT
from wilson.run.wet import WETrunner
import numpy as np
from math import log, e
import wcxf
import voluptuous as vol

class ConfigurableClass(object):
    """Class that provides the functionality to set and get configuration
    options.

    Methods:

    - `set_option`: Set configuration option
    - `get_option`: Show configuration option
    - `set_default_option`: Class method! Set deault configuration option
      affecting only future instances of the class.
    """

    # default config options:
    # dictionary with option name as 'key' and default option value as 'value'
    _default_options = {}

    # option schema:
    # Voluptuous schema defining allowed option values/types
    _option_schema = vol.Schema({})

    def __init__(self):
        self._options = self._default_options.copy()

    @classmethod
    def set_default_option(cls, key, value):
        """Class method. Set the default value of the option `key` (string)
        to `value` for all future instances of the class.

        Note that this does not affect existing instances or the instance
        called from."""
        cls._default_options.update(cls._option_schema({key: value}))

    def set_option(self, key, value):
        """Set the option `key` (string) to `value`.

        Instance method, affects only current instance.
        This will clear the cache."""
        self._options.update(self._option_schema({key: value}))
        self.clear_cache()

    def get_option(self, key):
        """Return the current value of the option `key` (string).

        Instance method, only refers to current instance."""
        return self._options.get(key, self._default_options[key])


class Wilson(ConfigurableClass):
    """Main interface to the wilson package, providing automatic running
    and matching in SMEFT and WET.

    Caching is used for intermediate results.

    Methods:

    - `from_wc`: Return a `Wilson` instance initialized by a `wcxf.WC` instance
    - `load_wc`: Return a `Wilson` instance initialized by a WCxf file-like object
    - `match_run`: Run the Wilson coefficients to a different scale (and possibly different EFT) and return them as `wcxf.WC` instance
    - `set_option`: Set configuration option
    - `get_option`: Show configuration option
    - `set_default_option`: Class method! Set deault configuration option
      affecting only future instances of the class.
    """

    # default config options:
    # dictionary with option name as 'key' and default option value as 'value'
    _default_options = {'smeft_accuracy': 'integrate',
                        'qed_order': 1,
                        'qcd_order': 1,
                        'smeft_matchingscale': 91.1876,
                        'mb_matchingscale': 4.2,
                        'mc_matchingscale': 1.3,
                        }

    # option schema:
    # Voluptuous schema defining allowed option values/types
    _option_schema = vol.Schema({
        'smeft_accuracy': vol.In(['integrate','leadinglog']),
        'qed_order': vol.In([0,1]),
        'qcd_order': vol.In([0,1]),
        'smeft_matchingscale': vol.Coerce(float),
        'mb_matchingscale': vol.Coerce(float),
        'mc_matchingscale': vol.Coerce(float),
    })

    def __init__(self, wcdict, scale, eft, basis):
        """Initialize the `Wilson` class.

        Parameters:

        - `wcdict`: dictionary of Wilson coefficient values at the input scale.
          The keys must exist as Wilson coefficients in the WCxf basis file.
          The values must be real or complex numbers (not dictionaries with key
          'Re'/'Im'!)
        - `scale`: input scale in GeV
        - `eft`: input EFT
        - `basis`: input basis
        """
        super().__init__()
        self.wc = wcxf.WC(eft=eft, basis=basis, scale=scale,
                          values=wcxf.WC.dict2values(wcdict))
        self.wc.validate()
        self._cache = {}

    @classmethod
    def from_wc(cls, wc):
        """Return a `Wilson` instance initialized by a `wcxf.WC` instance"""
        return cls(wcdict=wc.dict, scale=wc.scale, eft=wc.eft, basis=wc.basis)

    @classmethod
    def load_wc(cls, stream):
        """Return a `Wilson` instance initialized by a WCxf file-like object"""
        wc = wcxf.WC.load(stream)
        return cls.from_wc(wc)

    def _repr_html_(self):
        r_wcxf = self.wc._repr_html_()
        r_wcxf = '\n'.join(r_wcxf.splitlines()[2:])  # remove WCxf heading
        html = "<h3>Wilson coefficients</h3>\n\n"
        html += r_wcxf
        return html

    def _wetrun_opt(self):
        """Return a dictionary of options to pass to a `run.wet.WETrunner`
        instance."""
        return {'qed_order': self.get_option('qed_order'),
                'qcd_order': self.get_option('qcd_order')}

    def match_run(self, scale, eft, basis, sectors='all'):
        """Run the Wilson coefficients to a different scale
        (and possibly different EFT)
        and return them as `wcxf.WC` instance.

        Parameters:

        - `scale`: output scale in GeV
        - `eft`: output EFT
        - `basis`: output basis
        - `sectors`: in the case of WET (or WET-4 or WET-3), a tuple of sector
          names can be optionally provided. In this case, only the Wilson coefficients
          from this sector(s) will be returned and all others discareded. This
          can speed up the computation significantly if only a small number of sectors
          is of interest. The sector names are defined in the WCxf basis file.
        """
        cached = self._get_from_cache(sector=sectors, scale=scale, eft=eft, basis=basis)
        if cached is not None:
            return cached
        if sectors  == 'all':
            # the default value for sectors is "None" for translators
            translate_sectors = None
        else:
            translate_sectors = sectors
        scale_ew = self.get_option('smeft_matchingscale')
        mb = self.get_option('mb_matchingscale')
        mc = self.get_option('mc_matchingscale')
        if self.wc.basis == basis and self.wc.eft == eft and scale == self.wc.scale:
            return self.wc  # nothing to do
        if self.wc.eft == 'SMEFT':
            smeft_accuracy = self.get_option('smeft_accuracy')
            if eft == 'SMEFT':
                smeft = SMEFT(self.wc.translate('Warsaw', sectors=translate_sectors))
                # if input and output EFT ist SMEFT, just run.
                wc_out = smeft.run(scale, accuracy=smeft_accuracy)
                self._set_cache('all', scale, 'SMEFT', wc_out.basis, wc_out)
                return wc_out
            else:
                # if SMEFT -> WET-x: match to WET at the EW scale
                wc_ew = self._get_from_cache(sector='all', scale=scale_ew, eft='WET', basis='JMS')
                if wc_ew is None:
                    smeft = SMEFT(self.wc.translate('Warsaw'))
                    if self.wc.scale == scale_ew:
                        wc_ew = self.wc.match('WET', 'JMS')  # no need to run
                    else:
                        wc_ew = smeft.run(scale_ew, accuracy=smeft_accuracy).match('WET', 'JMS')
                self._set_cache('all', scale_ew, wc_ew.eft, wc_ew.basis, wc_ew)
                wet = WETrunner(wc_ew, **self._wetrun_opt())
        elif self.wc.eft in ['WET', 'WET-4', 'WET-3']:
            wet = WETrunner(self.wc.translate('JMS'), **self._wetrun_opt())
        else:
            raise ValueError("Input EFT {} unknown or not supported".format(self.wc.eft))
        if eft == wet.eft:  # just run
            wc_out = wet.run(scale, sectors=sectors).translate(basis, sectors=translate_sectors)
            self._set_cache(sectors, scale, eft, basis, wc_out)
            return wc_out
        elif eft == 'WET-4' and wet.eft == 'WET':  # match at mb
            wc_mb = wet.run(mb, sectors=sectors).match('WET-4', 'JMS')
            wet4 = WETrunner(wc_mb, **self._wetrun_opt())
            wc_out = wet4.run(scale, sectors=sectors).translate(basis, sectors=translate_sectors)
            self._set_cache(sectors, scale, 'WET-4', basis, wc_out)
            return wc_out
        elif eft == 'WET-3' and wet.eft == 'WET-4':  # match at mc
            wc_mc = wet.run(mc, sectors=sectors).match('WET-3', 'JMS')
            wet3 = WETrunner(wc_mc, **self._wetrun_opt())
            wc_out = wet3.run(scale, sectors=sectors).translate(basis, sectors=translate_sectors)
            return wc_out
            self._set_cache(sectors, scale, 'WET-3', basis, wc_out)
        elif eft == 'WET-3' and wet.eft == 'WET':  # match at mb and mc
            wc_mb = wet.run(mb, sectors=sectors).match('WET-4', 'JMS')
            wet4 = WETrunner(wc_mb, **self._wetrun_opt())
            wc_mc = wet4.run(mc, sectors=sectors).match('WET-3', 'JMS')
            wet3 = WETrunner(wc_mc, **self._wetrun_opt())
            wc_out = wet3.run(scale, sectors=sectors).translate(basis, sectors=translate_sectors)
            self._set_cache(sectors, scale, 'WET-3', basis, wc_out)
            return wc_out
        else:
            raise ValueError("Running from {} to {} not implemented".format(wet.eft, eft))

    def clear_cache(self):
        self._cache = {}

    def _get_from_cache(self, sector, scale, eft, basis):
        """Try to load a set of Wilson coefficients from the cache, else return
        None."""
        try:
            return self._cache[eft][scale][basis][sector]
        except KeyError:
            return None

    def _set_cache(self, sector, scale, eft, basis, wc_out):
        if eft not in self._cache:
            self._cache[eft] = {scale: {basis: {sector: wc_out}}}
        elif scale not in self._cache[eft]:
            self._cache[eft][scale] = {basis: {sector: wc_out}}
        elif basis not in self._cache[eft][scale]:
            self._cache[eft][scale][basis] = {sector: wc_out}
        else:
            self._cache[eft][scale][basis][sector] = wc_out


class RGsolution(object):
    """Class representing a continuous (interpolated) solution to the
    SMEFT RGEs to be used for plotting."""

    def __init__(self, fun, scale_min, scale_max):
        """Initialize.

        Parameters:

        - fun: function of the scale that is expected to return a
        dictionary with the RGE solution and to accept vectorized input.
        - scale_min, scale_max: lower and upper boundaries of the scale
        """
        self.fun = fun
        self.scale_min = scale_min
        self.scale_max = scale_max

    def plotdata(self, key, part='re', scale='log', steps=50):
        """Return a tuple of arrays x, y that can be fed to plt.plot,
        where x is the scale in GeV and y is the parameter of interest.

        Parameters:

        - key: dicionary key of the parameter to be plotted (e.g. a WCxf
          coefficient name or a SM parameter like 'g')
        - part: plot the real part 're' (default) or the imaginary part 'im'
        - scale: 'log'; make the x steps logarithmically distributed; for
          'linear', linearly distributed
        - steps: steps in x to take (default: 50)
        """
        if scale == 'log':
            x = np.logspace(log(self.scale_min),
                            log(self.scale_max),
                            steps,
                            base=e)
        elif scale == 'linear':
            x = np.linspace(self.scale_min,
                            self.scale_max,
                            steps)
        y = self.fun(x)
        y = np.array([d[key] for d in y])
        if part == 're':
            return x, y.real
        elif part == 'im':
            return x, y.imag

    def plot(self, key, part='re', scale='log', steps=50, legend=True, plotargs={}):
        """Plot the RG evolution of parameter `key`.

        Parameters:

        - part, scale, steps: see `plotdata`
        - legend: boolean, show the legend (default: True)
        - plotargs: dictionary of arguments to be passed to plt.plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Please install matplotlib if you want to use the plot method")
        pdat = self.plotdata(key, part=part, scale=scale, steps=steps)
        plt.plot(*pdat, label=key, **plotargs)
        if scale == 'log':
            plt.xscale('log')
        if legend:
            plt.legend()
