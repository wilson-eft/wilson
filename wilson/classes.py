from wilson.run.smeft import SMEFT
from wilson.run.wet import WETrunner


class Wilson(object):
    """Wilson."""
    def __init__(self, wc):
        self.wc = wc
        self._cache = {}

    def _repr_markdown_(self):
        r_wcxf = self.wc._repr_markdown_()
        r_wcxf = '\n'.join(r_wcxf.splitlines()[2:])  # remove WCxf heading
        md = "## `wilson` coefficients\n\n"
        md += "### Initial values:\n\n"
        md += r_wcxf
        return md

    def match_run(self, eft, basis, scale, sectors='all'):
        """Run the Wilson coefficients to a different scale
        (and possibly different EFT)
        and return them as `wcxf.WC` instance."""
        scale_ew = 91.1876
        mb = 4.2
        mc = 1.3
        if self.wc.basis == basis and self.wc.eft == eft and scale == self.wc.scale:
            return self.wc  # nothing to do
        if self.wc.eft == 'SMEFT':
            smeft = SMEFT(self.wc)
            if eft == 'SMEFT':
                # if input and output EFT ist SMEFT, just run.
                return smeft.run(scale)
            else:
                # if SMEFT -> WET-x: match to WET at the EW scale
                wc_ew = smeft.run(scale_ew).match('WET', 'Bern')
                wet = WETrunner(wc_ew)
        elif self.wc.eft in ['WET', 'WET-4', 'WET-3']:
            wet = WETrunner(self.wc.translate('Bern'))
        else:
            raise ValueError("Input EFT {} unknown or not supported".format(self.wc.eft))
        if eft == wet.eft:  # just run
            wc_out = wet.run(scale, sectors=sectors).translate(basis)
            self._set_cache(sectors, scale, eft, basis, wc_out)
            return wc_out
        elif eft == 'WET-4' and wet.eft == 'WET':  # match at mb
            wc_mb = wet.run(mb, sectors=sectors).match('WET-4', 'Bern')
            wet4 = WETrunner(wc_mb)
            wc_out = wet4.run(scale, sectors=sectors).translate(basis)
            self._set_cache(sectors, scale, 'WET-4', basis, wc_out)
            return wc_out
        elif eft == 'WET-3' and wet.eft == 'WET-4':  # match at mc
            wc_mc = wet.run(mc, sectors=sectors).match('WET-3', 'Bern')
            wet3 = WETrunner(wc_mc)
            wc_out = wet3.run(scale, sectors=sectors).translate(basis)
            return wc_out
            self._set_cache(sectors, scale, 'WET-3', basis, wc_out)
        elif eft == 'WET-3' and wet.eft == 'WET':  # match at mb and mc
            wc_mb = wet.run(mb, sectors=sectors).match('WET-4', 'Bern')
            wet4 = WETrunner(wc_mb)
            wc_mc = wet4.run(mc, sectors=sectors).match('WET-3', 'Bern')
            wet3 = WETrunner(wc_mc)
            wc_out = wet3.run(scale, sectors=sectors).translate(basis)
            self._set_cache(sectors, scale, 'WET-3', basis, wc_out)
            return wc_out
        else:
            raise ValueError("Running from {} to {} not implemented".format(wet.eft, eft))

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
