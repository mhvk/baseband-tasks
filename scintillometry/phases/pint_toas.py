# Licensed under the GPLv3 - see LICENSE
"""Utilities for converting times to PINT times of arrival.

See https://github.com/nanograv/PINT
"""
import warnings

import numpy as np
from astropy import units as u


__all__ = ['PintToas']


class PintToas:
    """Convert time samples to PINT TOAs using given ephemeris, etc.

    Parameters
    ----------
    observatory : str
        The observatory code or names
    frequency : float or `~astropy.units.Quantity`
        The observing frequency, the default unit is MHz
    ephem : str, optinal
        Solar system dynamic model file. Default is astropy's 'jpl'
        (see `~astropy.coordinates.solar_system_ephemeris`).
    include_bipm : bool, optional
        Flag to include the TT BIPM correction. Default is True.
    bipm_version : str, optional
        TT BIPM version. Default is 'BIPM2015'
    include_gps : bool, optional
        Flag to include the gps clock correction. Default is True.
    planets : bool, optional
        Flag to compute the planets' positions and velocities. Default is
        False.
    tdb_method : str, optional
        The method to compute the TDB time scale. Default is using astropy
        time objects' method.
    **kwargs
        Any further arguments to be passed on to ``pint.toa.get_TOAs_list``.

    Notes
    -----
    A TOA (time of arrival) represents the pulse time of arrival.
    Combined with metadata, it can be considered a timestamp
    (e.g., observatory, observing frequency, etc.)
    """
    def __init__(self, observatory, frequency, *,
                 ephem='jpl', include_bipm=True, bipm_version='BIPM2015',
                 include_gps=True, planets=False, tdb_method="default",
                 **kwargs):
        self.observatory = observatory
        self.frequency = frequency
        self.control_params = {'ephem': ephem, 'bipm_version': bipm_version,
                               'include_bipm': include_bipm,
                               'bipm_version': bipm_version,
                               'include_gps': include_gps, 'planets': planets,
                               'tdb_method': tdb_method}
        self.control_params.update(kwargs)

    def __call__(self, time):
        """Create list of TOAs for one or more times.

        Parameters
        ----------
        time : `~astropy.time.Time`
            Input time stamps.
        """
        # local import since we cannot count on PINT being present,
        # and doing it globally messes up sphinx.
        from pint import toa

        toa_list = make_toa_list(time, self.observatory, self.frequency)
        return toa.get_TOAs_list(toa_list, **self.control_params)


# NOTE, the functions below will be included in a future PINT release.
def make_toa_list(time, obs, freq, **other_meta):
    """Convert times to a list of PINT TOAs.

    The input timestamps will be flattened to a 1-dimensional array.

    Parameters
    ----------
    time : `~astropy.time.Time`
        Input timestamps.
    obs : str
        The observatory code or names
    freq : float or `~astropy.units.Quantity`
        The observing frequency, the default unit is MHz

    Returns
    -------
    List of `~pint.toa.TOA`.
    """
    # local import since we cannot count on PINT being present,
    # and doing it globally messes up sphinx.
    from pint import toa

    toa_list = []
    for t_stamp in time.ravel():
        # This format converting should be done by PINT in the future.
        if t_stamp.scale == 'utc':
            t_stamp = t_stamp.replicate(format='pulsar_mjd')
        toa_entry = toa.TOA(t_stamp, obs=obs, freq=freq, **other_meta)
        toa_list.append(toa_entry)
    return toa_list
