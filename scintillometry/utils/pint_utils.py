# Licensed under the GPLv3 - see LICENSE
"""The utility classes and functions for PINT
   (https://github.com/nanograv/PINT)
"""
import warnings

import numpy as np
import astropy.units as u
import astropy.time as time
import pint.toa as toa
import pint.models as model
from pint import pulsar_mjd


__all__ = ['PintToas', 'make_toa_list']


class PintToas(object):
    """This is an utility class to make convert time samples or .tim file to a
       PINT TOAs object.

       Parameter
       ---------
       ephem : str, optinal
           Solar system dynamic model file. Default is JPL 'de436'.
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

       Note
       ----
       A TOA (time of arrival) initial represents the pulse time of arrival.
       However, it can be considered as timestamps with some necessary metadata
       (e.g., observatory, observing frequency, etc.)
    """
    def __init__(self, ephem='de436', include_bipm=True,
                 bipm_version='BIPM2015', include_gps=True, planets=False,
                 tdb_method="default", **kwargs):
        self.control_params = {'ephem': ephem, 'bipm_version': bipm_version,
                               'include_bipm': include_bipm,
                               'bipm_version': bipm_version,
                               'include_gps': include_gps, 'planets': planets,
                               'tdb_method': tdb_method}
        self.control_params.update(kwargs)

    def make_toas(self, toa_list):
        """Read TOAs from a list the of timestamps.

           Parameter
           ---------
           toa_list : list of `~pint.toa.TOA` objects,
               Input time stamps as a list of TOA objects.
        """
        return toa.get_TOAs_list(toa_list, **self.control_params)


# NOTE, the functions below will be included in the future PINT release.
def make_toa_list(t, obs, frequency, **other_meta):
    """ This is a helper function to convert the timestamps to TOA format.

    Parameters
    ----------
    t : `~astropy.time.Time`
        Input timestamps
    obs : str
        The observatory code or names
    frequency : float or `~astropy.units.Quantity`
        The observing frequency, the default unit is MHz

    Return
    ------
    List of `~pint.toa.TOA` object.
    """
    t = t._apply(np.atleast_1d)
    toa_list = []
    frequency = u.Quantity(frequency, u.MHz)
    for t_stamp in t:
        # This format converting should be done by PINT in the futureself.
        if t_stamp.scale == 'utc':
            t_stamp = time.Time(t_stamp, format='pulsar_mjd')
        toa_entry = toa.TOA(t_stamp, obs=obs, freq=frequency, **other_meta)
        toa_list.append(toa_entry)
    return toa_list
