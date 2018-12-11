# Licensed under the GPLv3 - see LICENSE

import warnings

import numpy as np
import astropy.units as u
import astropy.time as time
import pint.toa as toa
import pint.models as model
from pint import pulsar_mjd


__all__ = ['PintUtils', 'make_toa_list']


class PintUtils(object):
    """This is an utility class to compute the pulse phase at given time using
       PINT (https://github.com/nanograv/PINT).

       Parameter
       ---------
       par_file : str
           Tempo style parameter file
       tim_file : str, optional
           Tempo style TOA file. Default is None.
       times : `~astropy.time.Time` or list of `~astropy.time.Time`, optional
           Input time stemps. Default is None. If 'times' is provided, both
           'obs' and 'obs_freq' have to be provided.
       obs : str, optinal
           Observatory name or code, Default is None
       obs_freq : float or `~astropy.units.Quantity`,
           Observering frequency, Default is None
       solar_ephem : str, optional
           Solar system ephemeris version. Default is 'de436'
       bipm_version : str, optional
           BIPM clock correction version. Default is 'BIPM2017'.

       Note
       ----
       This class accepts two type of input for time, the Tempo/Tempo2 style
       .tim and the `~astropy.time.Time` or a list of it. User can choose either
       way of input, but have to choose one.


       Raises
       ------
       RunTimeError
           If the time stemps are not provided.
       RunTimeError
           If the observatory name or observing frequency are not provided,
           when using time stemps.
    """
    def __init__(self, par_file):
        # Load model file
        self.model = model.get_model(par_file)
        self.toas = None

    def get_toas(self, tim_file=None, time=None, obs=None, obs_freq=None,
                 solar_ephem='de436', bipm_version='BIPM2015', **kwargs):
        # Load toas from tim file.
        if tim_file is not None:
            self.toas = toa.get_TOAs(tim_file, ephem, bipm_version, **kwargs)
        # Load from time stemps.
        elif time is not None:
            if obs is None:
                raise RuntimeError('Observatory name can not be None.')
            if obs_freq is None:
                raise RuntimeError('Observing frequency can not be None.')

            self.toas = toa.get_TOAs_list(make_toa_list(time, obs, obs_freq,
                                                        **kwargs),
                                          ephem=solar_ephem,
                                          bipm_version=bipm_version)
        else:
            raise RuntimeError("PINT utilities requires the stemps. Please "
                               "input them from 'tim_file' or 'times'.")

    def compute_phase(self):
        return self.model.phase(self.toas)

    def compute_delay(self):
        return self.model.delay(self.toas)

def make_toa_list(t, obs, obs_freq, **other_meta):
    """ This is a helper function to convert the time stemps to TOA format.

    Parameters
    ----------
    t : `~astropy.time.Time`
        Input time stemps
    obs : str
        The observatory code or names
    obs_freq : float or `~astropy.units.Quantity`
        The observing frequency, the default unit is MHz

    """
    t = t._apply(np.atleast_1d)
    toa_list = []
    obs_freq = u.Quantity(obs_freq, u.MHz)
    for t_stemp in t:
        # This format converting should be done by PINT in the futureself.
        if t_stemp.scale == 'utc':
            t_stemp = time.Time(t_stemp, format='pulsar_mjd')
        toa_entry = toa.TOA(t_stemp, obs=obs, freq=obs_freq, **other_meta)
        toa_list.append(toa_entry)
    return toa_list
