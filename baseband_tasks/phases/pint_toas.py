# Licensed under the GPLv3 - see LICENSE
"""Utilities for converting times to PINT times of arrival.

See https://github.com/nanograv/PINT
"""

import numpy as np


__all__ = ['PintToas']


class PintToas:
    """Convert time samples to PINT TOAs using given ephemeris, etc.

    The class is initialized with parameters common to all arrival times.
    When the instances is called on a list of times, it uses
    `pint.toa.get_TOAs_list` to create a `pint.toa.TOAs` instance
    (for which in turn phases can be calculated).

    Parameters
    ----------
    observatory : str
        The observatory code or names
    frequency : `~astropy.units.Quantity`.
        Observing frequency.  If not a scalar, one has to ensure it can be
        broadcast properly against time arrays for which lists of TOAs are
        calculated.
    ephemeris : str, optinal
        Solar system dynamic model file. Default is astropy's 'jpl'
        (see `~astropy.coordinates.solar_system_ephemeris`).  For consistency
        with PINT, this argument can also be passed in as ``ephem``.
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
        Any further arguments to be passed on to `pint.toa.get_TOAs_list`.

    Notes
    -----
    A TOA (time of arrival) represents the pulse time of arrival.
    Combined with metadata, it can be considered a timestamp
    (e.g., observatory, observing frequency, etc.)
    """

    def __init__(self, observatory, frequency, *,
                 ephemeris='jpl', include_bipm=True, bipm_version='BIPM2015',
                 include_gps=True, planets=False, tdb_method="default",
                 **kwargs):
        self.observatory = observatory
        self.frequency = frequency
        self.control_params = {'ephem': ephemeris,
                               'bipm_version': bipm_version,
                               'include_bipm': include_bipm,
                               'bipm_version': bipm_version,
                               'include_gps': include_gps,
                               'planets': planets,
                               'tdb_method': tdb_method}
        self.control_params.update(kwargs)

    def __call__(self, time):
        """Create list of TOAs for one or more times.

        Parameters
        ----------
        time : `~astropy.time.Time`
            Input time stamps.

        Returns
        -------
        toas : `~pint.toa.TOAs`
            Combining all TOAs.
        """
        # local import since we cannot count on PINT being present,
        # and doing it globally messes up sphinx.
        from pint import toa

        if time.scale == 'utc':
            time = time.replicate(format='pulsar_mjd')

        freq, _ = np.broadcast_arrays(self.frequency, time.jd1, subok=True)
        time = time._apply(np.broadcast_to, freq.shape)
        toa_list = []
        for t, f in zip(time.ravel(), freq.ravel()):
            # This format converting should be done by PINT in the future.
            toa_entry = toa.TOA(t, obs=self.observatory, freq=f)
            toa_list.append(toa_entry)

        toas = toa.get_TOAs_list(toa_list, **self.control_params)
        toas.shape = time.shape
        return toas
