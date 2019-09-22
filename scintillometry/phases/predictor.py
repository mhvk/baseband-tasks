# Licensed under the GPLv3 - see LICENSE
r"""Read in and use tempo1 polyco files (tempo2 predict to come).

Examples
--------
>>> psr_polyco = predictor.Polyco('polyco_new.dat')
>>> predicted_phase = psr_polyco(time)

>>> phasepol = psr_polyco.phasepol(Timeindex, rphase='fraction')

For use with folding codes with times since some start time t0 in seconds:

>>> psr_polyco.phasepol(t0, 'fraction', t0=t0, time_unit=u.second, convert=True)

Notes
-----
The format of the polyco files is (from
http://tempo.sourceforge.net/ref_man_sections/tz-polyco.txt)

.. code-block:: text

    Line  Columns Item
    ----  ------- -----------------------------------
    1      1-10   Pulsar Name
          11-19   Date (dd-mmm-yy)
          20-31   UTC (hhmmss.ss)
          32-51   TMID (MJD)
          52-72   DM
          74-79   Doppler shift due to earth motion (10^-4)
          80-86   Log_10 of fit rms residual in periods
    2      1-20   Reference Phase (RPHASE)
          21-38   Reference rotation frequency (F0)
          39-43   Observatory number
          44-49   Data span (minutes)
          50-54   Number of coefficients
          55-75   Observing frequency (MHz)
          76-80   Binary phase
    3-     1-25   Coefficient 1 (COEFF(1))
          26-50   Coefficient 2 (COEFF(2))
          51-75   Coefficient 3 (COEFF(3))

The pulse phase and frequency at time T are then calculated as::

    DT = (T-TMID)*1440
    PHASE = RPHASE + DT*60*F0 + COEFF(1) + DT*COEFF(2) + DT^2*COEFF(3) + ....
    FREQ(Hz) = F0 + (1/60)*(COEFF(2) + 2*DT*COEFF(3) + 3*DT^2*COEFF(4) + ....)

Example tempo2 call to produce one:

.. code-block:: text

    tempo2 -tempo1 \
        -f ~/packages/scintellometry/scintellometry/ephemerides/psrb1957+20.par \
        -polyco "56499 56500 300 12 12 aro 150.0"
                 |-- MJD start
                       |-- MJD end
                             |-- number of minutes for which polynomial is fit
                                 |-- degree of the polynomial
                                    |-- maxium Hour Angle (12 is continuous)
                                       |-- Observatory
                                           |-- Frequency in MHz
"""

from __future__ import division, print_function

from collections import OrderedDict

import numpy as np
from numpy.polynomial import Polynomial
from astropy.table import QTable
import astropy.units as u
from astropy.time import Time

from .phase import Phase


__doctest_skip__ = ['*']
__all__ = ['Polyco']


class Polyco(QTable):
    def __init__(self, *args, **kwargs):
        """Read in polyco file as Table, and set up class."""
        if len(args):
            data = args[0]
            args = args[1:]
        else:
            data = kwargs.pop('data', None)

        if isinstance(data, str):
            data = parse_polyco(data)
            super().__init__(data, *args, **kwargs)
            self['rphase'] = Phase.from_string(self['rphase'])
        else:
            super().__init__(data, *args, **kwargs)

    def to_polyco(self, name='polyco.dat', tempo1=False):
        header_fmt = ('{:<10s} {:9s}{:11.2f}{:20.11f}{:21.6f} {:6.3f}{:7.3f}\n'
                      '{:20.6f}{:18.12f}{:>5s}{:5d}{:5d}{:10.3f}\n')
        if 'binphase' in self.keys():
            header_fmt = header_fmt[:-1] + '{:7.4f}\n'
            if 'forb' in self.keys():
                header_fmt = header_fmt[:-1] + '{:9.4f}\n'

        coeff_fmt = fortran_fmt if tempo1 else '{:24.17e}'.format
        with open(name, 'w') as fh:
            for row in self:
                fh.write(header_fmt.format(*(
                    row[k] for k in converters if k in self.keys())))

                coeff = row['coeff']
                for i in range(0, len(coeff), 3):
                    fh.write(' ' + ' '.join([coeff_fmt(c)
                                             for c in coeff[i:i+3]]) + '\n')

    def __call__(self, time, index=None, rphase=None, deriv=0, time_unit=None):
        """Predict phase or frequency (derivatives) for given mjd (array)

        Parameters
        ----------
        mjd_in : `~astropy.time.Time` or float (array)
            Time instances of MJD's for which phases are to be generated.
            If float, assumed to be MJD (NOTE: less precise!)
        index : int (array), None, float, or `~astropy.time.Time`
            indices into Table for corresponding polyco's; if None, it will be
            deterined from ``mjd_in`` (giving an explicit index can help speed
            up the evaluation).  If not an index or `None`, it will be used to
            find the index. Hence if one has a large array if closely spaced
            times, one can pass in a single element to speed matters up.
        rphase : None, 'fraction', 'ignore', or float (array)
            Phase zero point; if None, use the one stored in polyco
            (those are typically large, so we ensure we preserve precision by
            using the `~scintillometry.phases.Phase` class for the result.)
            Can also set 'fraction' to use the stored one modulo 1, which is
            fine for folding, but breaks cycle count continuity between sets,
            'ignore' for just keeping the value stored in the coefficients,
            or a value that should replace the zero point.
        deriv : int
            Derivative to return (Default=0=phase, 1=frequency, etc.)
        time_unit : Unit
            Unit of time in which derivatives are expressed (Default: second)

        Returns
        -------
        result : `~scintillometry.phases.Phase` or `~astropy.units.Quantity`
            A phase if ``deriv=0`` and ``rphase=None`` to preserve precision;
            otherwise, a quantity with units of ``cycle / time_unit**deriv``.
        """
        time_unit = time_unit or u.s
        if not hasattr(time, 'mjd'):
            time = Time(time, format='mjd', scale='utc')
        try:  # This also catches index=None
            index = index.__index__()
        except (AttributeError, TypeError):
            index = self.searchclosest(time)

        mjd_mid = self['mjd_mid'][index]

        if np.any(np.abs(time.mjd - mjd_mid)*1440 > self['span'][index]/2):
            raise ValueError('(some) MJD outside of polyco range')

        result = np.zeros(time.shape) * u.cycle / time_unit**deriv
        do_phase = (deriv == 0 and rphase is None)
        if do_phase:
            result = Phase(result)
            rphase = 'ignore'

        def do_part(sel, index):
            if do_phase:
                result[sel] += self['rphase'][index]
            polynomial = self.polynomial(index, rphase, deriv)
            dt = (time[sel] - Time(self['mjd_mid'][index], format='mjd', scale='utc'))
            result[sel] += (polynomial(dt.to(u.min).value) * u.cycle / u.min**deriv)

        if time.isscalar:
            do_part(Ellipsis, index)

        else:
            for j in set(index):
                do_part(index == j, j)

        return result

    def polynomial(self, index, rphase=None, deriv=0,
                   t0=None, time_unit=u.min, out_unit=None,
                   convert=False):
        """Prediction polynomial set up for times in MJD

        Parameters
        ----------
        index : int or float
            index into the polyco table (or MJD for finding closest)
        rphase : None or 'fraction' or 'ignore' or float
            Phase zero point; if None, use the one stored in polyco.
            (Those are typically large, so one looses some precision.)
            Can also set 'fraction' to use the stored one modulo 1, which is
            fine for folding, but breaks cycle count continuity between sets,
            'ignore' for just keeping the value stored in the coefficients,
            or a value that should replace the zero point.
        deriv : int
            derivative of phase to take (1=frequency, 2=fdot, etc.); default 0

        Returns
        -------
        polynomial : Polynomial
            set up for MJDs between mjd_mid +/- span

        Notes
        -----
        Units for the polynomial are cycles/second**deriv.  Taking a derivative
        outside will be per day (e.g., self.polynomial(1).deriv() gives
        frequencies in cycles/day)
        """

        out_unit = out_unit or time_unit

        try:
            index = index.__index__()
        except (AttributeError, TypeError):
            index = self.searchclosest(index)
        window = np.array([-1, 1]) * self['span'][index]/2 * u.min

        polynomial = Polynomial(self['coeff'][index],
                                window.value, window.value)
        polynomial.coef[1] += self['f0'][index]*60.

        if deriv == 0:
            if rphase is None:
                polynomial.coef[0] += self['rphase'][index].value
            elif rphase == 'fraction':
                polynomial.coef[0] += self['rphase']['frac'][index].value % 1
            elif rphase != 'ignore':
                polynomial.coef[0] = rphase
        else:
            polynomial = polynomial.deriv(deriv)
            polynomial.coef /= u.min.to(out_unit)**deriv

        if t0 is None:
            dt = 0. * time_unit
        elif not hasattr(t0, 'jd1') and t0 == 0:
            dt = (-self['mjd_mid'][index] * u.day).to(time_unit)
        else:
            dt = ((t0 - Time(self['mjd_mid'][index], format='mjd', scale='utc')
                   ).jd * u.day).to(time_unit)

        polynomial.domain = (window.to(time_unit) - dt).value

        if convert:
            return polynomial.convert()
        else:
            return polynomial

    def phasepol(self, index, rphase=None, t0=0., time_unit=u.day,
                 convert=False):
        """Phase prediction polynomial set up for times in MJD

        Parameters
        ----------
        index : int or float
            index into the polyco table (or MJD for finding closest)
        rphase : None or 'fraction' or float
            phase zero point; if None, use the one stored in polyco.
            (Those are typically large, so one looses some precision.)
            Can also set 'fraction' to use the stored one modulo 1, which is
            fine for folding, but breaks phase continuity between sets.

        Returns
        -------
        phasepol : Polynomial
            set up for MJDs between mjd_mid +/- span
        """
        return self.polynomial(index, rphase, t0=t0, time_unit=time_unit,
                               convert=convert)

    def fpol(self, index, t0=0., time_unit=u.day, convert=False):
        """Frequency prediction polynomial set up for times in MJD

        Parameters
        ----------
        index : int
            index into the polyco table

        Returns
        -------
        freqpol : Polynomial
            set up for MJDs between mjd_mid +/- span
        """
        return self.polynomial(index, deriv=1,
                               t0=t0, time_unit=time_unit, out_unit=u.s,
                               convert=convert)

    def searchclosest(self, mjd):
        """Find index to polyco that is closest in time to (set of) Time/MJD"""
        mjd = getattr(mjd, 'mjd', mjd)
        i = np.clip(np.searchsorted(self['mjd_mid'], mjd), 1, len(self)-1)
        i -= mjd-self['mjd_mid'][i-1] < self['mjd_mid'][i]-mjd
        return i


converters = OrderedDict(
    (('psr', str),
     ('date', lambda x: x.rjust(9)),
     ('utc_mid', float),
     ('mjd_mid', float),
     ('dm', float),
     ('vbyc_earth', float),
     ('lgrms', float),
     ('rphase', str),
     ('f0', float),
     ('obs', str),
     ('span', int),
     ('ncoeff', int),
     ('freq', float),
     ('binphase', float),
     ('forb', float)))


def parse_polyco(name):
    """Parse a tempo1,2 polyco file.

    Parameters
    ----------
    name : string
        file name holding polyco data

    Returns
    -------
    t : list of dict
        each entry in the polyco file corresponds to one row, with a dict
        holding psr, date, utc_mid, mjd_mid, dm, vbyc_earth, lgrms,
        rphase, f0, obs, span, ncoeff, freq, binphase (optional), and
        coeff[ncoeff].
    """

    with open(name, 'r') as polyco:
        line = polyco.readline()
        t = []
        while line != '':
            header = line.split() + polyco.readline().split()
            d = OrderedDict(((key, converter(piece))
                             for (key, converter), piece in
                             zip(converters.items(), header)))

            d['coeff'] = []
            while len(d['coeff']) < d['ncoeff']:
                d['coeff'] += polyco.readline().split()

            d['coeff'] = np.array([float(item.translate(item.maketrans('Dd', 'ee')))
                                   for item in d['coeff']])

            t.append(d)

            line = polyco.readline()

    return t


def fortran_fmt(x, width=23, precision=16):
    base_fmt = '{:' + str(width) + '.' + str(precision) + 'e}'
    s = base_fmt.format(x)
    pre, dot, post = s.partition('.')
    mant, e, exp = post.partition('e')
    return pre[:-1] + '0' + dot + pre[-1] + mant + 'D{:+03d}'.format(int(exp) + 1)
