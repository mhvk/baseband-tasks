"""core.py defines the classes for reading pulsar data from a non-baseband
format.
"""

from ...base import Base
from astropy.io import fits
from .hdu import HDU_map
from astropy import log
from collections import defaultdict

__all__ = ['open', 'open_read', 'PsrfitsReader']


def open(filename, mode='r', **kwargs):
    """ Function to open a PSRFITS file.

    Parameters
    ----------
    filename : str
        Input PSRFITS file name.
    mode : str
        Open mode, currently, it only supports 'r'/'read' mode

    **kwargs
        Keyword arguments help the psrfits file handling.
        memmap : bool, optional
            Is memory mapping to be used? This value is obtained from the
            configuration item astropy.io.fits.Conf.use_memmap. Default is True.
        weighted : bool, optional
            Is the returning data weighted along the frequency axis.
            Default is True.
    Note
    ----
    The current version of open() function only opens and reads one SUBINT HDU
    and ignores the other types of HDUs. If more than one SUBINT HUD are
    provided, a RuntimeError will be raised.
    """

    if mode == 'r':
        reader_list = open_read(filename, **kwargs)
        # TODO, this need to be changed, if we can support more HDUs.
        if len(reader_list) != 1:
            raise RuntimeError("Current reader can only read one SUBINT HDU.")
        return reader_list[0]
    else:
        raise ValueError("Unknown mode '{}'. Currently only 'r' mode are"
                         " supported.".format(mode))


def open_read(filename, **kwargs):
    """ Function to read one PSRFITS file into a list of HDU Readers.

    Parameters
    ----------
    filename : str
        File name of the input PSRFITS file
    **kwargs:
        Other keyword arguments for creating the reader.
        memmap : bool, optional
            Is memory mapping to be used? This value is obtained from the
            configuration item astropy.io.fits.Conf.use_memmap. Default is True.
        weighted :  bool, optional
            Is the returning data weighted along the frequency axis.
            Default is True.
    Return
    ------
    A list of the HDU readers.
    """
    memmap = kwargs.get('memmap', None)
    hdus = fits.open(filename, 'readonly', memmap=memmap)
    buffer = defaultdict(list)
    for ii, hdu in enumerate(hdus):
        if hdu.name in HDU_map.keys():
            buffer[hdu.name].append(hdu)
        else:
            log.warn("Skipping HDU {} ({}), as it is not a known PSRFITs"
                     " HDU.".format(ii, hdu.name))

    primary = buffer.pop('PRIMARY', [])
    if len(primary) != 1:
        raise ValueError("File `{}` does not have a header"
                         " HDU or have more than one header"
                         " HDU.".format(filename))
    primary_hdu = HDU_map['PRIMARY'](primary[0])
    psrfits_hdus = []
    for k, v in buffer.items():
        for hdu in v:
            psrfits_hdus.append(HDU_map[k](primary_hdu, hdu))

    # Build reader on the HDUs
    readers = []
    for hdus in psrfits_hdus:
        readers.append(PsrfitsReader(hdus, **kwargs))
    return readers


class PsrfitsReader(Base):
    """Reader class defines the API for reading PSRFITS files.

    Parameters
    ----------
    hdu : object
        The input fits table HDU.
    frequency : `~astropy.units.Quantity`, optional
        Frequencies for each channel.  Should be broadcastable to the
        sample shape.  Default: unknown.
    sideband : array, optional
        Whether frequencies are upper (+1) or lower (-1) sideband.
        Should be broadcastable to the sample shape.  Default: unknown.
    polarization : array or (nested) list of char, optional
        Polarization labels.  Should broadcast to the sample shape,
        i.e., the labels are in the correct axis.  For instance,
        ``['X', 'Y']``, or ``[['L'], ['R']]``.  Default: unknown.
    dtype : `~numpy.dtype`, optional
        Dtype of the samples.
    weighted : bool, optional
        Is the returning data weighted along the frequency axis.
        Default is True.

    """
    def __init__(self, hdu, frequency=None, sideband=None, polarization=None,
                 dtype=None, weighted=True, **kwargs):
        self.fh_raw = hdu
        self._req_args = {'shape': None, 'start_time': None,
                          'sample_rate': None}
        # Get required arguments from the source file handle
        for rg in self._req_args.keys():
            try:
                self._req_args[rg] = getattr(self.fh_raw, rg)
            except AttributeError as exc:
                exc.args += ("souce file should define '{}'.".format(rg),)
                raise exc

        samples_per_frame = getattr(self.fh_raw, 'samples_per_frame', None)
        if frequency is None:
            frequency = getattr(self.fh_raw, 'frequency', None)
        if sideband is None:
            sideband = getattr(self.fh_raw, 'sideband', None)
        if polarization is None:
            polarization = getattr(self.fh_raw, 'polarization', None)
        if dtype is None:
            dtype = getattr(self.fh_raw, 'dtype', None)

        self.weighted = weighted

        super().__init__(self._req_args['shape'], self._req_args['start_time'],
                         self._req_args['sample_rate'],
                         samples_per_frame=samples_per_frame,
                         frequency=frequency, sideband=sideband,
                         polarization=polarization, dtype=dtype)

    def _read_frame(self, frame_index):
        res = self.fh_raw.read_data_row(frame_index, self.weighted).T
        return res.reshape((self.samples_per_frame, ) + self.sample_shape)
