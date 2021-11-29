# Licensed under the GPLv3 - see LICENSE
"""Interfaces for dealing with PSRFITS fold-mode data."""

from astropy.io import fits
from astropy import log
from collections import defaultdict

from baseband_tasks.base import BaseTaskBase
from .hdu import HDU_map, subint_map


__all__ = ['open', 'get_readers', 'get_writer', 'PSRFITSReader',
           'PSRFITSWriter']


def open(filename, mode='r', **kwargs):
    """Function to open a PSRFITS file.

    Parameters
    ----------
    filename : str
        Input PSRFITS file name.
    mode : {'r', 'w'}, optional
        Whether to open for reading (default) or writing.
    **kwargs
        Additional arguments for opening the fits file and creating the reader.

    --- For opening the fits file :

    memmap : bool, optional
        Is memory mapping to be used? By default, this value is obtained from
        the configuration item ``astropy.io.fits.Conf.use_memmap`` (the
        default of which, in turn, is `True`).

    --- For creating the reader :

    weighted :  bool, optional
        Whether the data should be weighted along the frequency axis.
        Default is `True`.
    verify : bool, optional
        Whether to do basic checks on the PSRFITS HDUs.
        Default is `True`.

    --- For creating the writer :

    primary_hdu : ~baseband_tasks.io.psrfits.PSRFITSPrimaryHDU
        Currently required to be constructed before opening the file.
        This limitation will be lifted in future.
    **kwargs
        Further arguments to set up the SUBINT HDU.  These can include
        'start_time', 'sample_rate', 'sample_shape', 'shape',
        'samples_per_frame', 'polarization', 'frequency'.

    Returns
    -------
    reader : file-handle like reader for the first SUBINT HDU

    Raises
    ------
    RuntimeError
        If more than one SUBINT HDU is present.

    Notes
    -----
    The current version` only opens and reads one SUBINT HDU and ignores all
    other types of HDUs.
    """
    if mode == 'r':
        memmap = kwargs.pop('memmap', None)
        hdu_list = fits.open(filename, 'readonly', memmap=memmap)
        reader_list = get_readers(hdu_list, **kwargs)
        # TODO: allow support for more than one HDU.
        if len(reader_list) != 1:
            raise RuntimeError("Current reader can only read one SUBINT HDU.")
        # TODO Returning one HDU for now. But in the future, we may want to
        # return a group of it.
        return reader_list[0]

    elif mode == 'w':
        # Build Writer from scratch
        primary_hdu = kwargs.pop('primary_hdu', None)
        if primary_hdu is None:
            raise ValueError("need a primary hdu/meta data for building a"
                             " PSRFITS file.")
        writer = get_writer(filename, primary_hdu, **kwargs)
        return writer
    else:
        raise ValueError("Unknown mode '{}'. Currently only modes 'r' and 'w' "
                         "are supported.".format(mode))


def get_readers(hdu_list, **kwargs):
    """Function to read one PSRFITS file into a list of HDU Readers.

    Parameters
    ----------
    hdu_list : str
        Opened PSRFITS file.
    **kwargs
        Additional arguments for opening the fits file and creating the reader.
        Passes on to `~baseband_tasks.io.psrfits.PSRFITSReader`.

    Returns
    -------
    readers : list of the HDU readers.
    """
    buffers = defaultdict(list)
    for ii, hdu in enumerate(hdu_list):
        if hdu.name in HDU_map.keys():
            buffers[hdu.name].append(hdu)
        else:
            log.warning("Skipping HDU {} ({}), as it is not a supported "
                        "PSRFITS HDU.".format(ii, hdu.name))

    primary = buffers.pop('PRIMARY', [])
    if len(primary) != 1:
        raise ValueError("HDUList `{}` does not have a header"
                         " HDU or have more than one header"
                         " HDU.".format(hdu_list))
    primary_hdu = HDU_map['PRIMARY'](primary[0])
    psrfits_hdus = []
    for k, v in buffers.items():
        for hdu in v:
            psrfits_hdus.append(HDU_map[k](hdu, primary_hdu))

    # Build reader on the HDUs
    readers = []
    for hdu in psrfits_hdus:
        readers.append(PSRFITSReader(hdu, **kwargs))

    return readers


def get_writer(filename, hdu, **kwargs):
    """Function to init a PSRFITS HDU for writing.

    Parameters
    ----------
    filename : str or filehandle
        File to eventually write the FITS data to.
    shape : tuple
        The shape for data.
    **kwargs
        Additional arguments for creating PSRFITS writer.
        Passes on to `~baseband_tasks.io.psrfits.PSRFITSWriter`.
    """
    # TODO: not sure if the best solution is to build from a primary HDU.
    # Might be nicer if we can construct from kwargs.  Might want to use
    # the hdu_list_template in hdu.py
    if not isinstance(hdu, HDU_map['PRIMARY']):
        raise ValueError("need a PSRFITSPrimaryHDU to write FITS data")

    # TODO, is there any PSRFITS has no subint hdu?
    try:
        subint = subint_map[hdu.obs_mode]
    except KeyError as exc:
        exc.args += ("observation mode '{}' is not a valid"
                     " mode. Available modes are: {}"
                     .format(hdu.obs_mode, list(subint_map.keys())),)
        raise exc

    hdu = subint(primary_hdu=hdu)

    # Take input from the keywords
    # TODO: in analogy with baseband, should there be a .fromvalues() class
    # method?  We should not be accessing private properties here...
    for ppt in hdu._properties:
        ppt_value = kwargs.pop(ppt, None)
        if ppt_value is not None:
            setattr(hdu, ppt, ppt_value)

    if kwargs:
        raise TypeError('unused keyword arguments: {} '.format(kwargs))

    # TODO: should the file already opened for writing here?  Not good
    # to get an error if the file exists only when the writer is closed!
    return PSRFITSWriter(filename, hdu)


class PSRFITSReader(BaseTaskBase):
    """Wrapper for reading PSRFITS HDUs.

    Parameters
    ----------
    ih : wrapped PSRFITS HDU
        The input fits table HDU, wrapped in an interface from
        `~baseband_tasks.io.psrfits`.
    dtype : `~numpy.dtype`, optional
        Dtype of the samples.  Default: inferred from ``hdu``.
    weighted : bool, optional
        Whether to weight the data along the frequency axis using the
        'DAT_WTS' column.  Default of `True` should suffice for most purposes,
        but sometimes the weights are incorrect.

    --- **kwargs : meta data for the stream, which usually include

    frequency : `~astropy.units.Quantity`, optional
        Frequencies for each channel.  Should be broadcastable to the
        sample shape.  Default: inferred from ``hdu``.
    sideband : array, optional
        Whether frequencies are upper (+1) or lower (-1) sideband. Should be
        broadcastable to the sample shape.  Default: inferred from ``hdu``.
    polarization : array or (nested) list of char, optional
        Polarization labels.  Should broadcast to the sample shape,
        i.e., the labels are in the correct axis.  For instance,
        ``['X', 'Y']``, or ``[['L'], ['R']]``.  Default: inferred from ``hdu``.
    """
    # Note: this very light-weight wrapper around SubintHDU is mostly here
    # because eventually it might unify different/multiple HDUs.

    def __init__(self, ih, *, dtype=None, weighted=True, **kwargs):
        self.weighted = weighted
        super().__init__(ih, dtype=dtype, **kwargs)

    def _read_frame(self, frame_index):
        res = self.ih.read_data_row(frame_index, weighted=self.weighted).T
        return res.reshape((self.samples_per_frame, ) + self.sample_shape)

    def close(self):
        file = self.ih.hdu._file
        super().close()
        file.close()


class PSRFITSWriter:
    """Interface class for writing the PSRFITS HDUs

    Parameters
    ----------
    filename : str
        Output file name.
    hdu : wrapped PSRFITS HDU
        The output fits table HDU, wrapped in an interface from
        `~baseband_tasks.io.psrfits`.

    Notes
    -----
    Currently it only support write the PSRFITS primary HDU and Subint HDU.
    """
    # TODO: this should have some of the attributes from the underlying HDU,
    # in particular sample_shape, sample_rate, etc.  May be good to split
    # off a ReaderBase from base.Base, and then define a WriterBase as well.
    def __init__(self, filename, hdu):
        self.filename = filename
        self.hdu = hdu
        self.offset = 0

    def write(self, data):
        open_space = self.hdu.shape[0] - self.offset - data.shape[0]
        # TODO This may have to change if we want to write data to
        # multiple files
        if open_space < 0:
            raise RuntimeError("Not enough space for input data.")

        # FIXME add scaling
        # We need to have a hdu data setter, otherwise, this will not
        # scaled right.  That should also deal with this reshaping;
        # i.e., do implement a write_data_row function!
        data_row = data.reshape(self.hdu.samples_per_frame, -1)
        self.hdu.data['DATA'][
            0, self.offset:self.offset+data.shape[0]] = data_row
        self.offset += data.shape[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Dump the data to the underlying file.

        Also removes references to the underlying HDU.
        """
        # dump the data out
        # First convert the psrfits hdu to HDUList
        fits_hdu_list = fits.HDUList([self.hdu.primary_hdu.hdu, self.hdu.hdu])
        fits_hdu_list.writeto(self.filename, overwrite=False)
