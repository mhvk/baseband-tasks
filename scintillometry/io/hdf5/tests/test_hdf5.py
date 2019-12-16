import numpy as np
import pytest
import h5py
from astropy import units as u
from astropy.time import Time

import baseband
import baseband.data

from scintillometry.base import SetAttribute

from ... import hdf5


class TestHDF5Basics:
    def setup(self):
        self.info = {'sample_shape': (8, 2),
                     'samples_per_frame': 100,
                     'sample_rate': 32*u.MHz,
                     'time': Time('2015-06-07T08:09:10')}

    def test_create_raw_header(self):
        header = hdf5.HDF5Header(dtype='c8', **self.info)
        assert set(header.keys()) == set(self.info.keys()) | {'dtype'}
        assert dict(header) == dict(dtype='c8', **self.info)
        assert all(getattr(header, key) == self.info[key]
                   for key in self.info.keys())
        assert isinstance(header['dtype'], str)
        assert header['dtype'] == 'c8'
        assert isinstance(header.dtype, np.dtype)
        assert header.dtype == 'c8'
        assert not hasattr(header, 'bps')
        assert not hasattr(header, 'complex_data')
        with pytest.raises(KeyError, match='dtype'):
            hdf5.HDF5Header(**self.info)

    def test_create_coded_header(self):
        header = hdf5.HDF5Header(bps=2, complex_data=False, **self.info)
        assert set(header.keys()) == (set(self.info.keys())
                                      | {'bps', 'complex_data'})
        assert dict(header) == dict(bps=2, complex_data=False, **self.info)
        assert all(getattr(header, key) == self.info[key]
                   for key in self.info.keys())
        assert header.bps == 2
        assert header.complex_data is False
        assert not hasattr(header, 'dtype')
        with pytest.raises(KeyError, match='complex_data'):
            hdf5.HDF5Header(bps=2, **self.info)


class TestHDF5:
    def setup(self):
        self.fh = baseband.open(baseband.data.SAMPLE_VDIF)
        self.data = self.fh.read()
        self.fh.seek(0)
        frequency = 311.25 * u.MHz + (np.arange(8.) // 2) * 16. * u.MHz
        sideband = np.tile([-1, +1], 4)
        self.wrapped = SetAttribute(self.fh, frequency=frequency,
                                    sideband=sideband)

    def teardown(self):
        self.wrapped.close()
        self.fh.close()

    def check(self, stream, header, attrs=None, as_key=False):
        if attrs is None:
            if hasattr(stream, 'bps'):
                attrs = ('sample_shape', 'sample_rate', 'time',
                         'bps', 'complex_data')
                exclude = ('dtype',)
            else:
                attrs = ('sample_shape', 'dtype', 'sample_rate', 'time',
                         'frequency', 'sideband')
                exclude = ()

        is_header = isinstance(header, hdf5.HDF5Header)

        assert stream.shape[0] == header.samples_per_frame
        if is_header:
            assert stream.shape[0] == header['samples_per_frame']
        for attr in attrs:
            stream_attr = getattr(stream, (attr if attr != 'time'
                                           else 'start_time'))
            header_attr = getattr(header, attr)
            assert np.all(header_attr == stream_attr)
            if is_header:
                if attr in exclude:
                    assert attr not in header
                else:
                    header_value = header[attr]
                    assert np.all(header_value == stream_attr)

    def test_header_from_bandband(self):
        header = hdf5.HDF5Header.fromvalues(self.fh)
        assert 'bps' in header
        self.check(self.fh, header)
        for attr in 'frequency', 'sideband', 'polarization':
            with pytest.raises(AttributeError):
                getattr(header, attr)

    def test_header_from_stream(self):
        header = hdf5.HDF5Header.fromvalues(self.wrapped)
        self.check(self.wrapped, header)
        with pytest.raises(AttributeError):
            header.polarization

    def test_payload_from_baseband(self, tmpdir):
        header = hdf5.HDF5Header.fromvalues(self.fh)
        filename = str(tmpdir.join('payload.hdf5'))
        with h5py.File(filename, 'w') as h5:
            pl = hdf5.HDF5Payload.fromfile(h5, header)
            assert pl.words.dtype == 'u4'
            assert pl.shape == (40000, 8)
            assert pl.words.shape == (20000,)

    def test_payload_from_stream(self, tmpdir):
        header = hdf5.HDF5Header.fromvalues(self.wrapped)
        filename = str(tmpdir.join('payload.hdf5'))
        with h5py.File(filename, 'w') as h5:
            pl = hdf5.HDF5Payload.fromfile(h5, header)
            assert pl.words.dtype == 'f4'
            assert pl.shape == (40000, 8)
            assert pl.words.shape == (40000, 8)

    @pytest.mark.parametrize('stream_name', ['fh', 'wrapped'])
    def test_copy_stream(self, stream_name, tmpdir):
        stream = getattr(self, stream_name)
        filename = str(tmpdir.join('copy.hdf5'))
        with hdf5.open(filename, 'w', template=stream) as f5w:
            if stream_name == 'fh':
                assert f5w.bps == 2
            else:
                assert not hasattr(f5w, 'bps')
            self.check(stream, f5w)
            header0 = f5w.header0
            self.check(stream, header0)
            f5w.write(self.data)

        with h5py.File(filename, 'r') as h5:
            assert set(h5.keys()) == {'header', 'payload'}
            header = hdf5.HDF5Header.fromfile(h5)
            self.check(stream, header)
            assert header == header0
            payload = hdf5.HDF5Payload.fromfile(h5, header)
            assert np.all(payload.data == self.data)
            assert np.all(payload[:] == self.data)

        with hdf5.open(filename, 'r') as f5r:
            self.check(stream, f5r)
            assert f5r.header0 == header0
            data = f5r.read()
            assert np.all(data == self.data)

    @pytest.mark.parametrize('stream_name', ['fh', 'wrapped'])
    def test_copy_stream_copy(self, stream_name, tmpdir):
        # Check that we can copy ourselves and not mess up depending
        # on raw vs encoded data.
        stream = getattr(self, stream_name)
        filename = str(tmpdir.join('copy.hdf5'))
        with hdf5.open(filename, 'w', template=stream) as f5w:
            f5w.write(self.data)

        copyname = str(tmpdir.join('copycopy.hdf5'))
        with hdf5.open(filename, 'r') as f5r:
            with hdf5.open(copyname, 'w', template=f5r) as f5w:
                if stream_name == 'fh':
                    assert f5w.bps == 2
                else:
                    assert not hasattr(f5w, 'bps')
                self.check(stream, f5w)
                header0 = f5w.header0
                self.check(stream, header0)
                f5w.write(self.data)

    @pytest.mark.parametrize('stream_name', ['fh', 'wrapped'])
    def test_stream_as_output(self, stream_name, tmpdir):
        stream = getattr(self, stream_name)
        filename = str(tmpdir.join('copy.hdf5'))
        with hdf5.open(filename, 'w', template=stream) as f5w:
            stream.seek(0)
            stream.read(out=f5w)

        with hdf5.open(filename, 'r') as f5r:
            self.check(stream, f5r)
            data = f5r.read()
            assert np.all(data == self.data)
