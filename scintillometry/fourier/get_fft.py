# Licensed under the GPLv3 - see LICENSE

from . import FFT_MAKER_CLASSES

def get_fft(engine_name, **kwargs):
    """FFT factory selector."""
    return FFT_MAKER_CLASSES[engine_name](**kwargs)
