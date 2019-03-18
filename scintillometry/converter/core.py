"""Converter core.py define the converter base classes for converting baseband
style file handler to the other format file objects (e.g., PSRFITS file object)
"""


from ..generators import StreamGenerator
import numpy as np


class FormatReader(StreamGenerator):
    """ FormatReader class is the base class for reading data to a `baseband`
    style file handler from other format object.

        Parameter
        ---------
        format_object : object.
            The file object of the input format.
        translator : dict or dictionary-like
            Translator functions for getting the information from the input
            file object.
    """
    def __init__(self, format_object, translator=None, **kwargs):
        self.format_object = format_object
        self.translator = translator
        self.args = {'function': self.read_format_data}
        self.args.update(kwargs)
        self.required_args = ['shape', 'start_time', 'sample_rate']
        self.optional_args = ['samples_per_frame', 'frequency', 'sideband',
                              'polarization', 'dtype']
        self._prepare_args()
        super(FormatReader, self).__init__(**self.args)


    def _prepare_args(self):
        """This setup function setups up the argrument for initializing the
        StreamGenerator.
        """
        input_args_keys = self.args.keys()
        translatort_keys = self.translator.keys()
        for rg in self.required_args:
            if rg not in translatort_keys and rg not in input_args_keys:
                raise ValueError("'{}' is required. You can input it while "
                                 "initialization or give a function in the "
                                 "translator.")
            self.args[rg] = self.translator[rg](self.format_object)

        for og in self.required_args:
            if og in translatort_keys and og not in input_args_keys:
                self.args[og] = self.translator[og](self.format_object)

    def read_format_header(self):
        """This function defines the function to read header from the input
        format object. In the base class, it will not be defined.
        """
        raise NotImplementedError

    def read_format_data(self):
        """This function defines the function to read data from the input
        format object. In the base class, it will not be defined.
        """
        raise NotImplementedError


class FormatWritter(self):
    """FormatWritter class is the base class for writing data from a `baseband`
    style file handler to other format.
    """
    def __init__(self):
        pass
