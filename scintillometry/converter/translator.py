""" translator.py defines the container class for translating functions betweet
different file format.
"""


class Translator(dict):
    """The base translator class which defines set of common API functions for
    translatort container subclasses.

    Parameter
    ---------
    name : str
        The name of translator instance.
    source : str
        The name of the input format.
    target : str
        The name of the target format.
    """
    def __init__(self, name, source, target):
        self.name = name
        self.source = source
        self.target = target

        super(Translator, self).__init__()

    def check_input(self, input_object):
        """ This function checks if the input file object matches the translator
        source.
        """
        raise NotImplementError
