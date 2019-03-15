""" translator.py defines the container class for translating functions betweet
different file format.
"""


class TranslatorBase:
    """The base translator class which defines set of common API functions for
    translatort container subclasses.

        Parameter
        ---------
        name : str
            The name of translator instance.
        format1 :
        format2 :
    """
    def __init__(self, name, format1, format2, mapper12={}, mapper21={}):
        self.name = name
        self.format1 = format1
        self.format2 = format2
        self.mapper = (mapper12, mapper21)

    def __call__(self, source, target_key):
        pass

    def add_mapper_func(self, source, target_key, method):
        mapper_entry = {target_key: method}
        self.mapper[source - 1].update(mapper_entry)
