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
        format1 : str
            The name for the first format.
        format2 : str
            The name for the second format
    """
    def __init__(self, name, format1, format2, mapper12={}, mapper21={}):
        self.name = name
        # TODO should this be the class type of the format?
        self.mapper = {format1: mapper12, format2: mapper21}

    def __call__(self, source, target_key):
        """The highlevel wrapper for translating qurey one key value from the
        source object.

            Parameter
            ---------
            source : object
                The source object for qureying the key value
            target_key : str
                The qurey key name.
        """
        # NOTE, not sure how this going to work.
        format_type = type(source)
        mapper_funcs = self.mapper[format_type]
        if target_key not in mapper_funcs.keys():
            raise ValueError("Target key '{}' is not in the mapper"
                             " functions".format(target_key))
        return mapper_funcs[target_key](source)


    def add_mapper_func(self, source, target_key, method):
        mapper_entry = {target_key: method}
        self.mapper[source].update(mapper_entry)
