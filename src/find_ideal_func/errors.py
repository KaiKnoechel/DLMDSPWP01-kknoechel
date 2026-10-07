""" Custom Exceptions for the find_ideal_func module. """


class NoMatchingError(Exception):
    """Raise when no testpoint is possible to map to any of the four ideal functions."""
    pass


class DatavalidatorError(Exception):
    """Raise when the data validation fails."""
    pass


class IdealFunctionError(Exception):
    """Base class for exceptions related to ideal functions."""
    pass

class IdealFunctionNotFoundError(IdealFunctionError):
    """Raise when the ideal function is not found in the database."""
    pass