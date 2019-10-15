"""
Some useful utilities / decorators for debuging
"""
import time
import functools
import logging

LOGGER = logging.getLogger(__name__)


def timing(fcn):
    """Time the execution of fcn. Use as decorator."""
    @functools.wraps(fcn)
    def wrap(*args, **kwargs):
        """Wrapped version of the function."""
        t_initial = time.time()
        result = fcn(*args, **kwargs)
        t_final = time.time()
        LOGGER.info(
            "TIMING: %s took: %.4f sec",
            fcn.__name__,
            t_final - t_initial
        )
        return result
    return wrap
