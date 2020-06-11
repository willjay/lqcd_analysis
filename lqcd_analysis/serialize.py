"""
Implements a serializable extension of the various "nonlinear_fit" objects
appearing in lsqfit.
"""
import numpy as np
import datetime as datetime
from . import statistics

def _to_text(adict):
    """ Wrapper for converting dicts to text for postgres"""
    new_dict = {}
    for key, val in sorted(adict.items()):
        new_dict[key] = str(val)
    return '$delim${{{0}}}$delim$'.format(str(new_dict))

class SerializableNonlinearFit:
    def __init__(self, fit):
        # Copy over attributes to serializable instance
        for attr in dir(fit):
            if attr in ['__class__', '__weakref__', 'p']:
                continue
            self.__setattr__(attr, fit.__getattribute__(attr))
        if np.isnan(fit.chi2) or np.isinf(fit.chi2):
            self.failed = True
        else:
            self.failed = False
        stats = statistics.FitStats(fit)
        for attr in ['chi2','chi2_aug','nparams','ndata','q_value','p_value']:
            self.__setattr__(attr, stats.__getattribute__(attr))    

    @property
    def p(self):
        return self._getp()
    
    def serialize(self):
        payload = {
            'prior': _to_text(self.prior),
            'params': _to_text(self.p),
            'Q': self.Q,
            'chi2': self.chi2,
            'chi2_aug': self.chi2_aug,
            'q_value': self.q_value,
            'p_value': self.p_value,
            'nparams': self.nparams,
            'npoints': self.ndata,
            'calcdate': datetime.datetime.now(),
        }
        return payload