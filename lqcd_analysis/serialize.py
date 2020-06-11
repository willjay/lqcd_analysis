"""
Implements a serializable extension of the various "nonlinear_fit" objects
appearing in lsqfit.
"""
import numpy as np
import datetime as datetime

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

    @property
    def p(self):
        return self._getp()
    
    def serialize(self):
        payload = {
            'prior': _to_text(self.prior),
            'params': _to_text(self.p),
            'Q': self.Q,
            'chi2': self.chi2,
            'nparams': sum([len(np.ravel(pj)) for pj in self.p.values()]),
            'npoints': sum([len(yi) for yi in self.y.values()]),
            'calcdate': datetime.datetime.now(),
        }
        return payload