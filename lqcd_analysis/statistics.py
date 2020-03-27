"""
Module containing various statistics functions and a container for evaluating
and storing them for a given fit.
"""
import logging
import numpy as np
import scipy
import gvar as gv

LOGGER = logging.getLogger(__name__)


def correlated_q(chi2_aug, ndata):
    """
    Computes the correlated Q-value using the survival function (sf),
    i.e. the complementary cumulative distribution function (CCDF).
    See Appendix B of A. Bazavov et al., PRD 93, 113016 (2016).
    [https://arxiv.org/abs/1602.03560]
    """
    return scipy.stats.distributions.chi2.sf(chi2_aug, ndata)


def correlated_p(chi2_value, ndata, nparams):
    """
    Computes the correlated p-value using the survival function (sf),
    i.e. the complementary cumulative distribution function (CCDF).
    See Appendix B of A. Bazavov et al., PRD 93, 113016 (2016).
    [https://arxiv.org/abs/1602.03560]
    """
    nu = ndata-nparams
    return scipy.stats.distributions.chi2.sf(chi2_value, nu)


def correlated_chi2(yfit, ydata):
    """Computes the correlated chi2 function."""
    # Get the fit values, data, and covariance matrix as dicts
    cov_dict = gv.evalcov(ydata)
    # Enforce an ordering of keys
    klist = list(ydata.keys())
    # Reserve space for arrays
    # Implementation note: flatten allows for the special case
    # of matrix-valued priors, e.g., for the transition matrix Vnn
    sizes = [len(np.asarray(ydata[key]).flatten()) for key in klist]
    total_size = sum(sizes)
    diff = np.empty(total_size)
    cov_arr = np.zeros((total_size, total_size))
    # Infer the start and end points for intervals
    ends = np.cumsum(sizes)
    starts = ends - sizes
    # Populate arrays
    for start_i, end_i, key_i in zip(starts, ends, klist):
        diff[start_i:end_i] = np.asarray(gv.mean(ydata[key_i] - yfit[key_i])).flatten()
        for start_j, end_j, key_j in zip(starts, ends, klist):
            try:
                cov_arr[start_i:end_i, start_j:end_j] =\
                    cov_dict[(key_i, key_j)]
            except ValueError:
                # Implementation note: matrix-valued priors have
                # multi-dimensional covariance matrices,
                # which must be reshaped in a 2x2 array
                cov_arr[start_i:end_i, start_j:end_j] = \
                    cov_dict[(key_i, key_j)].reshape(
                        end_i - start_i, end_j - start_j
                    )
    # The "usual" chi2 function (ydata-yfit).cov_inv.(ydata-yfit)
    try:
        result = np.dot(diff, np.linalg.solve(cov_arr, diff))
    except np.linalg.LinAlgError:
        result = np.nan
    return result


def chi2(fit, augmented=False, trust_lsqfit=False):
    """Computes the chi2 function."""
    if trust_lsqfit:
        if not augmented:
            LOGGER.warning((
                "Caution: lsqfit computes an augmented chi2 function."
                "Trusting lsqfit as specified anway."
            ))
        return fit.chi2
    # Standard chi2, without the term involving the prior
    result = correlated_chi2(fit.fcn(fit.p), fit.y)
    if augmented:
        # Augmeted chi2, including the term with the prior
        result += correlated_chi2(fit.p, fit.prior)
    return result


def count_ndata(ydata):
    """ Counts the number of data points in ydata."""
    ndata = 0
    if hasattr(ydata, 'keys'):
        for key in ydata.keys():
            ndata += len(np.asarray(ydata[key]).flatten())
    else:
        ndata = len(np.asarray(ydata).flatten())
    return ndata


def count_nparams(params):
    """
    Counts the number of fit parameters np, being careful
    to avoid double counting of "log priors" are present.
    """
    nparams = 0
    for pname, val in params.items():
        log_pname = 'log({0})'.format(pname)
        if log_pname in params.keys():
            # Skip this parameter
            continue
        if hasattr(val, '__len__'):
            nparams += len(np.asarray(val).flatten())
        else:
            nparams += 1
    return nparams


class FitStats(object):
    """Container for various fit statistics."""
    def __init__(self, fit):
        self.chi2 = chi2(fit, augmented=False)
        self.chi2_aug = chi2(fit, augmented=True)
        self.nparams = count_nparams(fit.p)
        self.ndata = count_ndata(fit.y)
        self.q_value = correlated_q(self.chi2_aug, self.ndata)
        self.p_value = correlated_p(self.chi2, self.ndata, self.nparams)
