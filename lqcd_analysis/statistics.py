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
                "Trusting lsqfit as specified anyway."
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


def aic(fit):
    """
    Computes the Akaike information criterion for a given fit.
    """
    return chi2(fit, augmented=True) + 2.0 * count_nparams(fit.p)


def aic_model_probability(fit):
    """
    Computes the model probability associated with the Akaike information
    criterion. Generically, the raw (unnormalized) model probability is
    raw prob = exp( -0.5 * IC).
    In the present case,
    IC = AIC - 2*(Ny - Ncut)
       = chi^2 + 2*Nparams - 2*(Ny - Ncut).
    This definition uses the following pieces:
    * chi^2 is the augmented chi2,
    * Nparams is the number of parameters in the model,
    * Ny is the total number data points, and
    * Ncut is the number of points cut / dropped by choosing tmin
    The difference (Ny - Ngamma) > 0 is simply "Ndata", the total number of 
    data points included in the fit. For a fixed model and fixed raw dataset, 
    Nparams and Ny are constant and cancel in the normalized probabilites used
    in model averaging. So, for fixed model and fixed raw dataset, 
    IC --> chi^2 + 2 Ncut.
    """
    ndata = count_ndata(fit.y)
    ic = aic(fit) - 2.0 * ndata
    # Recall the basic log-likelihood function in least-squares fitting is 
    # -1/2 * chi2^2, with a factor of -1/2. So we must multiply the information
    # criterion by -1/2 in order to get the full log-likelihood.
    log_likelihood = -0.5 * ic
    return np.exp(log_likelihood)


def model_avg(gv_list, pr_list):
    """
    Given a list of single-model expectation values {<f(a)>_M} as gvars, and a
    list of raw model probabilities, computes the model-averaged estimate for
    <f(a)> as a gvar.
    """
    # Normalize model probabilities to 1
    prob = pr_list / np.sum(pr_list)
    # Mean is just the weighted average
    mean_avg = np.sum(gv.mean(gv_list) * prob)
    # Variance
    # First, a weighted average of individual variances
    var_avg = np.sum(gv.var(gv_list) * prob)
    # Second, the "systematic error" due to model choice
    # <a>^2 P(M|D) - (<a> P(M|D))^2
    var_avg += np.sum(gv.mean(gv_list)**2 * prob)  
    var_avg -= (np.sum(gv.mean(gv_list) * prob))**2
    return gv.gvar(mean_avg, np.sqrt(var_avg))


class FitStats(object):
    """Container for various fit statistics."""
    def __init__(self, fit):
        self.chi2 = chi2(fit, augmented=False)
        self.chi2_aug = chi2(fit, augmented=True)
        self.nparams = count_nparams(fit.p)
        self.ndata = count_ndata(fit.y)
        self.q_value = correlated_q(self.chi2_aug, self.ndata)
        self.p_value = correlated_p(self.chi2, self.ndata, self.nparams)
        self.aic = aic(fit)
        self.model_probability = aic_model_probability(fit)
