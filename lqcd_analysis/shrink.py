"""
Shrink the data covariance matrix using linear or nonlinear techniques

##################################
# Nonlinear Shrinkage Estimation #
##################################

  S      - empirical covariance matrix, estimated from the data sample
  Sigma  - true covariance matrix underlying the data population distribution
  S*     - inferred covariance matrix
  T      - target (model) covariance matrix

The convex combination

     S* = Lambda * T + (1-Lambda) S

where 0 <= Lambda <= 1 is a valid covariance matrix.

The Ledoit and Wolf shrinkage procedure locates an optimal Lambda by minimizing
the squared error loss function between the inferred and true covariance

     L(Lambda) = || S* - Sigma || ~= || Lambda * T + (1-Lambda) S - Sigma ||

where ||x|| is the Frobenius norm.

References:

    1.  Chen Y., et al., "Shrinkage Algorithms for MMSE Covariance Estimation",
         2010, IEEE Transactions on Signal Processing, Vol. 58, No. 10.

    2.  Pope A., and Szapudi I., "Shrinkage estimation of the power spectrum
         covariance matrix", 2008, arXiv:0711.2509v2

    3.  Schaefer J., and Strimmer K., "A Shrinkage Approach to Large-Scale
         Covariance Estimation and Implications for Functional Genomics", 2005,

    4.  Ledoit O., Wolf M., 2003, J. Empirical Finance, 10, 603

#########################################
# Direct Nonlinear Shrinkage Estimation #
#########################################

A nonlinear shrinkage estimator of the covariance matrix that does not
require recovering the population eigenvalues first. We estimate the
sample spectral density and its Hilbert transform directly by
smoothing the sample eigenvalues with a variable-bandwidth
kernel. Relative to numerically inverting the so-called QuEST
function, the main advantages of direct kernel estimation are: (1) it
is much easier to comprehend because it is analogous to kernel density
estimation; (2) it is only twenty lines of code in Matlab - as opposed
to thousands - which makes it more verifiable and customizable; (3) it
is 200 times faster without significant loss of accuracy; and (4) it
can handle matrices of a dimension larger by a factor of ten. Even for
dimension 10,000, the code runs in less than two minutes on a desktop
computer; this makes the power of nonlinear shrinkage as accessible to
applied statisticians as the one of linear shrinkage.

Reference

"Direct Nonlinear Shrinkage Estimation of Large-Dimensional Covariance
 Matrices", Olivier Ledoit and Michael Wolf,
University of Zurich, Department of Economics, Working Paper Series,
Paper No. 264 (2017).
"""
# pylint: disable=invalid-name
import logging
import numpy as np
import gvar as gv
import time

LOGGER = logging.getLogger(__name__)


def main():
    """TODO: Add main function"""


def rblw_shrink_correlation_identity(samples):
    """
    Shrink the sample correlation matrix using the model T = Identity.
    The Rao-Blackwell Ledoit-Wolf estimator specialized for the case of
    Gaussian noise from Reference [1]. The reference notes that the RBLW
    estimator provably improves on the LW method (i.e. has a smaller min loss
    function) *under the Gaussian noise model*.
    """
    n, p = map(float, samples.shape)
    # S is sample correlation matrix, i.e.,
    # the max likelihood estimator of normalized distribution variance
    S = np.corrcoef(samples, rowvar=0)
    trSsq = np.trace(np.dot(S, S))
    Lambda = ((n - 2.) / n * trSsq + p**2) / ((n + 2.) * (trSsq - p))
    Lambda = min(Lambda, 1.)
    LOGGER.info(
        ('Rao-Blackwell Ledoit-Wolf data correlation matrix shrinkage.'
         'Lambda: %f'),
        Lambda
    )
    S *= 1.0 - Lambda
    np.fill_diagonal(S, 1.0)
    return (Lambda, S)


def oa_shrink_correlation_identity(samples):
    """
    Shrink the sample correlation matrix using the model T = Identity.
    The Oracle Approximating (OA) shrinkage estimator for the case of Gaussian
    noise from Reference [1]. The reference notes that the OA estimator usually
    outperforms LW or RBLW for Gaussian noise, especially when n/p is not
    large.
    Note: In Ref. [1] there are typos in Eqs(20-32)! e.g. (1-2)/p ==> (1-2/p)
    """
    n, p = map(float, samples.shape)
    # sample correlation matrix: the max likelihood estimator of normalized
    # distribution variance
    s = np.corrcoef(samples, rowvar=0)
    tr_s_sq = np.trace(np.dot(s, s))
    lam = ((1. - 2. / p) * tr_s_sq + p**2) /\
          ((n + 1. - 2. / p) * (tr_s_sq - p))
    lam = min(lam, 1.)
    LOGGER.info(
        'Oracle Approximating data correlation matrix shrinkage. Lambda: %f',
        lam
    )

    s *= 1.0 - lam
    np.fill_diagonal(s, 1.0)
    return (lam, s)


def lw_shrink_correlation_identity(samples):
    """
    Shrink the sample correlation matrix using the identity matrix,
    T[i,j] = delta(i,j), as the target. The Ledoit-Wolf estimator.
    data matrix layout: samples[sample,obs]
    """
    # copy data
    d = np.copy(samples)
    n, p = d.shape
    # normalize the data copy: mean=0 sigma=1
    mean = np.mean(d, axis=0)
    sigma = np.std(d, axis=0)
    for k in range(n):
        d[k, :] = (d[k, :] - mean) / sigma
    # compute S (the correlation matrix) the max likelihood estimator of
    # normalized distribution variance
    s = np.corrcoef(d, rowvar=0)
    # compute Frobenius norm squared
    f = 0.
    # off-diagonal
    for i in range(p):
        xi = d[:, i]
        for j in range(i + 1, p):
            f += np.sum((xi * d[:, j] - s[i, j])**2)
    f *= 2.0
    # diagonal
    for i in range(p):
        xi = d[:, i]
        f += np.sum((xi * xi - s[i, i])**2)
    tr_s_sq = np.trace(np.dot(s, s))
    lam = f / (n**2 * (tr_s_sq - p))
    lam = max(min(lam, 1.0), 0.0)
    LOGGER.info(
        'Ledoit-Wolf data correlation matrix shrinkage. Lambda: %f',
        lam
    )
    s *= 1.0 - lam
    np.fill_diagonal(s, 1.0)
    return (lam, s)


def _pav(dat):
    """
    PAV uses the pair adjacent violators method to produce a monotonic
    smoothing of dat. Adapted from Sean Collins (2006) as part of the EMAP
    toolbox.
    """
    ndat = len(dat)
    v = np.copy(dat)
    lvlsets = np.array([range(ndat), range(ndat)], dtype=int)
    while True:
        deriv = np.diff(v)
        viol = list()
        for j in range(ndat - 1):
            if deriv[j] < 0:
                viol.append(j)
        viol = np.array(viol, dtype=int)
        if len(viol) == 0:
            break
        start = lvlsets[0][viol[0]]
        last = lvlsets[1][viol[0] + 1]
        n = last - start + 1
        avg = np.sum(v[start:last + 1]) / n
        v[start:last + 1] = avg
        lvlsets[0][start:last + 1] = start
        lvlsets[1][start:last + 1] = last
    return v


def direct_nl_shrink(sample_ev, n):
    """Computes direct nonlinear shrinkage."""
    isDesending = np.all(sample_ev[:-1] >= sample_ev[1:])
    if isDesending:
        sam_ev = np.flip(sample_ev, 0)
    else:
        sam_ev = sample_ev
    p = len(sam_ev)
    c = float(p) / n
    lam = sam_ev[max(0, p - n):p]  # non-zero eigenvalues
    Lt = np.tile(lam, (min(p, n), 1))
    L = np.transpose(Lt)
    # Eq.(5.4)
    h = float(n)**(-.35)
    hsq = h * h
    # Eq.(5.2)
    ftilde = np.mean(
        np.sqrt(np.maximum(0., 4 * np.square(Lt) * hsq - np.square(L - Lt))) /
        (2 * np.pi * np.square(Lt) * hsq), axis=1)
    # Eq.(5.3)
    Hftilde = np.mean(
        (np.sign(L - Lt) *
         np.sqrt(np.maximum(0., np.square(L - Lt) - 4 * np.square(Lt) * hsq)) -
         L + Lt) / (2 * np.pi * np.square(Lt) * hsq),
        axis=1)

    if p < n:
        # Eq.(4.3)
        dtilde = lam / (np.square(np.pi * c * lam * ftilde) +
                        np.square(1. - c - np.pi * c * lam * Hftilde))
    else:
        # Eq.(C.8)
        Hftilde0 = (1. - np.sqrt(1. - 4. * hsq)) / \
            (2. * np.pi * hsq) * np.mean(1. / lam)
        # Eq.(C.5)
        dtilde0 = 1. / (np.pi * float(p - n) / n * Hftilde0)
        # Eq.(C.4)
        dtilde1 = lam / (np.pi**2 * np.square(lam) *
                         (np.square(ftilde) + np.square(Hftilde)))
        dtilde = np.concatenate(
            (dtilde0 * np.ones((p - n,), dtype=np.float64), dtilde1))
    # Eq.(4.5)
    dhat = _pav(dtilde)
    if isDesending:
        dhat = np.flip(dhat, 0)
    return dhat

def linear_shrink(corr, lam=0, target=None):
    """
    Computes the linear shrinkage estimator for the correlation matrix.
    Args:
        corr: np.ndarray, the input correlation matrix
        lam: float, the shrinkage parameter in (0, 1)
    Returns:
        np.ndarray, the shrinkage estimator, "lam*identity + (1-lam)*corr"
    """
    if (lam < 0) or (lam > 1):
        raise ValueError("Shrinkage parameter must be in (0, 1)")
    if target is None:
        target = np.eye(len(corr))
    if target.shape != corr.shape:
        raise ValueError("Incommensurate shapes.")
    return lam*target + (1 - lam)*corr


def linear_shrink_dataset(dataset, lam=0, target=None):
    """
    Applies linear shrinkage to the correlated dataset of gvars.
    Args:
        dataset: list / np.ndarray of correlated gvars
        lam: float, the shrinkage parameter in (0, 1)
    Returns:
        np.ndarray, the dataset but with modified correlation matrix
    """
    mean = gv.mean(dataset)
    sdev = np.diag(gv.sdev(dataset))
    corr = gv.evalcorr(dataset)
    # cov = sdev @ linear_shrink(corr, lam=lam, target=target) @ sdev
    cov = np.matmul(sdev, np.matmul(linear_shrink(corr, lam=lam, target=target), sdev))
    return gv.gvar(mean, cov)


class LinearShrinkage:
    """
    Computes the linear shrinkage estimator for a covariance matrix.
    The original reference is:
    O. Ledoit and M. Wolf, Journal of Multivariate Analysis 88, 365 (2004).
    [https://dx.doi.org/https://doi.org/10.1016/S0047-259X(03)00096-4]
    An accessible review in the context of lattice gauge theory is given by
    E. Rinaldi et al [https://arxiv.org/pdf/1901.07519.pdf] in Appendix B.
    The class is useful for using the "optimal" shrinkage parameter, which
    requires knowledge of the underlying samples. The "optimal" parameter is
    not always preferred in practice, since it sometimes chooses what seems
    like an overly aggressive value (e.g., lambda=1, dropping all correlations).
    """
    def __init__(self, samples, bstrap=False):
        """
        Args:
            samples: np.ndarray, assumed to be of shape, e.g., "(nconfigs, nt)"
            bstrap: bool, whether the samples come from bootstrap
        """
        self.bstrap = bstrap
        self.samples = samples
        self.y = self.normalize()
        self.lam = self.compute_optimal_shrinkage()

    def __call__(self, lam=None):
        """
        Computes the shrinkage estimator for the covariance matrix following
        Eqs. (B3) and (B7) of [https://arxiv.org/pdf/1901.07519.pdf].
        Args:
            lambda: float or None, the shrinkage parameter. Default is None,
                in which case the estimated optimal value is used.
        Returns:
            np.ndarray, the shrinkage estimator for the covariance matrix
        """
        mu = 1.0
        if lam is None:
            lam = self.lam
        if (lam < 0) or (lam > 1):
            raise ValueError("Shrinkage parameter must be in (0, 1)")
        ds_tmp = gv.dataset.avg_data(self.samples, bstrap=self.bstrap)
        corr = gv.evalcorr(ds_tmp)
        errs = np.diag(gv.sdev(ds_tmp))
        identity = np.eye(len(corr))
        rho_star = lam*mu*identity + (1 - lam)*corr
        return errs @ rho_star @ errs

    def normalize(self):
        """
        Normalize input data to zero mean and unit standard deviation.
        """
        ds_tmp = gv.dataset.avg_data(self.samples, bstrap=self.bstrap)
        xmean = gv.mean(ds_tmp)
        xerr = gv.sdev(ds_tmp)
        return (self.samples - xmean)/xerr

    def _b2(self, rho):
        """
        Computes the sum in Eq. (B5) of [https://arxiv.org/pdf/1901.07519.pdf].
        """
        n, m = self.y.shape
        if rho.shape != (m, m):
            raise ValueError("Incommensurate y and rho", y.shape, rho.shape)
        result = 0
        for i in range(n):
            for alpha in range(m):
                for beta in range(m):
                    result += (self.y[i, alpha]*self.y[i, beta] - rho[alpha, beta])**2
        result /= n**2
        return result

    def _d2(self, rho, mu=1):
        """
        Computes the sum in Eq. (B6) of [https://arxiv.org/pdf/1901.07519.pdf],
        which is estimates the dispersion of the eigenvalues of the sample
        correlation matrix.
        """
        m, n = rho.shape
        if m != n:
            raise ValueError("rho must be square", y.shape)
        result = 0
        for alpha in range(m):
            for beta in range(m):
                if alpha == beta:
                    result += (rho[alpha, beta] - mu)**2
                else:
                    result += rho[alpha, beta]**2
        return result

    def compute_optimal_shrinkage(self):
        """
        Computes the optimal value for the shrinkage parameter lambda according
        to Eq. (B4) of [https://arxiv.org/pdf/1901.07519.pdf].
        """
        corr = gv.evalcorr(gv.dataset.avg_data(self.samples, bstrap=self.bstrap))
        b2 = self._b2(corr)
        d2 = self._d2(corr)
        return min(b2, d2)/d2
# pylint: enable=invalid-name

if __name__ == '__main__':
    main()
