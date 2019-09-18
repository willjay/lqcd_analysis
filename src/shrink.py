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

    3.  Sch\;afer J., and Strimmer K., "A Shrinkage Approach to Large-Scale
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
import numpy as np


def main():
    print("Some shrinkage routines.")


def rblw_shrink_correlation_identity(samples, verbose=False):
    """
    Shrink the sample correlation matrix using the model T = Identity.
    The Rao-Blackwell Ledoit-Wolf estimator specialized for the case of Gaussian
    noise from Reference [1]. The reference notes that the RBLW estimator
    provably improves on the LW method (i.e. has a smaller min loss function)
    *under the Gaussian noise model*.
    """
    n, p = map(float, samples.shape)
    # S is sample correlation matrix, i.e.,
    # the max likelihood estimator of normalized distribution variance
    S = np.corrcoef(samples, rowvar=0)
    trSsq = np.trace(np.dot(S, S))
    Lambda = ((n - 2.) / n * trSsq + p**2) / ((n + 2.) * (trSsq - p))
    Lambda = min(Lambda, 1.)
    if verbose:
        print('Rao-Blackwell Ledoit-Wolf data correlation matrix shrinkage.')
        print('Lambda:', Lambda)
    S *= 1.0 - Lambda
    np.fill_diagonal(S, 1.0)
    return (Lambda, S)


def oa_shrink_correlation_identity(samples, verbose=False):
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
    if verbose:
        print('Oracle Approximating data correlation matrix shrinkage.')
        print('Lambda: {0}'.format(lam))

    s *= 1.0 - lam
    np.fill_diagonal(s, 1.0)
    return (lam, s)


def lw_shrink_correlation_identity(samples, verbose=False):
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
        pass
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
            pass
        pass
    f *= 2.0
    # diagonal
    for i in range(p):
        xi = d[:, i]
        f += np.sum((xi * xi - s[i, i])**2)
        pass
    tr_s_sq = np.trace(np.dot(s, s))
    lam = f / (n**2 * (tr_s_sq - p))
    lam = max(min(lam, 1.0), 0.0)
    if verbose:
        print('Ledoit-Wolf data correlation matrix shrinkage.')
        print('Lambda: {0}'.format(lam))
        pass
    s *= 1.0 - lam
    np.fill_diagonal(s, 1.0)
    # print S
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
                pass
            pass
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
        pass
    return v


def direct_nl_shrink(sample_ev, n):
    # np.set_printoptions(precision=4,linewidth=150)
    isDesending = np.all(sample_ev[:-1] >= sample_ev[1:])
    if isDesending:
        sam_ev = np.flip(sample_ev, 0)
    else:
        sam_ev = sample_ev
        pass
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
        np.sqrt(np.maximum(0., 4 * np.square(Lt) * hsq - np.square(L -Lt))) /
        (2 * np.pi * np.square(Lt) * hsq), axis=1)
    # Eq.(5.3)
    Hftilde = np.mean(
        (np.sign(L - Lt)\
         * np.sqrt(np.maximum(0., np.square(L - Lt) - 4 * np.square(Lt) * hsq))\
         - L + Lt) / (2 * np.pi * np.square(Lt) * hsq),
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
        pass
    # Eq.(4.5)
    dhat = _pav(dtilde)  
    if isDesending:
        dhat = np.flip(dhat, 0)
        pass
    return dhat


if __name__ == '__main__':

    main()
