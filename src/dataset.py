"""
jackknife
build_dataset
FormFactorDataset
"""
import collections
import functools
import numpy as np
import gvar as gv
import pylab as plt
import seaborn as sns
from . import shrink
from . import correlator
from . import visualize

Tags = collections.namedtuple('Tags', ['src', 'snk'])

def main():
    """Run the main function."""
    print("functions related to building datasets")


def fold(arr):
    """Fold periodic correlator data."""
    try:
        _, nt = arr.shape
        t = np.arange(nt)
        front = arr[:, :nt / 2 + 1]
        back = arr[:, (nt - t) % nt][:, :nt / 2 + 1]
        new_arr = np.mean([front, back], axis=0)
    except ValueError:
        nt, = arr.shape
        t = np.arange(nt)
        front = arr[:nt / 2 + 1]
        back = arr[(nt - t) % nt][:nt / 2 + 1]
        new_arr = np.mean([front, back], axis=0)
    return new_arr


def avg_bin(arr, binsize=1):
    """
    Average 2-dimensional data into bins along the "nsamples" axis.
    Args:
        arr: 2-dimensional array of shape (nsamples, nt)
        binsize: the size of the bins to use
    Returns:
        2-dimensional array of shape (nbins, nt), where
        "nbins = nsamples / binsize"
    """
    if binsize <= 1:
        # No binning to do
        return arr

    # Bin along each timeslice.
    nsamples, nt = arr.shape
    nbins = np.floor_divide(nsamples, binsize)
    binned = np.zeros((nbins, nt))

    for bin_idx in np.arange(nbins):
        for t_idx in np.arange(nt):
            binned[bin_idx, t_idx] = np.mean(
                arr[bin_idx * binsize:(bin_idx + 1) * binsize, t_idx])
    return binned


def nonlinear_shrink(samples, n_eff, verbose=False):
    """
    Shrink the correlation matrix using direct nonlinear shrinkage.

    Works as a wrapper function for shrink.direct_nl_shrink so that the call
    signature is similar to the linear shrinkage functions.

    Args:
        samples: array, of shape (nsamples, p)
        n_eff: the effective number of samples. Usually n <= nsamples
    Returns:
        array, the shrunken correlation matrix
    """
    if verbose:
        logger.info("[+] Direct nonlinear correlation matrix shrinkage.")
        logger.info("Using effective numer of samples n={0}".format(n_eff))
    corr = gv.evalcorr(gv.dataset.avg_data(samples))
    # Decompose into eigenvalues
    vals, vecs = np.linalg.eig(corr)  # (eigvals, eigvecs)
    # Sort in descending order
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    # Shrink the eigenvalue spectrum
    vals_shrink = shrink.direct_nl_shrink(vals, n_eff)
    # Reconstruct eigenvalue matrix
    corr_shrink = np.einsum("ab,bc,cd",
                            vecs, np.diag(vals_shrink), vecs.transpose())
    # Match output of other shrink functions
    pair = (None, corr_shrink)
    return pair


def decomp_blocks(corr_blocked, ordered_tags, sizes):
    """
    Decompose a blocked "correlation matrix" into a dictionary of
    individual correlation matrices.
    """
    ends = np.cumsum(sizes)
    starts = np.cumsum(sizes) - sizes
    corr_dict = {}
    for start_i, end_i, tag_i in zip(starts, ends, ordered_tags):
        for start_j, end_j, tag_j in zip(starts, ends, ordered_tags):
            corr_dict[(tag_i, tag_j)] =\
                corr_blocked[start_i:end_i, start_j:end_j]
    return corr_dict


def correct_covariance(data, binsize=1, shrink_choice=None, ordered_tags=None):
    """
    Correct the covariance using three steps:
    (a) adjust the size of the diagonal errors (via the variances)
        with "blocking" (a.ka. "binning") in Monte Carlo time,
    (b) adjust the correlations of the *full* dataset with shrinkage, and
    (c) combine the adjusted errors and correlation matrices.
    Args:
        data: dict with the full dataset. Assumes that data has keys
              13, 14, 15, 16, 'heavy-light', and 'light-light'
        binsize: int, the binsize to use. Default is 1 (no binning).
        shrink_choice: str, which shrinkage scheme to use. Default is None
            (no shrinkage). Valid options: 'RBLW', 'OA', 'LW', and 'nonlinear'.
    Returns:
        final_cov: the final correct covariance "matrix" as a dictionary
    """
    if ordered_tags is None:
        ordered_tags = sorted(data.keys())
    # shapes are (n,p), where n is nsamples and p is ndata
    sizes = [data[tag].shape[1] for tag in ordered_tags]

    shrink_fcns = {
        'RBLW': shrink.rblw_shrink_correlation_identity,
        'OA': shrink.oa_shrink_correlation_identity,
        'LW': shrink.lw_shrink_correlation_identity,
        'nonlinear': nonlinear_shrink,
    }

    # Estimate errors from binned variances
    binned_data = {tag: avg_bin(data[tag], binsize) for tag in ordered_tags}
    binned_cov = gv.evalcov(gv.dataset.avg_data(binned_data))
    binned_err = {}
    for key_pair in binned_cov:
        key1, key2 = key_pair
        if key1 == key2:
            binned_err[key1] = np.diag(np.sqrt(np.diag(binned_cov[key_pair])))

    # Estimate correlations from shrunken correlation matrices
    if shrink_choice is None:
        # No shrinkage -- take correlations from full dataset
        corr_shrink = gv.evalcorr(gv.dataset.avg_data(
            {tag: binned_data[tag] for tag in ordered_tags}))
    else:
        # Carry out the desired shrinkage
        samples = np.hstack([data[tag] for tag in ordered_tags])
        kwargs = {'verbose': True}
        if shrink_choice == 'nonlinear':
            kwargs['n'] = samples.shape[0] // binsize
        (_, corr_shrink_concat) = shrink_fcns[shrink_choice](samples, **kwargs)
        corr_shrink = decomp_blocks(corr_shrink_concat, ordered_tags, sizes)

    # Correlate errors according to the shrunken correlation matrix
    final_cov = {}
    for key_l, key_r in corr_shrink:
        final_cov[(key_l, key_r)] = np.einsum("ab,bc,cd",
                                              binned_err[key_l],
                                              corr_shrink[(key_l, key_r)],
                                              binned_err[key_r])
    return final_cov


def build_dataset(data_ind, do_fold=True, binsize=10, shrink_choice=None):
    """
    Builds a correlated dataset, folding periodic correlators,
    binning data, and applying the specified shrinkage to the
    covariance matrix.
    """
    def _correlate(data, **kwargs):
        """Correlates the data, including correction of covariance matrix."""
        mean = gv.mean(gv.dataset.avg_data(data))
        cov = correct_covariance(data, **kwargs)
        return gv.gvar(mean, cov)

    tmp = {}
    for key, value in data_ind.items():
        try:
            if (not isinstance(key, int)) and do_fold:
                # Fold two-point functions
                tmp[key] = fold(value)
            else:
                # Don't fold integer-indexed three-point functions
                tmp[key] = value
        except TypeError:
            print(key, value)
    # Correlate the data, including binning and shrinkage
    ds_binned = _correlate(tmp, binsize=binsize, shrink_choice=shrink_choice)
    return ds_binned


def normalization(ns, momentum, current, energy_src, m_snk):
    """
    Calculate the correct normalization for a form factor.
    See the comments after Eq (2.21) in
    Bailey et al. "|V_ub| from B->pi l nu decays and (2+1)-flavor lattice QCD",
    [https://arXiv:1503.07839].
    """
    if (momentum is None) or (current is None):
        msg = "Unspecified momentum/current. Defaulting to unit normalization."
        logger.info(msg)
        return 1.0

    p_vec = [2.0 * np.pi * float(pj) / ns for pj in momentum.lstrip("p")]
    # Define conventional signs.
    # They depend on the precise definition of the lattice current.
    # In practice, they have been determined empirically to give a positive
    # form factor.
    sign = {
        'S-S': +1.0,
        'V4-V4': -1.0,
        'V1-S': -1.0, 'V2-S': +1.0, 'V3-S': -1.0,
        'T14-V4': +1.0, 'T24-V4': -1.0, 'T34-V4': +1.0,
    }
    momentum_factor = {
        'V1-S': p_vec[0], 'V2-S': p_vec[1], 'V3-S': p_vec[2],
        'T14-V4': p_vec[0], 'T24-V4': p_vec[1], 'T34-V4': p_vec[2],
    }
    norm = 1.0
    if current in sign:
        norm *= sign[current]
    if current in momentum_factor:
        norm /= momentum_factor[current]
    # The tensor form factor has an additional kinematic prefactor, which
    # needs the rest mass of the light meson. So boost it back to rest frame!
    # Note that the heavy meson is already in its rest frame.
    if current in ['T14-V4', 'T24-V4', 'T34-V4']:
        m_src = np.sqrt(energy_src**2.0 - np.dot(p_vec, p_vec))
        tensor_factor = (m_src + m_snk) / np.sqrt(2.0 * m_snk)
        norm *= tensor_factor
    if np.isnan(norm):
        msg = '[-] Normalization factor was a nan. Defaulting back to unity.'
        logger.warning(msg)
        norm = 1.0
    return norm


class FormFactorDataset(object):
    """
    FormFactorDataset
    Args:
        ds: dataset, i.e., a dict of correlated data
        nt: int, the temporal size of the lattice. Default is None (which
            corresponds to nt being inferred from the shape of the data in ds).
        noise_threshy: float, noise-to-signal ratio for cutting on the data.
            Default is 0.03, i.e., 3 percent.
    """
    def __init__(self, ds, noise_threshy=0.03):

        self._tags = Tags(src='light-light', snk='heavy-light')
        self.c2 = {}
        for tag in self._tags:
            self.c2[tag] = correlator.TwoPoint(tag, ds.pop(tag), noise_threshy)
        tag = None
        self.c3 = correlator.ThreePoint(tag, ds, noise_threshy)
        self._verify_tdata()

    def _verify_tdata(self):
        """Verify that all correlators have matching tdata."""
        tdata = [
            self.c2_src.times.tdata,
            self.c2_src.times.tdata,
            self.c3.times.tdata
        ]
        for tdata_i in tdata:
            if not np.all(tdata_i == tdata[0]):
                raise ValueError('tdata does not match across correlators')

    @property
    def _tdata(self):
        """Get tdata from c3, verifying that it is safe to do so."""
        self._verify_tdata()
        return self.c3.times.tdata

    @property
    def tfit(self):
        """
        Compute tfit: restrict to constant separation from src/snk operators.
        """
        tfit = {}
        for tag in self._tags:
            tfit[tag] = np.arange(self.c2[tag].times.tmin,
                                  self.c2[tag].times.tmax + 1)
        for t_snk in self.c3.t_snks:
            tfit[t_snk] = np.arange(self.c2[self._tags.src].times.tmin,
                                    t_snk - self.c2[self._tags.snk].times.tmin)
        return tfit

    @property
    def t_snks(self):
        """Fetch the sink times 't_snk' from the ThreePoint function."""
        return list(self.c3.t_snks)

    @property
    def c2_src(self):
        """Fetch two-point correlator associated with source operator."""
        return self.c2[self._tags.src]

    @property
    def c2_snk(self):
        """Fetch two-point correlator associated with sink operator."""
        return self.c2[self._tags.snk]

    @property
    def m_src(self):
        """Estimate ground-state mass associated with source operator."""
        return self.c2_src.mass

    @property
    def m_snk(self):
        """Estimate ground-state mass associated with sink operator."""
        return self.c2_snk.mass

    @property
    def c2bar(self):
        """Calculate the smeared two-point correlation functions as a dict."""
        return {tag: self.c2[tag].avg() for tag in self.c2}

    @property
    def c2bar_src(self):
        """
        Fetch smeared two-point correlator associated with source operator.
        """
        return self.c2bar[self._tags.src]

    @property
    def c2bar_snk(self):
        """
        Fetch smeared two-point correlator associated with sink operator.
        """
        return self.c2bar[self._tags.snk]

    @property
    def c3bar(self):
        """Fetch smeared three-point correlator."""
        return self.c3.avg(m_src=self.c2_src.mass_avg,
                           m_snk=self.c2_snk.mass_avg)

    @property
    def r(self):
        """Compute the ratio of two- and three-point correlators."""
        return self._ratio(avg=False)

    @property
    def rbar(self):
        """Compute the ratio of smeared two- and three-point correlators."""
        return self._ratio(avg=True)

    @functools.lru_cache()
    def _ratio(self, avg=False):
        """
        Compute a useful ratio of correlation functions.
        Follow Eq. 39 of Bailey et al PRD 79, 054507 (2009)
        [https://arxiv.org/abs/0811.3640].
        This quantity is useful for extracting estimates of form factors.
        When avg=True, the quantity effectively suppresses contamination from
        opposite-parity states.
        """
        c3 = self.c3
        c2_src = self.c2_src
        c2_snk = self.c2_snk
        m_src = self.m_src
        m_snk = self.m_snk
        t = self._tdata
        if avg:
            # Switch to averaged versions of all the quantites
            c3 = self.c3bar
            m_src = c2_src.mass_avg
            m_snk = c2_snk.mass_avg
            c2_src = c2_src.avg()
            c2_snk = c2_snk.avg()
        # Compute the ratio
        r = {}
        for t_snk in c3:
            denom = np.sqrt(
                c2_src[t] * c2_snk[t_snk - t] *
                np.exp(-m_src * t) * np.exp(-m_snk * (t_snk - t))
            )
            tmax = min(len(c3[t_snk]), max(t))
            r[t_snk] = c3[t_snk][:tmax] * np.sqrt(2 * m_src) / denom[:tmax]
        return r

    def keys(self):
        """Get the keys of the two- and three-point correlators."""
        return list(self.c2.keys()) + list(self.c3.keys())

    def values(self):
        """Get the values of the two- and three-point correlators."""
        return [self[key] for key in self.keys()]

    def __getitem__(self, key):
        if key in self.c2.keys():
            return self.c2[key]
        elif key in self.c3.keys():
            return self.c3[key]
        else:
            msg = "Unrecognized key in FormFactorData: {0}".format(key)
            raise KeyError(msg)

    def __iter__(self):
        for key in self.keys():
            yield key

    def estimate_plateau(self):
        """Estimate the value of the plateau."""
        factor = +1.0
        # TODO: handle normalization
        # if normalization() < 0:
        #     factor = -1.0
        plateau = float('-inf')
        for t_snk, rbar in self.rbar.items():
            local_max = max(factor * gv.mean(rbar[1:t_snk - 1]))
            plateau = max(plateau, local_max)
        return plateau

    def plot_corr(self, ax=None):
        """Plot the correlation functions in the dataset."""
        if ax is None:
            _, ax = plt.subplots(1, figsize=(10, 5))
        colors = sns.color_palette()
        # Two-point functions
        for color, tag in zip(colors, self._tags):
            # Raw data, with sawtooth oscillations
            x = self.c2[tag].times.tdata
            y = self.c2[tag]
            visualize.errorbar(ax, x, y, color=colors[0], fmt='.')
            # Averaged data, with suppressed oscillations
            x = self.c2[tag].times.tdata
            y = self.c2[tag].avg()
            visualize.errorbar(ax, x, y, color=colors[0], fmt='-')
        # Three-point functions
        c3 = self.c3
        c3bar = self.c3bar
        for color, t_snk in zip(colors[2:], self.t_snks):
            y = c3[t_snk][:]
            x = c3.times.tdata
            visualize.errorbar(ax, x, y, fmt='.', color=color)
            if t_snk in c3bar:
                y = c3bar[t_snk][:]
                x = c3.times.tdata
                visualize.errorbar(ax, x, y, fmt='-', color=color)

        ax.set_yscale('log')
        ax.set_xlabel('t')
        ax.set_ylabel('C(t)')
        return ax

    def plot_ratio(self, ax=None, tmin=0, tmax=None, bands=False,
                   sign=1.0, plot_sawtooth=True, **plot_kwargs):
        """Plot a useful ratio of two- and three-point functions."""
        if ax is None:
            _, ax = plt.subplots(1, figsize=(10, 5))
        colors = sns.color_palette()
        # TODO: decide how we want to handle normalizations
        # norm = self.normalization()
        if tmax is None:
            tmax = max(self.t_snks)
        t = np.arange(tmin, tmax)
        r = self.r
        rbar = self.rbar
        for color, t_snk in zip(colors, sorted(r)):
            x = t
            if plot_sawtooth:
                # Unsmeared "saw-tooth" ratio
                label = "R, T={0}".format(t_snk)
                y = sign * r[t_snk][t]
                visualize.errorbar(
                    ax, x, y,
                    label=label, color=color, fmt='-.', **plot_kwargs
                )
            if t_snk in self.rbar:
                # Smeared ratio
                label = "Rbar, T={0}".format(t_snk)
                y = sign * rbar[t_snk][t]
                visualize.errorbar(
                    ax, x, y,
                    label=label, color=color, bands=bands, **plot_kwargs
                )

        ax.set_xlabel('t/a')
        ax.set_ylabel('$\\bar{R}$ (lattice units)')
        ax.legend(loc=0)
        return ax


if __name__ == '__main__':
    main()
