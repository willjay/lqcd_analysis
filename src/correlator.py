"""
TwoPoint
ThreePoint
effective_mass
fold
BaseCorrelator
"""
import numpy as np
import gvar as gv
import pylab as plt
from . import fastfit
from . import visualize


def main():
    pass


def effective_mass(data):
    """
    Computes the effective mass analytically using the following formula
    (which I first learned about from Evan Weinberg):

    meff = ArcCosh( (C(t+1)+ C(t-1)) / C(t) )

    This method correctly accounts for contributions both from forward- and
    backward-propagating states. It also work without modification both for
    Cosh-like and Sinh-like correlators.
    """
    # TODO: Handle possible "RuntimeWarning: invalid value encountered in sqrt"
    return np.arccosh((data[2:] + data[:-2]) / (2.0 * data[1:-1]))


def _infer_tmax(ydata, noise_threshy):
    """Infer the maximum time with noise-to-signal below a threshold."""
    good = gv.sdev(ydata) / gv.mean(ydata) < noise_threshy
    if np.all(good):
        tmax = len(ydata) - 1
    else:
        tmax = np.argmin(good)
    return tmax


class BaseTimes(object):
    """
    Basic container for holding times associated with correlation functions.
    """
    def __init__(self, tdata, tmin=5, tmax=None, nt=None, tp=None):
        self.tdata = np.asarray(tdata)
        if tmin < 0:
            raise ValueError('bad tmin')
        self.tmin = tmin

        if tmax is None:
            self.tmax = len(tdata) - 1
        else:
            if tmax > len(tdata):
                raise ValueError('bad tmax')
            self.tmax = tmax

        if nt is None:
            self.nt = len(tdata)
        else:
            self.nt = nt

        if tp is None:
            self.tp = self.nt
        else:
            self.tp = tp

    @property
    def tfit(self):
        """Get fit times."""
        return self.tdata[self.tmin:self.tmax]

    def __str__(self):
        return "BaseTimes(tmin={0},tmax={1},nt={2},tp={3})".\
            format(self.tmin, self.tmax, self.nt, self.tp)


class TwoPoint(object):
    """TwoPoint correlation function."""
    def __init__(self, tag, ydata, noise_threshy=0.03):

        self.tag = tag
        self.ydata = ydata
        self.noise_threshy = noise_threshy
        self.times = BaseTimes(tdata=np.arange(len(ydata)))
        self.times.tmax = _infer_tmax(ydata, noise_threshy)

        # Estimate the ground-state energy and amplitude
        self.fastfit = fastfit.FastFit(
            data=self.ydata[:self.times.tmax],
            tp=self.times.tp,
            tmin=self.times.tmin)

    @property
    def mass(self):
        """Estimate the mass using fastfit."""
        return self.fastfit.E

    @property
    def mass_avg(self):
        """Estimate the mass using fastfit on the averaged correlator."""
        return fastfit.FastFit(
            data=self.avg()[:self.times.tmax],
            tp=self.times.tp,
            tmin=self.times.tmin).E

    def meff(self, avg=False):
        """Compute the effective mass of the correlator."""
        if avg:
            return effective_mass(self.avg())
        else:
            return effective_mass(self.ydata)

    def avg(self, mass=-1.0):
        """
        Compute the time-slice-averaged two-point correlation function.
        This average is useful for suppressing contamination from
        opposite-parity states.
        Follows Eq. 37 of Bailey et al PRD 79, 054507 (2009)
        [https://arxiv.org/abs/0811.3640].
        Other similar quantities could be reasonably defined.
        """
        if mass < 0:
            mass = self.fastfit.E
        c2 = self.ydata
        t = np.arange(self.times.tmin, self.times.tmax - 2)
        c2bar = c2[t] / np.exp(-mass * t)
        c2bar += 2.0 * c2[t + 1] / np.exp(-mass * (t + 1))
        c2bar += c2[t + 2] / np.exp(-mass * (t + 2))
        c2bar *= np.exp(-mass * t)
        c2bar /= 4.0
        return c2bar

    def __getitem__(self, key):
        return self.ydata[key]

    def __setitem__(self, key, value):
        self.ydata[key] = value

    def __len__(self):
        return len(self.ydata)

    def __str__(self):
        return "TwoPoint[tag='{}', tmin={}, tmax={}, nt={}, mass={}]".\
            format(self.tag, self.times.tmin, self.times.tmax, self.times.nt,
                   self.mass)

    def plot_corr(self, ax=None, avg=False):
        """Plot the correlator on a log scale."""
        if ax is None:
            _, ax = plt.subplots(1, figsize=(10, 5))
        if avg:
            y = self.avg()
            x = self.times.tfit[1:-1]

        else:
            y = self.ydata
            x = self.times.tdata
        visualize.errorbar(ax, x, y, fmt='.', marker='o')
        ax.set_yscale('log')
        return ax

    def plot_meff(self, ax=None, avg=False, **kwargs):
        """Plot the effective mass of the correlator."""
        if ax is None:
            _, ax = plt.subplots(1)
        if avg:
            y = effective_mass(self.avg())
            x = self.times.tfit[2:-2]
            visualize.errorbar(ax, x, y, **kwargs)
        else:
            y = effective_mass(self.ydata)
            x = self.times.tdata[1:-1]
            visualize.errorbar(ax, x, y, **kwargs)
        return ax


class ThreePoint(object):
    """ThreePoint correlation function."""
    def __init__(self, tag, ydict, noise_threshy=0.03):
        self.tag = tag
        self.ydict = ydict
        self._verify_ydict()
        self.noise_threshy = noise_threshy
        tmp = list(self.values())[0]  # safe since we verified ydict
        self.times = BaseTimes(tdata=np.arange(len(tmp)))
        self.times.tmax = _infer_tmax(tmp, noise_threshy)

    def __str__(self):
        return "ThreePoint[tag='{}', tmin={}, tmax={}, nt={}]".\
            format(self.tag, self.times.tmin, self.times.tmax, self.times.nt)

    def _verify_ydict(self):
        for t_sink in self.ydict:
            if not isinstance(t_sink, int):
                raise TypeError("t_sink keys must be integers.")
        try:
            np.unique([len(arr) for arr in self.ydict.values()]).item()
        except ValueError as _:
            raise ValueError("Values in ydict must have same length.")

    def avg(self, m_src, m_snk):
        """
        Computes the time-slice-averaged three-point correlation function
        according to Eq. 38 of Bailey et al PRD 79, 054507 (2009)
        [https://arxiv.org/abs/0811.3640]. This average is useful for
        suppressing contamination from opposite-parity states.
        Args:
            m_src, m_snk: the masses (or energy) of the ground states
                associated with the source at time t=0 and and the sink at
                time t=t_snk. The literature often refers to t_snk as "T".
        Returns:
            c3bar: dict of arrays with the time-slice-averaged correlators
        """
        c3bar = {}
        t = np.arange(self.times.tmin, self.times.tmax - 2)
        # Note t_snk
        for t_snk in self.ydict:
            if t_snk + 1 not in self.ydict:
                # Need T and T+1, skip if missing
                continue
            c3 = self.ydict[t_snk]   # C(t,  t_snk)
            c3_t_snk_p1 = self.ydict[t_snk + 1]  # C(t,  t_snk+1)

            c3bar[t_snk] = c3[t] * np.exp(-m_snk * (t_snk - t)) / \
                np.exp(-m_src * t)
            c3bar[t_snk] += c3_t_snk_p1[t] / \
                (np.exp(-m_src * t) * np.exp(-m_snk * (t_snk + 1 - t)))
            c3bar[t_snk] += 2. * c3[t + 1] /\
                (np.exp(-m_src * (t + 1)) * np.exp(-m_snk * (t_snk - (t + 1))))
            c3bar[t_snk] += 2. * c3_t_snk_p1[t + 1] /\
                (np.exp(-m_src * (t + 1)) * np.exp(-m_snk * (t_snk - t)))
            c3bar[t_snk] += c3[t + 2] /\
                (np.exp(-m_src * (t + 2)) * np.exp(-m_snk * (t_snk - (t + 2))))
            c3bar[t_snk] += c3_t_snk_p1[t + 2] /\
                (np.exp(-m_src * (t + 2)) * np.exp(-m_snk * (t_snk - t - 1)))
            c3bar[t_snk] *= np.exp(-m_src * t) * np.exp(-m_snk * (t_snk - t))
            c3bar[t_snk] /= 8.
        return c3bar

    def __getitem__(self, key):
        return self.ydict[key]

    def __setitem__(self, key, value):
        self.ydict[key] = value

    def __len__(self):
        return len(self.ydict)

    def __iter__(self):
        for key in self.keys():
            yield key

    def items(self):
        """items from ydict"""
        return self.ydict.items()

    def keys(self):
        """keys from ydict"""
        return self.ydict.keys()

    def values(self):
        """values from ydict"""
        return self.ydict.values()


if __name__ == '__main__':
    main()
