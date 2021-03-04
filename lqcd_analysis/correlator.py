"""
TwoPoint
ThreePoint
effective_mass
fold
BaseCorrelator
"""
import pathlib
import logging
import numpy as np
import gvar as gv
from . import fastfit
from . import visualize as plt
from . import utils

LOGGER = logging.getLogger(__name__)

def main():
    pass


def effective_mass_local(data, ti=0, dt=1):
    """
    Computes a "local" effective mass.
    Args:
        data: array or list, correlator data C(t)
        ti: int, the starting timeslice. Default is 0.
        dt: int, the step between timeslices. Default is 1.
    Returns:
        array, the effective mass
    Notes:
    The standard "local" effective mass use the logarithm of the ratio of C(t)
    on consecutive time slices: meff = log[C(t)/C(t+1)].
    A trivial generalization works with C(t) on time slices separated by dt:
    meff = (1/dt) * log[C(t)/C(t+dt)]. Choosing this form and using dt=2 is
    particularly useful when working with staggered fermions, since
    contamination from opposite-parity states enters with phase (-1)^t. That is,
    such contributions change sign every other timeslice. In this case, it is
    also useful to construct effective masses using even or odd timeslices only.

    Examples:
    >>> c2 = get_some_correlator_data(...)
    >>> meff = effective_mass_local(c2)  # The usual local version
    >>> meff_even = effective_mass_local(c2, ti=0, dt=2)  # Even timeslices only
    >>> meff_odd = effective_mass_local(c2, ti=1, dt=2)  # Odd timeslices only
    """
    # tmp = data[ti::dt]
    # c_t = tmp[:-1]
    # c_tpdt = tmp[1:]
    c_t = data[ti::dt][:-1]
    c_tpdt = data[ti::dt][1:]
    return np.array((1/dt)*np.log(c_t/c_tpdt))


def effective_mass(data):
    """
    Computes the effective mass analytically using the following formula
    (which I first learned about from Evan Weinberg):

    meff = ArcCosh( (C(t+1)+ C(t-1)) / C(t) )

    This method correctly accounts for contributions both from forward- and
    backward-propagating states. It also work without modification both for
    Cosh-like and Sinh-like correlators.
    """
    cosh_m = (data[2:] + data[:-2]) / (2.0 * data[1:-1])
    meff = np.zeros(len(cosh_m), dtype=gv._gvarcore.GVar)
    # The domain of arccosh is [1, Infinity).
    # Set entries outside of the domain to nans.
    domain = (cosh_m > 1)
    meff[domain] = np.arccosh(cosh_m[domain])
    meff[~domain] = gv.gvar(np.nan)
    return meff
#     # TODO: Handle possible "RuntimeWarning: invalid value encountered in sqrt"
#     return np.arccosh((data[2:] + data[:-2]) / (2.0 * data[1:-1]))


def _infer_tmax(ydata, noise_threshy):
    """Infer the maximum time with noise-to-signal below a threshold."""
    if noise_threshy is None:
        return len(ydata) - 1
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

    @property
    def tdata_avg(self):
        """Fetch tdata safe for use with averaging functions."""
        # return np.arange(self.tmin, self.tmax - 2)
        return np.arange(len(self.tdata) - 2)

    def __str__(self):
        return "BaseTimes(tmin={0},tmax={1},nt={2},tp={3})".\
            format(self.tmin, self.tmax, self.nt, self.tp)

    def __repr__(self):
        return self.__str__()

class TwoPoint(object):
    """TwoPoint correlation function."""
    def __init__(self, tag, ydata, noise_threshy=0.03, skip_fastfit=False, **time_kwargs):
        self.tag = tag
        self.ydata = ydata
        self.noise_threshy = noise_threshy
        tdata = time_kwargs.pop('tdata', None)
        if tdata is None:
            tdata = np.arange(len(ydata))
        self.times = BaseTimes(tdata=tdata, **time_kwargs)
        self.times.tmax = _infer_tmax(ydata, noise_threshy)
        # Estimate the ground-state energy and amplitude
        if skip_fastfit:
            self.fastfit = None
        else:
            self.fastfit = fastfit.FastFit(
                data=self.ydata[:self.times.tmax],
                tp=self.times.tp,
                tmin=self.times.tmin)
        self._mass = None

    def set_mass(self, mass):
        self._mass = mass

    @property
    def mass(self):
        """Estimate the mass using fastfit."""
        if self._mass is not None:
            return self._mass
        if self.fastfit is not None:
            return self.fastfit.E
        msg = (
            "Missing mass. No FastFit result, "
            "and mass hasn't been set externally."
        )
        raise AttributeError(msg)

    @property
    def mass_avg(self):
        """Estimate the mass using fastfit on the averaged correlator."""
        if self._mass is not None:
            return self._mass
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
        if self._mass is not None:
            mass = self._mass
        elif mass < 0:
            mass = self.fastfit.E
        c2 = self.ydata
        tmax = len(c2)
        c2_tp1s = np.roll(self.ydata, -1, axis=0)
        c2_tp2s = np.roll(self.ydata, -2, axis=0)
        # pylint: disable=protected-access
        c2bar = np.empty((tmax,), dtype=gv._gvarcore.GVar)
        # pylint: enable=protected-access
        for t in range(tmax):
            c2bar[t] = c2[t] / np.exp(-mass * t)
            c2bar[t] += 2 * c2_tp1s[t] / np.exp(-mass * (t + 1))
            c2bar[t] += c2_tp2s[t] / np.exp(-mass * (t + 2))
            c2bar[t] *= np.exp(-mass * t)
        return c2bar / 4.

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

    def plot_corr(self, ax=None, avg=False, **kwargs):
        """Plot the correlator on a log scale."""
        if ax is None:
            _, ax = plt.subplots(1, figsize=(10, 5))
        if avg:
            y = self.avg()
            x = self.times.tfit[1:-1]
        else:
            y = self.ydata
            x = self.times.tdata
        ax = plt.mirror(ax=ax, x=x, y=y, **kwargs)
        ax.set_yscale('log')
        return ax

    def plot_meff(self, ax=None, avg=False, a_fm=None, **kwargs):
        """Plot the effective mass of the correlator."""
        if ax is None:
            _, ax = plt.subplots(1)
        if avg:
            y = effective_mass(self.avg())
            x = self.times.tdata[1:-1]
        else:
            y = effective_mass(self.ydata)
            x = self.times.tdata[1:-1]
        if a_fm is not None:
            y = y * 197 / a_fm
        plt.errorbar(ax, x, y, **kwargs)
        return ax

    def plot_n2s(self, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots(1)
        y = self.ydata
        y = gv.sdev(y) / gv.mean(y) * 100
        x = self.times.tdata
        ax = plt.mirror(ax=ax, x=x, y=y, **kwargs)
        ax.set_yscale("log")
        ax.set_ylabel("n/s [%]")
        ax.set_xlabel("t/a")
        return ax

    def plot_summary(self, axarr=None, a_fm=None, avg=False, label=None):
        """
        Plot a brief summary of the correlator as a row
        of three figures:
        [Eorrelator C(t)] [Effective mass] [Noise-to-signal]
        Args:
            axarr: optional array of axes on which to plot
            a_fm: float, the lattice spacing for conversion to MeV
        Returns:
            axarr: array of axes
        """
        if axarr is None:
            _, axarr = plt.subplots(ncols=3, figsize=(15, 5))
        ax1, ax2, ax3 = axarr

        self.plot_corr(ax=ax1, label=label)
        self.plot_meff(ax=ax2, a_fm=a_fm, fmt='o', avg=avg, label=label)
        self.plot_n2s(ax=ax3, label=label)

        ax1.set_title("Correlator C(t)")
        ax2.set_title("Effective mass")
        ax3.set_title("Noise-to-signal [%]")
        return axarr

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
        return "ThreePoint[tag='{}', tmin={}, tmax={}, nt={}, t_snks={}]".\
            format(self.tag, self.times.tmin, self.times.tmax,
                   self.times.nt, sorted(list(self.t_snks)))

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
        Computes a time-slice-averaged three-point correlation function.
        Generalizes Eq. 38 of Bailey et al PRD 79, 054507 (2009)
        [https://arxiv.org/abs/0811.3640] to work for non-adjacent sink times T.
        This average is useful for suppressing contamination from
        opposite-parity states.
        Args:
            m_src, m_snk: the masses (or energy) of the ground states
                associated with the source at time t=0 and and the sink at
                time t=t_snk. The literature often refers to t_snk as "T".
        Returns:
            c3bar: dict of arrays with the time-slice-averaged correlators
        """
        def _combine(ratio):
            """
            Combines according to (R(t) + 2*R(t+1) + R(t+2)) / 4
            """
            return 0.25 * (
                ratio
                + 2.0*np.roll(ratio, -1, axis=0)
                + np.roll(ratio, -2, axis=0)
            )

        t = np.arange(self.times.nt)
        c3bar = {}
        t_snks = np.sort(np.array(self.t_snks))
        dt_snks = t_snks[1:] - t_snks[:-1]
        # pylint: disable=invalid-name,protected-access
        for dT, T in zip(dt_snks, t_snks):
            c3 = self.ydict[T]  # C(t, T)
            ratio = c3 / np.exp(-m_src*t) / np.exp(-m_snk*(T-t))
            tmp = _combine(ratio)
            # When dT is odd, average the results for T and T+dT.
            # For the case dT=1, this average reduces to the cited equation.
            # When dT is even, just use the result for T by itself
            if bool(dT % 2):
                c3 = self.ydict[T+dT]  # C(t, T+dT)
                ratio = c3 / np.exp(-m_src*t) / np.exp(-m_snk*(T+dT-t))
                tmp = 0.5 * (tmp + _combine(ratio))
            c3bar[T] = tmp * np.exp(-m_src*t) * np.exp(-m_snk*(T-t))
        # pylint: enable=invalid-name,protected-access
        return c3bar

    @property
    def t_snks(self):
        return list(self.keys())

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
