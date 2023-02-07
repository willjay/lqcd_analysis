import numpy as np
import gvar as gv
from . import dataset
from . import visualize as plt

def autocorr(x):
    """
    Computes the autocorrelation.
    See: https://en.wikipedia.org/wiki/Autocorrelation
    """
    n = len(x)
    k = np.arange(n)
    mu = np.mean(x)
    var = np.var(x)
    result = np.correlate(x-mu, x-mu, mode='full') / var
    return result[result.size//2:] / (n-k)


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def compute_tau(samples, c=10):
    """
    Computes the integrated autocorrelation time.

    Asymptotically, the autocorrelation function should behave as
    C(t) ~ exp(-t/tau),
    where tau is the longest autocorrelation time in the system.
    Letting t become artibrarily large and focusing on the contribution
    from the slow mode, gives the following result for the integrated
    autocorrelation function:
    \int_{t=0}^{tmax} dt C(t) = tau * (1 - exp(-tmax/tau)).
    For large times, the integrated autocorrelation function should plateau
    at the value of autocorrelation time tau.
    For this reason, the integral is sometimes called the integrated
    autocorrelation time and denoted tau(tmax). Becuase tau(tmax) decays
    exponentially, it tends to become noisy for large tmax.

    See the discussion here https://dfm.io/posts/autocorr/
    This page references lecture notes by A. Sokal.
    The key useful point is to estimate the integrated autocorrelation
    time by picking some t << Nt (N is the number of Monte Carlo samples)
    such that:
        t >= C x tau(t)
        or
        t/tau(t) >= C
    where C is a constant numerically near 5, to be determined empirically.
    """
    acorr = autocorr(samples)
    # if acorr[1] < 0.05:
    #     return 1
    tau = 2*(0.5 + np.cumsum(acorr[1:]))
    tmax = np.arange(len(tau))
    return tau[np.argmax(tmax/tau > c)]


class AutoCorrelation:
    def __init__(self, data):
        """
        Args:
            data: np.ndarray, assumed to have shape (nsamples, nt)
        """
        self.data = data
        self.tau_final = None

    def autocorrelation(self, t, binsize=1):

        binned = dataset.avg_bin(self.data, binsize=binsize)
        # Autocorrelation fucntion
        acorr = autocorr(binned[:, t])
        tau = 2*(0.5 + np.cumsum(acorr[1:]))
        tau_final = compute_tau(binned[:, t])
        # # Integrated autocorrelation time
        # tau = 1 + 2*np.cumsum(acorr)
        # if acorr[1] < 0.05:
        #     tau_final = 1
        # else:
        #     stop = np.argmax(np.arange(len(tau)) >= (5*tau))
        #     tau_final = tau[stop]
        self.tau_final = tau_final
        return acorr, tau, tau_final

    def check_binning(self, t, binsizes=None):
        if binsizes is None:
            binsizes = np.arange(1,100, 2)
        # Statistical errors vs bin size
        x, y = [], []
        for binsize in binsizes:
            binned = dataset.avg_bin(self.data, binsize=binsize)
            ds =gv.dataset.avg_data(dataset.fold(binned))
            x.append(binsize)
            y.append(ds[t])
        y = np.array(y)
        y /= gv.mean(y[0])
        return x, y

    def plot_summary(self, axarr=None, t=10, binsize=1):
        if axarr is None:
            fig, axarr = plt.subplots(ncols=3, figsize=(15, 5))
        ax1, ax2, ax3 = axarr

        acorr, tau, tau_final = self.autocorrelation(t, binsize)

        # Plot the autocorrelation function
        y = acorr[:100]
        x = (binsize * np.arange(len(y)))[:100]
        plt.mirror(ax=ax1, x=x, y=y, label=f"binsize={binsize}")
        ax1.axhline(y=0, color='k', ls='--')
        ax1.legend(loc=0)

        plt.errorbar(ax=ax1, x=x, y=np.exp(-x/tau_final)*acorr[1], fmt='-', color='k')
        plt.errorbar(ax=ax1, x=x, y=np.exp(-x/(0.5*tau_final))*acorr[1], fmt='-', color='orange')

        # Plot the integrated autocorrelation time
        y = np.arange(100)/tau[:100]
        plt.errorbar(ax2, x, y, fmt='.')
        ax2.axhline(y=tau_final, color='k', ls='--')
        ax1.axvline(x=tau_final, color='k', ls='--')

        # Plot the result of the binning test
        x, y = self.check_binning(t)
        plt.errorbar(ax3, x, y, fmt='o')

        # Inflate the unbinned error by sqrt(tau_final) as a check
        y_err = gv.sdev(y[0]) * np.sqrt(tau_final)
        plt.errorbar(ax3, x=[tau_final], y=[gv.gvar(1,y_err)], fmt='o', capsize=5,
             label=r'Error rescale by $\sqrt{\tau}$')

        # ax1.set_title(f"{ens} \n (m1,m2)=({m1},{m2})")
        ax1.set_xlabel(r"Separation in Monte Carlo time $\tau$")
        ax1.set_ylabel(r"Autocorrelation function $C(\tau)$")

        ax2.set_title(r"Integrated Autocorrelation Time")
        ax2.set_ylabel(r"$\sum_\tau^{\tau_{max}} C(\tau)$")
        ax2.set_xlabel(r"$\tau_{max}$")

        ax3.set_title("Statistical errors vs bin size")
        ax3.set_ylabel("Normalized correlation function \n(fixed Euclidean time t)")
        ax3.set_xlabel("Bin size")
        ax3.legend()

        fig.tight_layout()

        return axarr
