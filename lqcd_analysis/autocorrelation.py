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
        # Integrated autocorrelation time
        tau = 1 + 2*np.cumsum(acorr)

        if acorr[1] < 0.05:
            tau_final = 1
        else:
            stop = np.argmax(np.arange(len(tau)) >= (5*tau))
            tau_final = tau[stop]
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

        # Plot the integrated autocorrelation time
        plt.errorbar(ax2, x, tau[:100], fmt='.')
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
