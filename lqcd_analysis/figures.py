import numpy as np
from collections import namedtuple
import seaborn as sns
from . import visualize as plt


def plot_form_factor_results(form_factor_fit, ax=None):
    """
    Plots the ratio of the fit to data.
    For good fits, this ratio should statistically consistent with unity.
    """
    if ax is None:
        _, ax = plt.subplots(1, figsize=(10, 5))
    fit = form_factor_fit.fits['full']
    yticks = []
    yticklabels = []
    for idx, model in enumerate(form_factor_fit.fitter.models):
        offset = idx*0.5
        tag = model.datatag
        ratio = form_factor_fit.ds[tag][model.tfit] / fit.fcn(fit.p)[tag]
        y = ratio + offset
        kwargs = {'markeredgewidth':2, 'capsize':5, 'fmt':'.',}
        plt.errorbar(ax=ax, x=model.tfit, y=y, **kwargs)
        ax.axhline(y=1+offset, color='k')
        yticks.append(1+offset)
        yticklabels.append(tag)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_title('data/fit')
    ax.set_xlabel('t/a')
    return ax


def _plot_meff(ax, corr, a_fm=None):
    """
    Plots the effective mass and the FastFit guess for the given tag.
    """
    ax = corr.plot_meff(ax=ax, avg=False, fmt='.', a_fm=a_fm,
                        label='Effective mass')
    ax = corr.plot_meff(ax=ax, avg=True, fmt='.', a_fm=a_fm,
                        label='Smeared effective mass')
    y = corr.fastfit.E
    if y is not None:
        if a_fm is not None:
            y = y * 197 / a_fm
        plt.axhline(ax, y, color='k', ls=':', label='FastFit guess')
    return ax


def _plot_amp_eff(ax, corr):
    """
    Plots the effective amplitude, which (neglecting the
    backward-propagating state) is given by: A_eff^2 = C(t)*Exp(m_eff*t).
    Note that the effective mass function combines adjacent time slices and
    so takes [tmin, tmin+1, ..., tmax-1, tmax] --> [tmin+1, ..., tmax-1],
    which explains the slicing below.
    """
    meff = corr.meff(avg=True)
    t = corr.times.tdata[1:-1]
    y = np.sqrt((np.exp(meff * t)) * corr.avg()[1:-1])
    energy = corr.fastfit.E
    ampl = corr.fastfit.ampl
    if energy:
        y = y * np.sqrt(2.0 * energy)
        # stop around halfway, since we neglect backward propagation
        tmax = min(corr.times.nt // 2, max(t))
        plt.errorbar(ax, t[:tmax], y[:tmax],
                     fmt='.', label='Effective amplitude')
    if energy and ampl:
        # fastfit amplitude is the coefficient in A * exp(-m*t)
        # The overlap is <0|O|state> = Sqrt(A) * Sqrt(2*E)
        overlap = np.sqrt(ampl) * np.sqrt(2.0 * energy)
        plt.axhline(ax, overlap, label='ffit guess', color='k', ls=':')
    return ax


def plot_states(form_factor_fit, nstates, a_fm=None):
    """
    Plots a grid summarizing the eneriges and amplitudes, comparing the prior
    and posterior values along with the effective "mass" and amplitude. The 
    energy is on the left, the amplitude is on the right.
    """
    fit = form_factor_fit.fits['full']
    fig, axarr = plot_comparison(nstates, fit.prior, fit.p, a_fm)
    def overlay(ax, corr, plot_func, **kwargs):
        # grab limits *before* plotting on twin axis
        ylim = ax.get_ylim()  
        ax_twin = ax.twiny()
        plot_func(ax_twin, corr, **kwargs)
        # match to original ylim
        ax.set_ylim(*ylim)
        # shift to right side
        _, xmax = ax_twin.get_xlim()
        ax_twin.set_xlim(-xmax, xmax)
        # sensible x-axis on the right side
        ax_twin.set_xticks(np.arange(0, int(xmax), 5))
        ax_twin.set_xticklabels(np.arange(0, int(xmax), 5))
        ax_twin.legend(loc=1)
        return ax_twin
    channels = get_channels(nstates)
    nrows = len(channels)
    if nrows == 4:
        (ax1, ax2), _, (ax3, ax4), _ = axarr
    elif nrows == 3:
        (ax1, ax2), (ax3, ax4), _ = axarr
    else:
        raise ValueError("Unrecognized number of channels?")
    overlay(ax1, form_factor_fit.ds.c2_src, _plot_meff, a_fm=a_fm)
    overlay(ax2, form_factor_fit.ds.c2_src, _plot_amp_eff)
    overlay(ax3, form_factor_fit.ds.c2_snk, _plot_meff, a_fm=a_fm)
    overlay(ax4, form_factor_fit.ds.c2_snk, _plot_amp_eff)
    fig.tight_layout()
    return fig, axarr


def plot_form_factor(form_factor_fit, ax=None, tmax=None, color='k', prior=True):
    """
    Plot the ratio which delivers the form factor together
    with the prior estimate and fit result.
    """
    # TODO: Handle possibility of nontrivial normalization factor
    # norm = self.ds.normalization()
    norm = 1.0 * form_factor_fit.ds.sign
    # Plot ratio "R" of two- and three-point functions
    ax = form_factor_fit.ds.plot_ratio(ax=ax, tmax=tmax)
    # Plot the prior value for the plateau in "R"
    plt.axhspan(ax, y=form_factor_fit.r_prior * norm,
                    alpha=0.25, color=color, label='Prior: R')
    # Plot the fit value for the plateau in "R"
    plt.axhline(ax, y=form_factor_fit.r * norm,
                        alpha=0.50, color=color, label='Fit: R')
    ax.set_title("Form factor compared with estimates")
    ax.legend(loc=1)
    return ax


Channel = namedtuple("Channel", ["prefix", "suffix", "title"])


def get_channels(nstates):
    """
    Gets a list of "channels" to plot.
    """
    channels = []
    if bool(nstates.n):
        channels.append(Channel('light-light', '', 'Decaying Pion'))
    if bool(nstates.no):
        channels.append(Channel('light-light', 'o', 'Oscillating Pion'))
    if bool(nstates.m):
        channels.append(Channel('heavy-light', '', 'Decaying D-meson'))
    if bool(nstates.mo):
        channels.append(Channel('heavy-light', 'o', 'Oscillating D-meson'))
    return channels


def add_comparison(ax, prior, posterior, a_fm=None):
    """
    Adds a comparison plot of the prior vs the posterior to the given axis.
    """
    colors = sns.color_palette()
    # Left side
    for idx, value in enumerate(prior):
        y = value
        if a_fm is not None:
            y = y * 197 / a_fm
        plt.axhline(ax=ax, y=y, xmin=0.0, xmax=0.5,
                    color=colors[idx], label=f"State {idx}")
    # Right side
    for idx, value in enumerate(posterior):
        y = value
        if a_fm is not None:
            y = y * 197 / a_fm
        plt.axhline(ax=ax, y=y, xmin=0.5, xmax=1.0,
                          color=colors[idx])
    # Formatting
    ax.axvline(0.5, color='k')    
    ax.set_xticks([0.25, 0.75])
    ax.set_xticklabels(['Prior', 'Posterior'])
    ax.legend(loc=2)


def plot_comparison(nstates, prior, posterior, a_fm=None):    
    """
    Plots the prior vs the posterior values for the energies and amplitudes.
    The resulting figure has two columns of plots, with energies on the left and
    amplitudes on the right. In individual plots, the prior is on the left and
    the posterior is on the right.
    """
    channels = get_channels(nstates)
    nrows = len(channels)    
    fig, axarr = plt.subplots(nrows=nrows, ncols=2, figsize=(20, 10*nrows))
    
    for channel, ax_row in zip(channels, axarr):
        axL, axR = ax_row
        key_dE = f'{channel.prefix}:dE{channel.suffix}'
        key_amp = f'{channel.prefix}:a{channel.suffix}'            
        # Left: Energy
        energy_posterior = np.cumsum(posterior[key_dE])
        energy_prior = np.cumsum(prior[key_dE])
        add_comparison(axL, energy_prior, energy_posterior, a_fm)
        # Right: Amplitude
        amp_posterior =  posterior[key_amp] * np.sqrt(2.0 * energy_posterior) 
        amp_prior = prior[key_amp] * np.sqrt(2.0 * energy_prior)
        add_comparison(axR, amp_prior, amp_posterior)
        # Formatting
        axL.set_title(f"{channel.title}: Energies")
        axR.set_title(f"{channel.title}: Amplitudes")
        axL.set_ylabel("Energy (MeV)")
        axR.set_ylabel(r"$A\times \sqrt{2 E}$ (lattice units)")

    return fig, axarr
