import numpy as np
from . import visualize as plt

def plot_form_factor_results(form_factor_fit, axarr=None):
    """
    Plots the ratio of the fit to data.
    For good fits, this ratio should statistically consistent with unity.
    """
    nrows = len(form_factor_fit.fitter.models)
    if axarr is None:
        fig, axarr = plt.subplots(nrows=nrows, sharex=True,
                                        figsize=(10, 10))
    if len(axarr) < nrows:
        raise ValueError("Too few rows for plot_results()?")
    fit = form_factor_fit.fits['full']
    for ax, model in zip(axarr, form_factor_fit.fitter.models):
        tag = model.datatag
        ratio = form_factor_fit.ds[tag][model.tfit] / fit.fcn(fit.p)[tag]
        plt.errorbar(ax, x=model.tfit, y=ratio, fmt='.')
        ax.axhline(1.0, ls='--', color='k')
        ax.set_ylabel(f'{tag}')
        ax.set_title('data/fit')
    axarr[-1].set_xlabel('t/a')
    return fig, axarr


def _plot_fit_energies(ax, form_factor_fit, energy_tag, with_priors=True):
    """Plots the fit energies associated with 'energy_tag'."""
    # Plot the fit spectrum
    masses = np.cumsum(form_factor_fit.fits['full'].p[energy_tag])
    colors = plt.color_palette(n_colors=len(masses))
    for idx, mass in enumerate(masses):
        plt.axhline(ax, mass, label=f'Fit: E{idx}',
                        alpha=0.75, color=colors[idx])
    # Overlay the priors
    if with_priors:
        masses = np.cumsum(form_factor_fit.fits['full'].prior[energy_tag])
        print(colors, masses)
        for idx, mass in enumerate(masses):
            plt.axhspan(ax, mass, label=f'Prior: E{idx}',
                            alpha=0.25, color=colors[idx])
    return ax

def _plot_meff(ax, corr):
    """Plots the effective mass and the FastFit guess for the given tag."""
    ax = corr.plot_meff(ax=ax, avg=False, fmt='.', label='Effective mass')
    ax = corr.plot_meff(ax=ax, avg=True, fmt='.',label='Smeared effective mass')
    plt.axhline(ax, corr.fastfit.E, color='k', ls=':', label='FastFit guess')
    return ax


def plot_energy_summary(ax, form_factor_fit, tag, osc=False, with_priors=True):
    """ Makes summary plot of the energies."""
    if osc:
        energy_tag = f'{tag}:dEo'
        title = f'Energies (oscillating states): {tag}'
    else:
        energy_tag = f'{tag}:dE'
        title = f'Energies: {tag}'
        # Effective mass defined for decaying states only
        _plot_meff(ax, form_factor_fit.ds[tag])
    _plot_fit_energies(ax, form_factor_fit, energy_tag, with_priors)
    ax.set_title(title)
    ax.set_xlabel("$t/a$")
    ax.set_ylabel("$Ea$")
    ax.legend(loc=1)
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
    # stop around halfway, since we neglect backward propagation
    tmax = min(corr.times.nt // 2, max(t))
    plt.errorbar(ax, t[:tmax], y[:tmax],
                        fmt='.', label='Effective amplitude')
    # Fastfit guess
    amp_ffit = np.sqrt(corr.fastfit.ampl)
    plt.axhline(ax, amp_ffit, label='ffit guess', color='k', ls=':')
    return ax

def _plot_fit_amplitudes(ax, form_factor_fit, amp_tag, with_priors=True):
    """Plots the fit energies associated with 'energy_tag'."""
    # Plot the fit amplitudes
    amps = form_factor_fit.fits['full'].p[amp_tag]
    colors = plt.color_palette(n_colors=len(amps))
    for idx, amp in enumerate(amps):
        label = f'Fit: A{idx}'
        plt.axhline(ax, amp, label=label, color=colors[idx])
    # Overlay the priors
    if with_priors:
        amps = form_factor_fit.fits['full'].prior[amp_tag]
        for idx, amp in enumerate(amps):
            label = f'Prior: A{idx}'
            plt.axhspan(ax, amp, label=label,
                                alpha=0.25, color=colors[idx])
    return ax

def plot_amplitude_summary(ax, form_factor_fit, tag, osc=False, with_priors=True):
    """ Make summary plot of the amplitudes"""
    if osc:
        amp_tag = f'{tag}:ao'
        title = f'Partner Amplitudes: {tag}'
    else:
        amp_tag = f'{tag}:a'
        title = f'Amplitudes: {tag}'
        # Effective amplitude defined for decaying states only
        ax = _plot_amp_eff(ax, form_factor_fit.ds[tag])

    _plot_fit_amplitudes(ax, form_factor_fit, amp_tag, with_priors)
    ax.set_title(title)
    ax.set_xlabel("$t/a$")
    ax.set_ylabel("Amplitude (lattice units)")
    ax.legend(loc=1)
    return ax

def plot_states(form_factor_fit, axarr=None, osc=False, with_priors=True):
    """Plots a 2x2 summary of the masses and amplitudes."""
    if axarr is None:
        fig, axarr = plt.subplots(nrows=2, ncols=2,
                                        sharex=True, figsize=(20, 20))
    ((ax1, ax2), (ax3, ax4)) = axarr
    tags = form_factor_fit.ds._tags
    # Masses in first row
    for ax, tag in zip([ax1, ax2], tags):
        _ = plot_energy_summary(ax, form_factor_fit, tag, osc=osc,
                                        with_priors=with_priors)
    # Amplitudes in second row
    for ax, tag in zip([ax3, ax4], tags):
        _ = plot_amplitude_summary(ax, form_factor_fit, tag, osc=osc,
                                        with_priors=with_priors)
    # Bands for fit range
    ax_cols = [(ax1, ax3), (ax2, ax4)]
    for tag, ax_col in zip(tags, ax_cols):
        for ax in ax_col:
            tmin = form_factor_fit.ds[tag].times.tmin
            tmax = form_factor_fit.ds[tag].times.tmax
            plt.axvline(ax, tmin, color='k', ls='--')
            plt.axvline(ax, tmax, color='k', ls='--')
    fig.tight_layout()
    return axarr

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
