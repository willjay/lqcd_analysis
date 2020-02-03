"""
TwoPointAnalysis
FormFactorAnalysis
"""
import logging
import collections
import numpy as np
import gvar as gv
import corrfitter as cf
from . import correlator
from . import dataset
from . import bayes_prior
from . import statistics
from . import visualize

LOGGER = logging.getLogger(__name__)

Nstates = collections.namedtuple(
    'NStates', ['n', 'no', 'm', 'mo'], defaults=(1, 0, 0, 0)
)


def phat2(ptag):
    """
    Strip out the squared momentum $\\hat{p}^2$ from strings like 'p123'.
    """
    return sum([int(pj)**2 for pj in ptag.lstrip("p")])


def p2(ptag, ns):
    """ Convert compute the squared momentum"""
    return phat2(ptag) * (2.0*np.pi/ns)**2.0


def count_nstates(params, key_map=None, tags=None):
    """
    Count the number of states used in fit.
    Default behavior assumes names 'light-light' and 'heavy-light' for src and
    snk, respectively.
    """
    if tags is None:
        tags = dataset.Tags(src='light-light', snk='heavy-light')
    src = tags.src
    snk = tags.snk
    if key_map is None:
        key_map = {
            'n': f'{src}:dE', 'no': f'{src}:dEo',
            'm': f'{snk}:dE', 'mo': f'{snk}:dEo',
        }
    kwargs = {k1: len(params[k2]) for k1, k2 in key_map.items()}
    return Nstates(**kwargs)


def get_two_point_model(two_point, osc=True):

    tag = two_point.tag
    a_pnames = (f'{tag}:a', f'{tag}:ao')
    b_pnames = (f'{tag}:a', f'{tag}:ao')
    dE_pnames = (f'{tag}:dE', f'{tag}:dEo')

    # Handle edge case of no oscillating states
    if not osc:
        a_pnames = a_pnames[0]
        b_pnames = b_pnames[0]
        dE_pnames = dE_pnames[0]

    model = cf.Corr2(
        datatag=tag,
        tp=two_point.times.tp,
        tmin=two_point.times.tmin,
        tmax=two_point.times.tmax,
        tdata=two_point.times.tdata,
        a=a_pnames,
        b=b_pnames,
        dE=dE_pnames,
        s=(1.0, -1.0)
    )
    return model


def get_three_point_model(t_snk, tfit, tdata, nstates, tags=None):
    if tags is None:
        tags = dataset.Tags(src='light-light', snk='heavy-light')
    src = tags.src
    snk = tags.snk
    
    if tfit.size and np.all(np.isin(tfit, tdata)):
        a_pnames = (f'{src}:a', f'{src}:ao')
        b_pnames = (f'{snk}:a', f'{snk}:ao')
        dEa_pnames = (f'{src}:dE', f'{src}:dEo')
        dEb_pnames = (f'{snk}:dE', f'{snk}:dEo')
        vnn = 'Vnn'
        von = 'Von'
        vno = 'Vno'
        voo = 'Voo'

        # Handle edge case of no oscillating states
        if nstates.no == 0:
            a_pnames = a_pnames[0]
            dEa_pnames = dEa_pnames[0]
            von = None
            voo = None
        if nstates.mo == 0:
            b_pnames = b_pnames[0]
            dEb_pnames = dEb_pnames[0]
            vno = None
            voo = None

        model = cf.Corr3(
            datatag=t_snk, T=t_snk, tdata=tdata, tfit=tfit,
            # Amplitudes in src 2-pt function
            a=a_pnames,
            # Amplitudes in snk 2-pt function
            b=b_pnames,
            # Energies in src 2-pt function
            dEa=dEa_pnames,
            # Energies in src 2-pt function
            dEb=dEb_pnames,
            # sign factors in src 2-pt function
            sa=(1.0, -1.0),
            # sign factors in snk 2-pt function
            sb=(1.0, -1.0),
            # connect src decay --> snk decay
            Vnn=vnn,
            # connect src decay --> snk oscillating
            Vno=vno,
            # connect src oscillating --> snk decay
            Von=von,
            # connect src oscillating --> snk oscillating
            Voo=voo
        )
    else:
        # Empty tfit -- no model
        model = None

    return model


def get_model(ds, tag, nstates):
    """Gets a corrfitter model"""
    if isinstance(ds[tag], correlator.TwoPoint):
        osc = bool(nstates.no)
        return get_two_point_model(ds[tag], osc)
    if isinstance(tag, int):
        t_snk = tag
        return get_three_point_model(t_snk, ds.tfit[t_snk], ds.tdata, nstates)
    raise TypeError("get_model() needs TwoPoint or ThreePoint objects.")


def compute_yfit(ds, params):
    """
    Computes the model values "yfit".
    The envisioned usage is that some set of stored fit parameters should be
    used to construct "yfit" for comparison to the data stored in "ds". So
    "yfit" should mirror the structure of "ds".
    Args:
        ds: FormFactorDataset with the data
        params: dict of fit parameters
    Returns:
        yfit: dict
    """
    nstates = count_nstates(params)
    yfit = {}
    for tag in ds:
        model = get_model(ds, tag, nstates)
        tdata = np.array(model.tdata, dtype=float)
        yfit[tag] = model.fitfcn(t=tdata, p=params)
    return yfit


def convert_vnn_to_ratio(m_src, matrix_element):
    """
    Converts the matrix element "Vnn[0,0]" to the quantity "R", a ratio which
    is closely related to the form factors themselves. See Eqs. 7-12 and
    39-41 of J. Bailey et al., Phys.Rev. D79 (2009) 054507
    [https://arxiv.org/pdf/0811.3640] for more context.
    """
    return matrix_element * np.sqrt(2.0 * m_src)


class TwoPointAnalysis(object):
    """
    A basic fitter class for two-point correlation functions.
    Args:
        c2: TwoPoint object
    """

    def __init__(self, c2):
        self.tag = c2.tag
        self.c2 = c2

    def run_fit(self, nstates=Nstates(1, 0), prior=None, **fitter_kwargs):
        """
        Run the fit.
        Args:
            nstates: tuple (n_decay, n_osc) specifying the number of decaying
                and oscillating states, respectively. Defaults to (1,0).
            prior: BasicPrior object. Default is None, for which the fitter
                tries to constuct a prior itself.
        """

        if prior is None:
            prior = bayes_prior.MesonPrior(
                nstates.n, nstates.no, amps=['a', 'ao'],
                tag=self.tag, ffit=self.c2.fastfit,
                extend=True
            )
        # Model construction infers the fit times from c2
        model = get_two_point_model(self.c2, bool(nstates.no))
        fitter = cf.CorrFitter(models=model)
        data = {self.tag: self.c2}
        fit = fitter.lsqfit(
            data=data, prior=prior, p0=prior.p0, **fitter_kwargs
        )
        if np.isnan(fit.chi2):
            fit = None
        return fit


class FormFactorAnalysis(object):

    def __init__(self, ds, positive_ff=True):

        self.ds = ds
        self.positive_ff = positive_ff
        self.prior = None
        self.fits = {}
        self.fitter = None
        self.stats = {}
        self.r = None

    def run_sequential_fits(
            self, nstates, tmin_override=None,
            width=0.1, fractional_width=False,
            prior=None,
            **fitter_kwargs):
        """
        Runs sequential fits.
        First runs two-point functions, whose results are used to update
        the (central values of) the priors for the joint fit. The runs the
        joint fit.
        """
        if prior is None:            
            self.prior = bayes_prior.FormFactorPrior(
                nstates, self.ds, positive_ff=self.positive_ff)
        else:
            self.prior = prior
        if tmin_override is not None:
            if tmin_override.src is not None:
                self.ds.c2_src.times.tmin = tmin_override.src
            if tmin_override.snk is not None:
                self.ds.c2_snk.times.tmin = tmin_override.snk
        self.fit_two_point(
            nstates=nstates,
            width=width,
            fractional_width=fractional_width,
            **fitter_kwargs)
        self.fit_form_factor(nstates=nstates, **fitter_kwargs)
        self.collect_statistics()
        if self.fits['full'] is not None:
            self.r = convert_vnn_to_ratio(self.m_src, self.matrix_element)

    def mass(self, tag):
        """Gets the mass/energy of the ground state from full fit."""
        params = self.fits['full'].p
        return params[f'{tag}:dE'][0]

    @property
    def m_src(self):
        """Gets the mass/energy of the "source" ground state."""
        src_tag = self.ds._tags.src
        return self.mass(src_tag)

    @property
    def m_snk(self):
        """Gets the mass/energy of the "snk" ground state."""
        snk_tag = self.ds._tags.snk
        return self.mass(snk_tag)

    @property
    def matrix_element(self):
        """Fetches the matrix element Vnn[0, 0] needed for the form factor."""
        if self.fits['full'] is not None:
            return self.fits['full'].p['Vnn'][0, 0]

    @property
    def r_prior(self):
        src_tag = self.ds._tags.src
        m_src = gv.mean(self.prior[f'{src_tag}:dE'][0])
        matrix_element = self.prior['Vnn'][0, 0]
        return convert_vnn_to_ratio(m_src, matrix_element)

    @property
    def is_sane(self):
        """
        Checks whether the resulting value of the ratio "R" is sane.
        Assumes that the "guess" value was a lower bound on the value of "R".
        """
        if self.fits['full'] is None:
            return False
        r_fit = np.abs(gv.mean(self.r))
        r_guess = np.abs(gv.mean(self.ds.r_guess))
        return r_fit >= r_guess

    def fit_two_point(self, nstates, width=0.1, fractional_width=False, **fitter_kwargs):
        """Run the fits of two-point functions."""
        for tag in self.ds.c2:
            _nstates = nstates
            if tag == self.ds._tags.snk:
              _nstates = Nstates(n=nstates.m, no=nstates.mo)
            # TODO: handle possible re-running if fit fails initially
            # In the other code, I reset the priors on dEo to match dE
            fit = TwoPointAnalysis(self.ds.c2[tag]).\
                run_fit(_nstates, **fitter_kwargs)
            if fit is None:
                LOGGER.warning('Fit failed for two-point function %s.', tag)
            else:
                self.prior.update(
                    update_with=fit.p, width=width, fractional_width=fractional_width)
            self.fits[tag] = fit

    def fit_form_factor(self, nstates, **fitter_kwargs):
        """Run the joint fit of 2- and 3-point functions for form factor."""
        models = [get_model(self.ds, tag, nstates) for tag in self.ds]
        models = [model for model in models if model is not None]
        if len(models) == len(set(self.ds.keys())):        
            self.fitter = cf.CorrFitter(models=models)
            fit = self.fitter.lsqfit(data=self.ds, prior=self.prior, **fitter_kwargs)
            if np.isnan(fit.chi2) or np.isinf(fit.chi2):
                LOGGER.warning('Full joint fit failed.')
                fit = None
        else:
            self.fitter = None
            fit = None
            LOGGER.warning('Insufficient models found. Skipping joint fit.')
        self.fits['full'] = fit

    def collect_statistics(self):
        """Collect statistics about the fits."""
        for tag, fit in self.fits.items():
            if fit is not None:
                self.stats[tag] = statistics.FitStats(fit)
            else:
                self.stats[tag] = None

    def plot_results(self, axarr=None):
        """
        Plots the result of the fit, taking the ratio of the fit to the data.
        For good fits, this ratio should statistically consistent with unity.
        """
        nrows = len(self.fitter.models)
        if axarr is None:
            fig, axarr = visualize.subplots(nrows=nrows, sharex=True,
                                            figsize=(10, 10))
        if len(axarr) < nrows:
            raise ValueError("Too few rows for plot_results()?")
        fit = self.fits['full']
        for ax, model in zip(axarr, self.fitter.models):
            tag = model.datatag
            ratio = self.ds[tag][model.tfit] / fit.fcn(fit.p)[tag]
            visualize.errorbar(ax, x=model.tfit, y=ratio, fmt='.')
            ax.axhline(1.0, ls='--', color='k')
            ax.set_ylabel(f'{tag}')
            ax.set_title('data/fit')
        axarr[-1].set_xlabel('t/a')
        return fig, axarr

    def _plot_meff(self, ax, tag):
        """Plots the effective mass and the FastFit guess for the given tag."""
        corr = self.ds[tag]
        label = 'Effective mass'
        ax = corr.plot_meff(ax=ax, avg=False, fmt='.', label=label)
        label = 'Smeared effective mass'
        ax = corr.plot_meff(ax=ax, avg=True, fmt='.', label=label)
        label = "FastFit guess"
        visualize.axhline(ax, corr.fastfit.E, label=label, color='k', ls=':')
        return ax

    def _plot_fit_energies(self, ax, energy_tag, with_priors=True):
        """Plots the fit energies associated with 'energy_tag'."""
        # Plot the fit spectrum
        masses = np.cumsum(self.fits['full'].p[energy_tag])
        colors = visualize.color_palette(n_colors=len(masses))
        for idx, mass in enumerate(masses):
            visualize.axhline(ax, mass, label=f'Fit: E{idx}',
                              alpha=0.75, color=colors[idx])
        # Overlay the priors
        if with_priors:
            masses = np.cumsum(self.prior[energy_tag])
            for idx, mass in enumerate(masses):
                visualize.axhspan(ax, mass, label=f'Prior: E{idx}',
                                  alpha=0.25, color=colors[idx])
        return ax

    def plot_energy_summary(self, ax, tag, osc=False, with_priors=True):
        """ Makes summary plot of the energies."""
        if osc:
            energy_tag = f'{tag}:dEo'
            title = f'Energies (oscillating states): {tag}'
        else:
            energy_tag = f'{tag}:dE'
            title = f'Energies: {tag}'
            # Effective mass defined for decaying states only
            self._plot_meff(ax, tag)
        self._plot_fit_energies(ax, energy_tag, with_priors)
        ax.set_title(title)
        ax.set_xlabel("$t/a$")
        ax.set_ylabel("$Ea$")
        ax.legend(loc=1)
        return ax

    def _plot_amp_eff(self, ax, tag):
        """
        Plots the effective amplitude, which (neglecting the
        backward-propagating state) is given by: A_eff^2 = C(t)*Exp(m_eff*t).
        Note that the effective mass function combines adjacent time slices and
        so takes [tmin, tmin+1, ..., tmax-1, tmax] --> [tmin+1, ..., tmax-1],
        which explains the slicing below.
        """
        corr = self.ds[tag]
        meff = corr.meff(avg=True)
        t = corr.times.tdata[1:-1]
        y = np.sqrt((np.exp(meff * t)) * corr.avg()[1:-1])
        # stop around halfway, since we neglect backward propagation
        tmax = min(corr.times.nt // 2, max(t))
        visualize.errorbar(ax, t[:tmax], y[:tmax],
                           fmt='.', label='Effective amplitude')
        # Fastfit guess
        amp_ffit = np.sqrt(corr.fastfit.ampl)
        visualize.axhline(ax, amp_ffit, label='ffit guess', color='k', ls=':')
        return ax

    def _plot_fit_amplitudes(self, ax, amp_tag, with_priors=True):
        """Plots the fit energies associated with 'energy_tag'."""
        # Plot the fit amplitudes
        amps = self.fits['full'].p[amp_tag]
        colors = visualize.color_palette(n_colors=len(amps))
        for idx, amp in enumerate(amps):
            label = f'Fit: A{idx}'
            visualize.axhline(ax, amp, label=label, color=colors[idx])
        # Overlay the priors
        if with_priors:
            amps = self.prior[amp_tag]
            for idx, amp in enumerate(amps):
                label = f'Prior: A{idx}'
                visualize.axhspan(ax, amp, label=label,
                                  alpha=0.25, color=colors[idx])
        return ax

    def plot_amplitude_summary(self, ax, tag, osc=False, with_priors=True):
        """ Make summary plot of the amplitudes"""
        if osc:
            amp_tag = f'{tag}:ao'
            title = f'Partner Amplitudes: {tag}'
        else:
            amp_tag = f'{tag}:a'
            title = f'Amplitudes: {tag}'
            # Effective amplitude defined for decaying states only
            ax = self._plot_amp_eff(ax, tag)

        self._plot_fit_amplitudes(ax, amp_tag, with_priors)
        ax.set_title(title)
        ax.set_xlabel("$t/a$")
        ax.set_ylabel("Amplitude (lattice units)")
        ax.legend(loc=1)
        return ax

    def plot_states(self, axarr=None, osc=False, with_priors=True):
        """Plots a 2x2 summary of the masses and amplitudes."""
        if axarr is None:
            fig, axarr = visualize.subplots(nrows=2, ncols=2,
                                            sharex=True, figsize=(20, 20))
        ((ax1, ax2), (ax3, ax4)) = axarr
        tags = self.ds._tags
        # Masses in first row
        for ax, tag in zip([ax1, ax2], tags):
            _ = self.plot_energy_summary(ax, tag, osc=osc,
                                         with_priors=with_priors)
        # Amplitudes in second row
        for ax, tag in zip([ax3, ax4], tags):
            _ = self.plot_amplitude_summary(ax, tag, osc=osc,
                                            with_priors=with_priors)
        # Bands for fit range
        ax_cols = [(ax1, ax3), (ax2, ax4)]
        for tag, ax_col in zip(tags, ax_cols):
            for ax in ax_col:
                tmin = self.ds[tag].times.tmin
                tmax = self.ds[tag].times.tmax
                visualize.axvline(ax, tmin, color='k', ls='--')
                visualize.axvline(ax, tmax, color='k', ls='--')
        fig.tight_layout()
        return axarr

    def plot_form_factor(self, ax=None, tmax=None, color='k'):
        """
        Plot the ratio which delivers the form factor together
        with the prior estimate and fit result.
        """
        # TODO: Handle possibility of nontrivial normalization factor
        # norm = self.ds.normalization()
        norm = 1.0
        # Plot ratio "R" of two- and three-point functions
        ax = self.ds.plot_ratio(ax=ax, tmax=tmax)
        # Plot the prior value for the plateau in "R"
        visualize.axhspan(ax, y=self.r_prior * norm,
                          alpha=0.25, color=color, label='Prior: R')
        # Plot the fit value for the plateau in "R"
        visualize.axhline(ax, y=self.r * norm,
                          alpha=0.50, color=color, label='Fit: R')
        ax.set_title("Form factor compared with estimates")
        ax.legend(loc=1)
        return ax
