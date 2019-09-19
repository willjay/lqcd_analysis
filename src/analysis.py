"""
TwoPointAnalysis
FormFactorAnalysis
FitStats
"""
import collections
import numpy as np
import corrfitter as cf
from . import correlator
from . import dataset
from . import bayes_prior

Nstates = collections.namedtuple('NStates', ['n', 'no', 'm', 'mo'])

def count_nstates(params, key_map=None, tags=None):
    """
    Count the number of states used in fit.
    Default behavior assumes names using 'light-light' and 'heavy-light'.
    """
    if tags is None:
        tags = dataset.Tags(src='light-light', snk='heavy-light')
    src = tags.src
    snk = tags.snk
    if key_map is None:
        key_map = {
            f'n': '{src}:dE',
            f'no': '{src}:dEo',
            f'm': '{snk}:dE',
            f'mo': '{snk}:dEo',
        }
    kwargs = {k1: len(params[k2]) for k1, k2 in key_map.iteritems()}
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
        tp=two_point.tp,
        tmin=two_point.tmin,
        tmax=two_point.tmax,
        tdata=two_point.tdata,
        a=a_pnames,
        b=b_pnames,
        dE=dE_pnames,
        s=(1.0, -1.0)
    )
    return model


def get_three_point_model(ds, t_sink, nstates, tags=None):
    if tags is None:
        tags = Tags(src='light-light', snk='heavy-light')
    src = tags.src
    snk = tags.snk

    tmin = ds[tags.src].tmin
    tmax = t_sink - ds[tags.snk].tmin
    tfit = np.arange(tmin, tmax)

    if tfit.size:
        tdata = ds.c3.tdata
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
            datatag=t_sink, T=t_sink, tdata=tdata, tfit=tfit,
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

    if isinstance(ds[tag], correlator.TwoPoint):
        if nstates.no:
            osc = True
        else:
            osc = False
        return get_two_point_model(ds[tag], osc)
    if isinstance(tag, int):
        return get_three_point_model(ds, tag, nstates)
    raise TypeError("get_model() needs TwoPoint or ThreePoint objects.")


class TwoPointAnalysis(object):
    """
    A basic fitter class for two-point correlation functions.
    Args:
        c2: TwoPoint object
    """
    def __init__(self, c2):
        
        self.tag = c2.tag
        self.c2 = c2

    def run_fit(self, nstates=(1,0), prior=None, **fitter_kwargs):
        """
        Run the fit.
        Args:
            nstates: tuple (n_decay, n_osc) specifying the number of decaying
                and oscillating states, respectively. Defaults to (1,0).
            prior: BasicPrior object. Default is None, for which the fitter
                tries to constuct a prior itself.
        """
        n_decay, n_osc = nstates
        if prior is None:
            prior = bayes_prior.MesonPrior(
                n_decay, n_osc,
                amps=['a','ao'], tag=self.c2.tag, ffit=self.c2.ffit,
                extend=True
            )

        model = get_two_point_model(self.c2, bool(n_osc))
        fitter = cf.CorrFitter(models=model)
        data = {self.tag : self.c2}
        fit = fitter.lsqfit(
            data=data, prior=prior, p0=prior.p0, **fitter_kwargs
        )
        if np.isnan(fit.chi2):
            fit = None
        return fit


class FormFactorAnalysis(object):

    def __init__(self, ds, times, nstates, positive_ff, **fitter_kwargs):

        self.ds = ds
        self.times = times
        self.nstates = nstates
        
        # Run fits
        self.prior = bayes_prior.FormFactorPrior(nstates, ds=ds, positive_ff=True)
        self.fits = {}
        self.fit_two_point()
        self.fitter = None  # Set in fit_form_factor
        self.fit_form_factor()

        # Collect diagnostic information
        self.stats = {}
        self.collect_statistics()




        self.r_guess = self.prior.r_guess


        def convet_vnn_to_r(self):

            # Convert results about V[0,0] into the ratio "Rbar"
            # which is supposed to deliver the form factor
    #         m_pi    = self.prior.get_gaussian('light-light:dE')[0]
    #         v_guess = self.prior.get_gaussian('Vnn')[0,0]
    #         v_guess = self.prior['Vnn'][0,0]
    #         self.r_guess = v_guess*gv.mean(np.sqrt(2.0*m_pi))

            if self.fits['full'] is not None:

                m_pi = self.fits['full'].p['light-light:dE'][0]
                v = self.fits['full'].p['Vnn'][0, 0]
                self.r = v * np.sqrt(2.0 * m_pi)

                # Is the fit sane?
                # Simple check: does the fit lie "above" the guess?
                self.is_sane = np.abs(
                    gv.mean(
                        self.r)) >= np.abs(
                    gv.mean(
                        self.ds.r_guess))
                m_ll = self.fits['full'].p['light-light:dE'][0]
                m_hl = self.fits['full'].p['heavy-light:dE'][0]
                self.ds_fit = self.build_fit_dataset()
                if self.ds_fit is not None:
                    self.ds_fit.update(m_ll=m_ll, m_hl=m_hl)

            else:
                self.is_sane = False

    def fit_two_point(self, nstates, **fitter_kwargs):
        """Run the fits of two-point functions."""
        for tag in self.ds.c2:
            # TODO: handle possible re-running if fit fails initially
            # In the other code, I reset the priors on dEo to match dE
            fit = TwoPointAnalysis(self.ds.c2[tag]).\
                run_fit(nstates, **fitter_kwargs)
            if fit is None:
                logger.warning('[-] Warning: {0} fit failed'.format(tag))
            else:
                self.prior.update(update_with=fit.p, width=0.1)
            self.fits[tag] = fit

    def fit_form_factor(self, **fitter_kwargs):
        """Run the joint fit of 2- and 3-point functions for form factor."""
        models = [get_model(tag) for tag in self.ds]
        models = [model for model in models if model is not None]
        if len(models) >= 3:
            prior = self.prior
            prior.positive_params()
            fitter = cf.CorrFitter(models=models)
            fit = fitter.lsqfit(data=self.ds, prior=prior, **fitter_kwargs)
            
        else:
            fit = None
            fitter = None
            logger.warning('[-] insufficient models. skipping')

        if np.isnan(fit.chi2):
            logger.warning('[-] full fit failed')
            fit = None

        self.fitter = fitter
        self.fits['full'] = fit
    


    def collect_statistics(self):
        """Collect statistics about the fits."""
        for tag, fit in self.fits.items():
            if fit is not None:
                self.stats[tag] = FitStats(fit)
            else:
                self.stats[tag] = None


    def fit_jackknife(self, n_elim=1, **fitter_kwargs):
        jk = Jackknife(self.data, n_elim)
        jk_fits = []
        for jk_ds in jk:
            jk_fits.append(
                self.fitter.lsqfit(
                    data=jk_ds,
                    prior=self.prior,
                    **fitter_kwargs))
        return jk_fits

    def build_fit_dataset(self, p=None):
        if p is None:
            p = self.fits['full'].p
        tmin_hl = self.times['heavy-light:tmin']
        tmin_ll = self.times['light-light:tmin']

        times = {}
        for key in self.ds.c2:
            times[key] = np.array(self.ds.c2[key].tdata, dtype=float)
        for key in self.ds.c3:
            times[key] = np.array(self.ds.c3.tdata, dtype=float)

        yfit = {}
        for tag in self.ds.keys():
            try:
                yfit[tag] = self.get_model(tag).fitfcn(t=times[tag], p=p)
            except AttributeError:
                # Some times get_model returns a None...
                pass

        times = {
            'light-light:tmin': 1,
            'heavy-light:tmin': 1,
            'light-light:tmax': 14,
            'heavy-light:tmax': 14,
        }

        try:
            ds_fit = FormFactorDataset(yfit, self.ds.ns, self.ds.nt, times,
                                       momentum=self.ds.momentum, spin_taste_current=self.ds.spin_taste_current)

        except (RuntimeError, np.linalg.LinAlgError) as err:
            ds_fit = None

        return ds_fit

    def plot_results(self, axarr=None):

        nrows = len(self.fitter.models)
        if axarr is None:
            fig, axarr = plt.subplots(
                nrows=nrows, sharex=True, figsize=(10, 10))
        elif len(axarr) < nrows:
            raise ValueError("Too few rows for plot_results()?")

        fit = self.fits['full']

        for ax, model in zip(axarr, self.fitter.models):
            tag = model.datatag
            tfit = model.tfit
            ratio = self.ds[tag][tfit] / fit.fcn(fit.p)[tag]
            errorbar(ax, x=tfit, y=ratio, fmt='.')
            ax.axhline(1.0, ls='--', color='k')
            ax.set_ylabel(tag)
            ax.set_title('data/fit')

        axarr[-1].set_xlabel('t/a')
        return fig, axarr

    def plot_energy(self, ax, tag, do_partners=False):
        """ Make summary plot of the energies """

        if not do_partners:
            energy_tag = '{0}:dE'.format(tag)
            title = "Energies: {tag}".format(tag=tag)

            # Effective mass
            plot_meff(self.ds[tag], ax=ax, fmt='.', label='Effective mass')
            plot_meff(self.ds.c2bar[tag][:-5], ax=ax, fmt='.',
                      color='k', label='Smeared effective mass')

            # Fastfit guess
            E_ffit = self.ds[tag].ffit.E
            axhline(ax, E_ffit, label='ffit guess', color='k', ls=':')

        else:
            energy_tag = '{0}:dEo'.format(tag)
            title = "Partner Energies: {tag}".format(tag=tag)

        # Fit masses
        Es = np.cumsum(self.fits['full'].p[energy_tag])
        colors = sns.color_palette(n_colors=len(Es))
        for idx, E in enumerate(Es):
            label = "Fit: E{0}".format(idx)
            axhline(ax, E, label=label, alpha=0.75, color=colors[idx])

        # Priors
        Es = np.cumsum(self.prior.get_gaussian(energy_tag))
        for idx, E in enumerate(Es):
            label = "Prior: E{0}".format(idx)
            axhspan(ax, E, label=label, alpha=0.25, color=colors[idx])

        # Formatting
        ax.set_title(title)
        ax.set_xlabel("$t/a$")
        ax.set_ylabel("$Ea$")
        ax.legend(loc=1)
        return ax

    def plot_amplitude(self, ax, tag, do_partners=False):
        """ Make summary plot of the amplitudes"""

        if not do_partners:
            amp_tag = '{0}:a'.format(tag)
            title = "Amplitudes: {tag}".format(tag=tag)

            # Effective amplitude A_eff = C(t)*Exp(m_eff*t)
            corr = self.ds[tag]
            t = corr.tdata
            meff = effective_mass(corr)
            x = t[1:-1]
            y = np.sqrt(np.exp(meff * x) * corr[1:-1])
            errorbar(ax, x, y, fmt='.', label='Effective amplitude')

            # Fastfit guess
            amp_ffit = np.sqrt(corr.ffit.ampl)
            axhline(ax, amp_ffit, label='ffit guess', color='k', ls=':')

        else:
            amp_tag = '{0}:ao'.format(tag)
            title = "Partner Amplitudes: {tag}".format(tag=tag)

        # Fit amplitudes
        amps = self.fits['full'].p[amp_tag]
        colors = sns.color_palette(n_colors=len(amps))
        for idx, amp in enumerate(amps):
            label = 'Fit: A{0}'.format(idx)
            axhline(ax, amp, label=label, color=colors[idx])

        # Priors
        amps = self.prior.get_gaussian(amp_tag)
        for idx, amp in enumerate(amps):
            label = "Prior: A{0}".format(idx)
            axhspan(ax, amp, label=label, alpha=0.25, color=colors[idx])

        if not do_partners:
            # Effective amplitude unstable at long times
            # Set limits by hand as precautionary measure
            ax.set_ylim(ymin=0.0, ymax=1.0)

        # Formatting
        ax.set_title(title)
        ax.set_xlabel("$t/a$")
        ax.set_ylabel("Amplitude (lattice units)")
        ax.legend(loc=1)
        return ax

    def plot_states(self, axarr=None, do_partners=False):

        if axarr is None:
            fig, axarr = plt.subplots(
                nrows=2, ncols=2, sharex=True, figsize=(
                    20, 20))
        ((ax1, ax2), (ax3, ax4)) = axarr

        # Masses in first row
        for ax, tag in zip([ax1, ax2], ['light-light', 'heavy-light']):
            _ = self.plot_energy(ax, tag, do_partners)

        # Amplitudes in second row
        for ax, tag in zip([ax3, ax4], ['light-light', 'heavy-light']):
            _ = self.plot_amplitude(ax, tag, do_partners)

        # Bands for fit range
        ax_cols = [(ax1, ax3), (ax2, ax4)]
        for tag, ax_col in zip(['light-light', 'heavy-light'], ax_cols):
            for ax in ax_col:
                tmin = self.ds[tag].tmin
                tmax = self.ds[tag].tmax
                axvline(ax, tmin, color='k', ls='--')
                axvline(ax, tmax, color='k', ls='--')

        fig.tight_layout()

        return axarr

    def plot_form_factor(self, ax=None, xmax=12, color=None):
        """
        Plot the ratio which delivers the form factor together
        with the prior estimate and fit result.
        """
        if color is None:
            color = 'k'

        if self.positive_ff:
            flip = 1.0
        else:
            flip = -1.0
            print(
                '[+] The form factor is negative; reflecting form factor visualization to be positive.')

        # positive_ff=self.positive_ff)
        ax = self.ds.plot_ratio(ax=ax, xmax=xmax)
        norm = self.ds.normalization()
        axhspan(
            ax,
            self.r_guess *
            norm,
            alpha=0.25,
            color=color,
            label='Prior: R')
        axhline(ax, self.r * norm, alpha=0.50, color=color, label='Fit: R')
        ax.set_title("Form factor compared with estimates")
        ax.legend(loc=1)
        return ax

    def plot_overlay(self, ax=None):

        if self.positive_ff:
            flip = 1.0
        else:
            flip = -1.0
            print(
                '[+] The form factor is negative; reflecting overlay visualization to be positive.')

        ds_fit = self.ds_fit
        r_guess = self.r_guess
        r = self.r
        fitter = self.fitter
        ds = self.ds
        tfits = {model.datatag: model.tfit for model in fitter.models}

        nrows = max(len(ds_fit.rbar), len(ds.rbar))
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(10, 5))
        xmin = 1
        xmax = 14

        colors = {
            T: color for T,
            color in zip(
                ds.rbar.keys(),
                sns.color_palette())}

        for idx, (T, rbar) in enumerate(ds.rbar.iteritems()):
            # The data
            label = 'Data: $\\bar{{R}}$, T={0}'.format(T)
            y = rbar[xmin:xmax]
            x = np.arange(xmin, xmax)
            norm = ds.normalization()
            errorbar(
                ax,
                x,
                norm * y,
                fmt='.',
                marker='o',
                label=label,
                color=colors[T])

            # The fit
            norm = ds_fit.normalization()
            if T in ds_fit.rbar.keys():
                rbar = ds_fit.rbar[T]
                label = 'Fit: $\\bar{{R}}$, T={0}'.format(T)
                y = rbar[xmin:xmax]
                x = np.arange(xmin, xmax)
                errorbar(
                    ax,
                    x,
                    norm * y,
                    bands=True,
                    alpha=0.5,
                    label=label,
                    color=colors[T])

            # Fit window
            try:
                tmin = min(tfits[T])
                tmax = max(tfits[T])
                axvline(ax, tmin, color='k', ls='--')
                axvline(ax, tmax + 1, color=colors[T], ls='--')
            except BaseException:
                pass

        norm = ds.normalization()
        axhspan(ax, norm * r_guess, alpha=0.25, color='k', label='Prior: R')
        norm = ds_fit.normalization()
        axhline(ax, norm * r, alpha=0.50, color='k', label='Fit: R')

        ax.legend(loc=1)
        ax.set_ylabel('$\\bar{R}$ (lattice units)')
        ax.set_xlabel('t/a')
        ax.set_ylim(ymin=0.0, ymax=1.1 * gv.mean(self.r * norm))
        return ax
