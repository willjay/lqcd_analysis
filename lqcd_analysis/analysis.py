"""
TwoPointAnalysis
FormFactorAnalysis
"""
import logging
import collections
import numpy as np
import gvar as gv
import corrfitter as cf
from . import models
from . import correlator
from . import dataset
from . import bayes_prior
from . import figures
from . import serialize

LOGGER = logging.getLogger(__name__)

Nstates = collections.namedtuple(
    'NStates', ['n', 'no', 'm', 'mo'], defaults=(1, 0, 0, 0)
)
def _abs(val):
    return val * np.sign(val)

def n2s(val):
    """
    Computes the noise-to-signal ratio.
    """
    return gv.sdev(val) / gv.mean(val)


def phat2(ptag):
    """
    Strip out the squared momentum $\\hat{p}^2$ from strings like 'p123'.
    """
    return sum([int(pj)**2 for pj in ptag.lstrip("p")])


def p2(ptag, ns):
    """ Convert compute the squared momentum"""
    return phat2(ptag) * (2.0*np.pi/ns)**2.0


def delta_continuum_dispersion(ea, pa2, ma, alpha_v):
    """
    The continuum dispersion relation E^2 = p^2 + m^2 is expected to be
    satisfied up to lattice artifacts of order alpha (ap)^2. In other words,
    | 1 - E^2 / (p^2 + m^2) | < Order(alpha_v (ap)^2)
    To quantify how well this inequality is satisfied, divide both sides by the
    RHS to obtain a test statistic which we call delta:
    delta = | 1 - E^2 / (p^2 + m^2) | / (alpha_v (ap)^2)).
    Continuum-like results should typically satisfy
    (delta < 1) or perhaps (delta < 2).
    The utility of this quantity, beyond the usual plots of E^2 / (p^2 + m^2),
    is that provides an easy number to use as a cut for rejecting fits.
    Args:
        ea: float or gvar, the energy E*a in lattice units
        pa2: float or gvar, the squared momentum (pa)^2 in lattice units
        ma: float or gvar, the mass in lattice units
        alpha_v: float or gvar, the strong coupling constant
    Returns:
        delta: float, the value of the test statistic
    """
    if pa2 > 0:
        ratio = gv.mean(ea**2 / (ma**2 + pa2))
        delta = np.abs(ratio - 1) / (alpha_v * pa2)
        return gv.mean(delta)
    return 0


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
    kwargs = {key1: len(params.get(key2, [])) for key1, key2 in key_map.items()}
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


def get_three_point_model(t_snk, tfit, tdata, nstates, tags=None,
                          pedestal=None, constrain=False):
    if tags is None:
        tags = dataset.Tags(src='light-light', snk='heavy-light')
    src = tags.src
    snk = tags.snk
    if max(tfit) > max(tdata):
        LOGGER.error('Caution: max(tfit) exceeds max(tdata)')
        LOGGER.error('Restrict max(tfit) to max(tdata)')
        raise ValueError('Error: invalid tfit.')
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
        if constrain:
            _Model = models.ConstrainedCorr3
        else:
            _Model = models.Corr3
        model = _Model(
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
            Voo=voo,
            # optional "pedestal+fluctuation" treatment of target matrix element
            pedestal=pedestal
        )
    else:
        # Empty tfit -- no model
        model = None

    return model


def get_model(ds, tag, nstates, pedestal=None, constrain=False):
    """Gets a corrfitter model"""
    if isinstance(ds[tag], correlator.TwoPoint):
        osc = bool(nstates.no) if tag == ds.tags.src else bool(nstates.mo)
        return get_two_point_model(ds[tag], osc)
    if isinstance(tag, int):
        t_snk = tag
        tdata = ds.c3.times.tdata
        #if max(tdata) > max(ds.tdata):
            #LOGGER.warning("Caution: Ignoring noise_threshy.")
        return get_three_point_model(t_snk, ds.tfit[t_snk], tdata, nstates,
                                     pedestal=pedestal, constrain=constrain)
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
    Note that the matrix element Vnn[0,0] coming from the fit using corrfitter
    gives Vnn[0,0] = <final|J|initial> / sqrt(2 E_final) sqrt(2 E_initial).
    The multiplicative factor of sqrt(2 m_src) below removes this factor coming
    from the relativistic normalization of states, leaving the desired factor
    from the sink intact.
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
        self.prior = None
        self.fitter = None
        self._nstates = None
        self._fit = None

    def run_fit(self, nstates=Nstates(1, 0), prior=None, **fitter_kwargs):
        """
        Run the fit.
        Args:
            nstates: tuple (n_decay, n_osc) specifying the number of decaying
                and oscillating states, respectively. Defaults to (1,0).
            prior: BasicPrior object. Default is None, for which the fitter
                tries to constuct a prior itself.
        """
        self._nstates = nstates
        if prior is None:
            prior = bayes_prior.MesonPrior(
                nstates.n, nstates.no, amps=['a', 'ao'],
                tag=self.tag, ffit=self.c2.fastfit,
                extend=True
            )
        self.prior = prior
        # Model construction infers the fit times from c2
        model = get_two_point_model(self.c2, bool(nstates.no))
        self.fitter = cf.CorrFitter(models=model)
        data = {self.tag: self.c2}
        fit = self.fitter.lsqfit(
            data=data, prior=prior, p0=prior.p0, **fitter_kwargs
        )
        fit = serialize.SerializableNonlinearFit(fit)
        self._fit = fit
        if fit.failed:
            fit = None
        return fit

    def serialize(self, rawtext=True):
        payload = self._fit.serialize(rawtext)
        payload['tmin'] = self.c2.times.tmin
        payload['tmax'] = self.c2.times.tmax
        payload['n_decay'] = self._nstates.n
        payload['n_oscillating'] = self._nstates.no
        payload['energy'] = self._fit.p[f"{self.tag}:dE"][0]
        payload['amp'] = self._fit.p[f"{self.tag}:a"][0]
        return payload


class FormFactorAnalysis(object):

    def __init__(self, ds, positive_ff=True):

        self.ds = ds
        self.positive_ff = positive_ff
        self.prior = None
        self.fits = {}
        self.fitter = None
        self.r = None
        self.pedestal = None

    def run_sequential_fits(
            self, nstates, tmin_override=None,
            width=0.1, fractional_width=False,
            prior=None, chain=False, constrain=False,
            **fitter_kwargs):
        """
        Runs sequential fits.
        First runs two-point functions, whose results are used to update
        the (central values of) the priors for the joint fit. The runs the
        joint fit.
        """
        self.pedestal = fitter_kwargs.pop('pedestal', None)
        if prior is None:
            self.prior = bayes_prior.FormFactorPrior(
                nstates,
                self.ds,
                pedestal=self.pedestal,
                positive_ff=self.positive_ff)
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
        self.fit_form_factor(
            nstates=nstates,
            chain=chain,
            constrain=constrain,
            **fitter_kwargs)

    def mass(self, tag):
        """Gets the mass/energy of the ground state from full fit."""
        params = self.fits['full'].p
        return params[f'{tag}:dE'][0]

    @property
    def m_src(self):
        """Gets the mass/energy of the "source" ground state."""
        src_tag = self.ds.tags.src
        return self.mass(src_tag)

    @property
    def m_snk(self):
        """Gets the mass/energy of the "snk" ground state."""
        snk_tag = self.ds.tags.snk
        return self.mass(snk_tag)

    @property
    def matrix_element(self):
        """Fetches the matrix element Vnn[0, 0] needed for the form factor."""
        if self.fits['full'] is not None:
            return self.fits['full'].p['Vnn'][0, 0]

    @property
    def r_prior(self):
        src_tag = self.ds.tags.src
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
            if tag == self.ds.tags.snk:
              _nstates = Nstates(n=nstates.m, no=nstates.mo)
            # TODO: handle possible re-running if fit fails initially
            # In the other code, I reset the priors on dEo to match dE
            fit = TwoPointAnalysis(self.ds.c2[tag]).\
                run_fit(_nstates, **fitter_kwargs)
            if fit is None:
                LOGGER.warning('Fit failed for two-point function %s.', tag)
            else:
                pass
                # self.prior.update(
                #     update_with=fit.p, width=width, fractional_width=fractional_width)
            self.fits[tag] = fit

    def fit_form_factor(self, nstates, chain=False, constrain=False, **fitter_kwargs):
        """Run the joint fit of 2- and 3-point functions for form factor."""
        # Handle pedestal
        pedestal = fitter_kwargs.pop('pedestal', None)
        if pedestal is not None:
            self.pedestal = pedestal

        # Handle prior
        prior = fitter_kwargs.get('prior')
        if prior is not None:
            self.prior = prior
        else:
            prior = self.prior
        fitter_kwargs['prior'] = prior

        # Handle models
        models = []
        for tag in self.ds:
            model = get_model(self.ds, tag, nstates,self.pedestal, constrain)
            if model is not None:
                models.append(model)

        # Abort if too few models found
        if len(models) != len(set(self.ds.keys())):
            self.fitter = None
            fit = None
            LOGGER.warning('Insufficient models found. Skipping joint fit.')
            return

        # Run fit
        self.fitter = cf.CorrFitter(models=models)
        if chain:
            _lsqfit = self.fitter.chained_lsqfit
        else:
            _lsqfit = self.fitter.lsqfit
        fit = _lsqfit(data=self.ds, **fitter_kwargs)
        # fit = serialize.SerializableNonlinearFit(fit)
        fit = serialize.SerializableFormFactor(fit, tags=self.ds.tags)
        self.fits['full'] = fit
        if fit.failed:
            LOGGER.warning('Full joint fit failed.')
        else:
            # Update mass estimates in dataset, since the visualization of
            # the ratio R (or Rbar) is rather sensitive to the masses
            self.ds.set_masses(fit.p['light-light:dE'][0],
                               fit.p['heavy-light:dE'][0])

        # Tidy up final results
        if self.pedestal is not None:
            sign = np.sign(self.pedestal)
            vnn = self.pedestal + sign * _abs(fit.p['fluctuation'])
            fit.p['Vnn'][0, 0] = vnn
            fit.palt['Vnn'][0, 0] = vnn
        else:
            vnn = fit.p['Vnn'][0, 0]
        self.r = convert_vnn_to_ratio(self.m_src, vnn)

    def serialize(self, rawtext=True):
        sanitize = str if rawtext else lambda x: x
        payload = self.fits['full'].serialize(rawtext)
        payload['tmin_ll'] = self.ds[self.ds.tags.src].times.tmin
        payload['tmin_hl'] = self.ds[self.ds.tags.snk].times.tmin
        payload['tmax_ll'] = self.ds[self.ds.tags.src].times.tmax
        payload['tmax_hl'] = self.ds[self.ds.tags.snk].times.tmax
        payload['r'] = sanitize(self.r)
        payload['r_guess'] = self.ds.r_guess
        payload['is_sane'] = self.is_sane
        payload['prior_alias'] = 'standard prior'
        payload['pedestal'] = self.pedestal
        # Infer nstates
        nstates = count_nstates(self.fits['full'].p)
        payload['n_decay_ll'] = nstates.n
        payload['n_decay_hl'] = nstates.m
        payload['n_oscillating_ll'] = nstates.no
        payload['n_oscillating_hl'] = nstates.mo
        return payload

    def plot_results(self, ax=None):
        return figures.plot_form_factor_results(self, ax)

    def plot_energy_summary(self, ax, tag, osc=False, with_priors=True):
        return figures.plot_energy_summary(self, ax, tag, osc, with_priors)

    def plot_amplitude_summary(self, ax, tag, osc=False, with_priors=True):
        return figures.plot_amplitude_summary(self, ax, tag, osc, with_priors)

    def plot_states(self, a_fm=None):
        nstates = count_nstates(self.fits['full'].p)
        return figures.plot_states(self, nstates, a_fm)

    def plot_form_factor(self, ax=None, tmax=None, color='k', prior=True):
        return figures.plot_form_factor(self, ax, tmax, color, prior)

    def plot_comparison(self, a_fm=None):
        fit = self.fits['full']
        nstates = count_nstates(fit.p)
        return figures.plot_comparison(nstates, fit.prior, fit.p, a_fm)

class RatioAnalysis(object):

    def __init__(self, ds, nstates, restrict=None):
        self.ds = ds
        self.n = nstates.n
        self.m = nstates.m
        self.restrict = restrict
        assert False, "RatioAnalysis is not debugged. Use with care!"

    @property
    def t_snks(self):
        if self.restrict is None:
            return self.ds.t_snks
        else:
            return [t_snk for t_snk in self.ds.t_snks if t_snk in self.restrict]

    @property
    def tfit(self):
        return self.ds.tfit

    def model(self, t, t_snk, params):

        r = params['r']
        ans = r
        if self.n > 0:
            for ai, dEi in zip(params['amp_src'], np.cumsum(params['dE_src'])):
                ans = ans - ai * np.exp(-dEi * t)
        if self.m > 0:
            for ai, dEi in zip(params['amp_snk'], np.cumsum(params['dE_snk'])):
                ans = ans - ai * np.exp(-dEi * (t_snk - t))
        return ans

    def fitfcn(self, params):
        ans = {}
        for t_snk in self.t_snks:
            tfit = self.tfit[t_snk]
            ans[t_snk] = self.model(tfit, t_snk , params)
        return ans

    def buildy(self):
        y = {}
        for t_snk in self.ds.rbar:
            x = self.tfit[t_snk]
            y[t_snk] = self.ds.rbar[t_snk][x]
        return y

    def buildprior(self):
        r_guess = self.ds.r_guess
        prior = {'log(r)': np.log(gv.gvar(r_guess, 0.1 * r_guess))}
        if self.n > 0:
            prior['amp_src'] = [gv.gvar("1.0(1.0)") for _ in np.arange(self.n)]
            prior['log(dE_src)'] = [np.log(gv.gvar("0.5(0.5)")) for _ in np.arange(self.n)]
        if self.m > 0:
            prior['amp_snk'] = [gv.gvar("1.0(1.0)") for _ in np.arange(self.m)]
            prior['log(dE_snk)'] = [np.log(gv.gvar("0.5(0.5)")) for _ in np.arange(self.m)]
        return prior

    def lsqfit(self, **fitter_kwargs):

        y = self.buildy()
        if fitter_kwargs.get("prior") is None:
            fitter_kwargs["prior"] = self.buildprior()
        fit = lsqfit.nonlinear_fit(data=y, fcn=self.fitfcn, **fitter_kwargs)
        if np.isnan(fit.chi2) or np.isinf(fit.chi2):
            LOGGER.warning('Full joint fit failed.')
            fit = None
        return fit