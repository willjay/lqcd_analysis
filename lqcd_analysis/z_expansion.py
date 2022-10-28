import numpy as np
import gvar as gv
import lsqfit
from collections import namedtuple
import itertools
from . import serialize
from allhisq_analysis import data_tables

UsePoles = namedtuple("UsePoles", ["vector", "scalar"], defaults=["pdg", "pdg"])

class ConformalMap:
    """
    Kinematic information related to the conformal map z(q2).
    """
    def __init__(self, process, t0='zero'):
        """
        Args:
            process: str, 'D to pi', 'D to K' or 'Ds to K'
            t0: str, 'zero', 'topt', or 'tminus'
        """
        process = process.replace("2", " to ")
        if process not in ('D to pi', 'D to K', 'Ds to K'):
            raise NotImplementedError('Unrecognized process', process)
        self.process = process
        self.mother, self.daughter = self.get_hadron_masses()
        # tplus corresponds to the 2-particle production threshold
        self.tplus = self.get_tplus()
        # tminus corresponds to q2max, where both the mother and daughter are at rest
        self.tminus = gv.mean((self.mother - self.daughter)**2)
        if t0 in ('zero', 0):
            self.t0 = 0
        elif t0 == 'tminus':
            self.t0 = self.tminus
        elif t0 == 'topt':
            self.t0 = (self.mother + self.daughter) * (np.sqrt(self.mother) - np.sqrt(self.daughter))**2
        elif t0 == 'BESIII':
            self.t0 = self.tplus * (1 - np.sqrt(1 - self.tminus / self.tplus))
        elif isinstance(t0, str):
            raise NotImplementedError("Unsupported choice for 't0'.")
        else:
            self.t0 = t0


    def get_tplus(self):
        """
        Gets the location of tplus, i.e., the start of the 2-particle cut.
        """
        if self.process in ('D to pi', 'D to K'):
            return gv.mean((self.mother + self.daughter)**2)
        elif self.process == 'Ds to K':
            ctm = data_tables.ContinuumConstants()
            md = ctm.pdg['D']
            mpi = ctm.pdg['pi']
            return gv.mean((md + mpi)**2)
        else:
            raise ValueError("Unexpected process", self.process)


    def get_hadron_masses(self):
        """
        Gets the hadron masses from the PDG associated with the process.
        Returns:
            mother, daughter: the hadron masses in MeV
        """
        ctm = data_tables.ContinuumConstants()
        if self.process in ('Ds to K', 'Ds2K'):
            mother = ctm.pdg['D_s']
            daughter = ctm.pdg['K']
            return mother, daughter

        if self.process in ('D to K', 'D2K'):
            mother = ctm.pdg['D']
            daughter = ctm.pdg['K']
            return mother, daughter

        if self.process in ('D to pi', 'D2pi'):
            mother = ctm.pdg['D']
            daughter = ctm.pdg['pi']
            return mother, daughter

    def z(self, q2):
        """
        Computes the dimensional parameter z.
        """
        tplus = self.tplus
        t0 = self.t0
        return (np.sqrt(tplus - q2) - np.sqrt(tplus - t0)) / (np.sqrt(tplus - q2) + np.sqrt(tplus - t0))

    def ztoq2(self, z):
        """
        Compute the value of q2 given the value of z.
        """
        tplus = self.tplus
        t0 = self.t0
        return (t0*(z + 1)**2 - 4*tplus*z) / (z - 1)**2

    def q2(self, energy):
        """
        Computes the squared momentum transfer q2 in MeV^2 from the energy of the daughter hadron.
        Assumes that the energy is measured in the rest frame of the mother hadron.
        Args:
            energy: the energy of the daughter hadron in MeV
        """
        ML = self.daughter
        MH = self.mother
        return MH**2 + ML**2 - 2.0*MH*energy

    def get_q2_bounds(self):
        """
        Gets the mininum and maximum values of q2 in MeV^2 for the decay.
        """
        q2min = 0
        q2max = gv.mean(self.tminus)
        return (q2min, q2max)

    def get_pole(self, channel):
        """
        Gets the nearest pole mass (in MeV) listed in the PDG for the specified channel.
        Args:
            channel: str, 'vector' or 'scalar'
        """
        ctm = data_tables.ContinuumConstants()
        poles = {
            # Vectors, J^P = 1-
            ("D to pi", "vector"): ctm.pdg['Dstar'],
            ("D to K", "vector"):  ctm.pdg['Dstar_s'],
            ("Ds to K", "vector"): ctm.pdg['Dstar'],
            # Scalars, J^P = 0+
            ("D to pi", "scalar"): ctm.pdg['D0star'],
            ("D to K", "scalar"):  ctm.pdg['D0star_s'],
            ("Ds to K", "scalar"): ctm.pdg['D0star'],
        }
        pole = gv.mean(poles.get((self.process, channel)))
        if pole is None:
            raise ValueError("Unable to locate pole")
        return pole

    def has_subthreshold_pole(self, channel):
        """
        Checks whether the channel has a subthreshold pole
        Args:
            channel: str, 'vector' or 'scalar'
        Returns:
            bool
        """
        pole = gv.mean(self.get_pole(channel))
        if pole**2 < self.tplus:
            return True
        return False

    def p_daughter(self, q2, use_GeV=False):
        """
        By definition, in the rest frame of the mother hadron
        q2 = MH^2 + ML^2 - 2*MH*EL --> EL = (MH^2 + ML^2 - q2) / (2 * MH).
        Then the relativistic dispersion relation gives
        E^2 = M^2 + p^2 --> p = sqrt(E^2 - M^2).
        """
        MH, ML = self.get_hadron_masses()
        if use_GeV:
            MH /= 1e3  # MeV --> GeV
            ML /= 1e3
        EL = (MH**2 +ML**2 - q2)/(2*MH)
        eps = 1e-12
        return np.sqrt(EL**2 - ML**2 + eps)

class ZFitter:
    """ foo """
    def __init__(self, process, energy, f_parallel, f_perp, t0='zero'):
        """
        Carries out fits to the model-independent "BCL" expansion due to
        Bourrely, Caprini, Lellouch
        "Model-independent description of B -> pi l nu decays and a determination of |V(ub)|"
        PRD 79 (2009) 013008, https://arxiv.org/pdf/0807.2722.pdf.

        For a recent use in similar lattice QCD, see the calculation
        J. Bailey et al. "|Vub| from B -> pi l nu decays and (2+1)-flavor lattice QCD"
        PRD 92 (2015) 1, 014024, https://arxiv.org/pdf/1503.07839.pdf

        Args:
            process: str, the name of the decay (e.g., 'D to pi')
            energy: array, the energy of the daughter hadron in w0 units
                (and assumed to be in the rest frame of the mother hadron)
            f_parallel, f_perp: array, the form factor data in w0 units.
        """
        if process not in ('D to pi', 'D to K', 'Ds to K'):
            raise NotImplementedError('Unrecognized process', process)
        self.process = process
        self.t0 = t0
        self.conformal_map = ConformalMap(self.process, self.t0)
        # Input data
        self.energy = energy
        self.f_parallel = f_parallel
        self.f_perp = f_perp
        # Splines for data
        self.fplus_data = None
        self.f0_data = None
        self.build_data_splines()
        # Splines for fits
        self.fplus_fit = None
        self.f0_fit = None

    def _model_fplus(self, x, p):
        """
        The constrained z-expansion function which imposes that
        Im f_+(q2) ~ (q2 + t_+)^{3/2} near the 2-particle threshold.
        See Eq (15), and the preceding discussion in BCL,
        https://arxiv.org/pdf/0807.2722.pdf.
        """
        z = x['z']
        a = p['a']
        N = len(a)
        result = np.zeros(z.shape)
        for n, an in enumerate(a):
            result = result + an * (z**n - (n/N)*(-1)**(n-N)*z**N)
        return result

    def _model_f0(self, x, p):
        z = x['z']
        b = p['b']
        result = np.zeros(z.shape)
        for n, bn in enumerate(b):
            result = result + bn * z**n
        return result

    def model_z(self, x, p):
        return {
            'fplus': self._model_fplus(x, p),
            'f0': self._model_f0(x, p),
        }

    def get_q2_bounds(self, units='MeV'):
        GeV = 1e-3  # MeV --> GeV conversion
        q2min, q2max = self.conformal_map.get_q2_bounds()
        if units == 'MeV':
            pass
        elif units == 'GeV':
            q2min *= GeV**2
            q2max *= GeV**2
        else:
            raise ValueError("Unrecognized units", units)
        return (q2min, q2max)

    def get_pole(self, channel, use_pole):
        """

        """
        if use_pole not in [True, False, "pdg"]:
            raise ValueError("Unrecognized treatment")
        if use_pole == "pdg":
            use_pole = self.conformal_map.has_subthreshold_pole(channel)
        if use_pole:
            return self.conformal_map.get_pole(channel)
        return np.nan

    def build_data_splines(self):
        """
        Builds splines for fplus and f0 using data for f_parallel and f_perp.
        """
        GeV = 1e-3  # MeV --> GeV conversion
        ctm = data_tables.ContinuumConstants()
        scale = data_tables.ScaleSetting()
        w0 = gv.mean(scale.w0_fm / ctm.hbarc)  # w0 in units of 1/MeV

        # Construct f+ and f0 from f_perp and f_parallel
        # f_perp and f_parallel are assumed to be w0 units, so the masses must be too.
        ML = gv.mean(self.conformal_map.daughter * w0)
        MH = gv.mean(self.conformal_map.mother * w0)
        EL = gv.mean(self.energy)
        fplus_knots = (self.f_parallel + (MH - EL) * self.f_perp) / np.sqrt(2*MH)
        f0_knots = np.sqrt(2*MH) / (MH**2 - ML**2) * ((MH-EL) * self.f_parallel + (EL**2 - ML**2)*self.f_perp)

        # fplus and f0 are dimensionless
        # Carry out the z-expansion in physical units
        # Input the energy in MeV, but use q2 in GeV^2
        # so that typical values are O(1) and not O(10^6)
        q2_knots = gv.mean(self.conformal_map.q2(EL/w0) * GeV**2)

        # Create splines for data
        self.fplus_data = gv.cspline.CSpline(q2_knots, fplus_knots, warn=True)
        self.f0_data = gv.cspline.CSpline(q2_knots, f0_knots, warn=True)

    def build_fit_splines(self, fit, use_poles):
        """
        Builds the splines for fplus and f0 using a joint fit to the z-expansion.
        """
        GeV = 1e-3  # MeV --> GeV conversion

        # Create splines for the fits
        z_knots = self.conformal_map.z(np.linspace(*self.get_q2_bounds('MeV')))
        q2_knots = np.linspace(*self.get_q2_bounds('GeV'))
        fit_knots = fit.fcn({'z': z_knots}, fit.p)

        fplus_knots = fit_knots['fplus']
        pole_vector = self.get_pole('vector', use_poles.vector) * GeV
        if not np.isnan(pole_vector):
            fplus_knots = fplus_knots/(1-q2_knots/pole_vector**2)
        self.fplus_fit = gv.cspline.CSpline(q2_knots, fplus_knots)

        f0_knots = fit_knots['f0']
        pole_scalar = self.get_pole('scalar', use_poles.scalar) * GeV
        if not np.isnan(pole_scalar):
            f0_knots = f0_knots/(1-q2_knots/pole_scalar**2)
        self.f0_fit = gv.cspline.CSpline(q2_knots, f0_knots)


    def __call__(self, q2, nterms, use_poles=UsePoles("pdg", "pdg"), **kwargs):
        """
        Runs a simultaneous correlated fit to the z-expansion for f0 and f+.
        Args:
            q2: array, q2 points at which to generate synthetic data. Assumed to be in GeV^2.
            nterms: int, the number of terms to include in the fit to z-expansion
            use_poles: namedtuple, specifying how to handle possible subthreshold poles
            kwargs: keyword arguments passed to the fitter
        """
        GeV = 1e-3  # MeV --> GeV conversion
        q2 = np.asarray(q2)
        # ctm = data_tables.ContinuumConstants()
        # scale = data_tables.ScaleSetting()
        # w0 = gv.mean(scale.w0_fm / ctm.hbarc)  # w0 in units of 1/MeV

        # # Construct f+ and f0 from f_perp and f_parallel
        # # f_perp and f_parallel are assumed to be w0 units, so the masses must be too.
        # ML = gv.mean(self.conformal_map.daughter * w0)
        # MH = gv.mean(self.conformal_map.mother * w0)
        # EL = gv.mean(self.energy)
        # fplus_knots = (self.f_parallel + (MH - EL) * self.f_perp) / np.sqrt(2*MH)
        # f0_knots = np.sqrt(2*MH) / (MH**2 - ML**2) * ((MH-EL) * self.f_parallel + (EL**2 - ML**2)*self.f_perp)

        # # fplus and f0 are dimensionless
        # # Carry out the z-expansion in physical units
        # # Input the energy in MeV, but use q2 in GeV^2
        # # so that typical values are O(1) and not O(10^6)
        # q2_knots = gv.mean(self.conformal_map.q2(EL/w0) * GeV**2)

        # # Create splines for data
        # self.fplus_data = gv.cspline.CSpline(q2_knots, fplus_knots, warn=True)
        # self.f0_data = gv.cspline.CSpline(q2_knots, f0_knots, warn=True)

        # Set up and run the fit
        x = {'z': gv.mean(self.conformal_map.z(q2/GeV**2))}  # conformal_map expects values in MeV
        y = {'fplus': self.fplus_data(q2), 'f0': self.f0_data(q2)}
        pole_vector = self.get_pole('vector', use_poles.vector) * GeV
        pole_scalar = self.get_pole('scalar', use_poles.scalar) * GeV

        if not np.isnan(pole_vector):
            y['fplus'] = y['fplus'] * (1 - q2/pole_vector**2)
        if not np.isnan(pole_scalar):
            y['f0'] = y['f0'] * (1 - q2/pole_scalar**2)

        prior = {
            'a': [gv.gvar(0, 100) for _ in range(nterms)],
            'b': [gv.gvar(0, 100) for _ in range(nterms)],
        }
        fit = lsqfit.nonlinear_fit(data=(x, y), fcn=self.model_z, p0=gv.mean(prior), prior=prior, **kwargs)
        self.build_fit_splines(fit, use_poles)
        # Create splines for the fits
        # z_knots = self.conformal_map.z(np.linspace(*self.get_q2_bounds('MeV')))
        # q2_knots = np.linspace(*self.get_q2_bounds('GeV'))
        # fit_knots = fit.fcn({'z': z_knots}, fit.p)

        # fplus_knots = fit_knots['fplus']
        # if not np.isnan(pole_vector):
        #     fplus_knots = fplus_knots/(1-q2_knots/pole_vector**2)
        # self.fplus_fit = gv.cspline.CSpline(q2_knots, fplus_knots)

        # f0_knots = fit_knots['f0']
        # if not np.isnan(pole_scalar):
        #     f0_knots = f0_knots/(1-q2_knots/pole_scalar**2)
        # self.f0_fit = gv.cspline.CSpline(q2_knots, f0_knots)

        return serialize.SerializableNonlinearFit(fit)


class ZFitter2:
    """ foo """
    def __init__(self, process, energy, fplus, fzero, t0='zero'):
        """
        Carries out fits to the model-independent "BCL" expansion due to
        Bourrely, Caprini, Lellouch
        "Model-independent description of B -> pi l nu decays and a determination of |V(ub)|"
        PRD 79 (2009) 013008, https://arxiv.org/pdf/0807.2722.pdf.

        For a recent use in similar lattice QCD, see the calculation
        J. Bailey et al. "|Vub| from B -> pi l nu decays and (2+1)-flavor lattice QCD"
        PRD 92 (2015) 1, 014024, https://arxiv.org/pdf/1503.07839.pdf

        Args:
            process: str, the name of the decay (e.g., 'D to pi')
            energy: array, the energy of the daughter hadron in w0 units
                (and assumed to be in the rest frame of the mother hadron)
            fplus, fzero: array, the form factor data
        """
        if process not in ('D to pi', 'D to K', 'Ds to K'):
            raise NotImplementedError('Unrecognized process', process)
        self.process = process
        self.t0 = t0
        self.conformal_map = ConformalMap(self.process, self.t0)
        # Input data
        self.energy = energy
        # Splines for data
        self.fplus_data = fplus
        self.f0_data = fzero
        self.build_data_splines()
        # Splines for fits
        self.fplus_fit = None
        self.f0_fit = None

    def _model_fplus(self, x, p):
        """
        The constrained z-expansion function which imposes that
        Im f_+(q2) ~ (q2 + t_+)^{3/2} near the 2-particle threshold.
        See Eq (15), and the preceding discussion in BCL,
        https://arxiv.org/pdf/0807.2722.pdf.
        """
        z = x['z']
        a = p['a']
        c0 = p['c0']
        a = [c0] + list(a)
        N = len(a)
        result = np.zeros(z.shape)
        for n, an in enumerate(a):
            result = result + an * (z**n - (n/N)*(-1)**(n-N)*z**N)
        return result

    def _model_f0(self, x, p):
        z = x['z']
        b = p['b']
        c0 = p['c0']
        b = [c0] + list(b)
        result = np.zeros(z.shape)
        for n, bn in enumerate(b):
            result = result + bn * z**n
        return result

    def model_z(self, x, p):
        return {
            'fplus': self._model_fplus(x, p),
            'f0': self._model_f0(x, p),
        }

    def get_q2_bounds(self, units='MeV'):
        GeV = 1e-3  # MeV --> GeV conversion
        q2min, q2max = self.conformal_map.get_q2_bounds()
        if units == 'MeV':
            pass
        elif units == 'GeV':
            q2min *= GeV**2
            q2max *= GeV**2
        else:
            raise ValueError("Unrecognized units", units)
        return (q2min, q2max)

    def get_pole(self, channel, use_pole):
        """

        """
        if use_pole not in [True, False, "pdg"]:
            raise ValueError("Unrecognized treatment")
        if use_pole == "pdg":
            use_pole = self.conformal_map.has_subthreshold_pole(channel)
        if use_pole:
            return self.conformal_map.get_pole(channel)
        return np.nan

    def build_data_splines(self):
        """
        Builds splines for fplus and f0 using data for f_parallel and f_perp.
        """
        GeV = 1e-3  # MeV --> GeV conversion
        ctm = data_tables.ContinuumConstants()
        scale = data_tables.ScaleSetting()
        w0 = gv.mean(scale.w0_fm / ctm.hbarc)  # w0 in units of 1/MeV

        # Construct f+ and f0 from f_perp and f_parallel
        # f_perp and f_parallel are assumed to be w0 units, so the masses must be too.
        EL = gv.mean(self.energy)
        fplus_knots = self.fplus_data
        f0_knots = self.f0_data

        # fplus and f0 are dimensionless
        # Carry out the z-expansion in physical units
        # Input the energy in MeV, but use q2 in GeV^2
        # so that typical values are O(1) and not O(10^6)
        q2_knots = gv.mean(self.conformal_map.q2(EL/w0) * GeV**2)

        # Create splines for data
        self.fplus_data = gv.cspline.CSpline(q2_knots, fplus_knots, warn=True)
        self.f0_data = gv.cspline.CSpline(q2_knots, f0_knots, warn=True)

    def build_fit_splines(self, fit, use_poles):
        """
        Builds the splines for fplus and f0 using a joint fit to the z-expansion.
        """
        GeV = 1e-3  # MeV --> GeV conversion

        # Create splines for the fits
        z_knots = self.conformal_map.z(np.linspace(*self.get_q2_bounds('MeV')))
        q2_knots = np.linspace(*self.get_q2_bounds('GeV'))
        fit_knots = fit.fcn({'z': z_knots}, fit.p)

        fplus_knots = fit_knots['fplus']
        pole_vector = self.get_pole('vector', use_poles.vector) * GeV
        if not np.isnan(pole_vector):
            fplus_knots = fplus_knots/(1-q2_knots/pole_vector**2)
        self.fplus_fit = gv.cspline.CSpline(q2_knots, fplus_knots)

        f0_knots = fit_knots['f0']
        pole_scalar = self.get_pole('scalar', use_poles.scalar) * GeV
        if not np.isnan(pole_scalar):
            f0_knots = f0_knots/(1-q2_knots/pole_scalar**2)
        self.f0_fit = gv.cspline.CSpline(q2_knots, f0_knots)

    def __call__(self, q2, nterms, use_poles=UsePoles("pdg", "pdg"), **kwargs):
        """
        Runs a simultaneous correlated fit to the z-expansion for f0 and f+.
        Args:
            q2: array, q2 points at which to generate synthetic data. Assumed to be in GeV^2.
            nterms: int, the number of terms to include in the fit to z-expansion
            use_poles: namedtuple, specifying how to handle possible subthreshold poles
            kwargs: keyword arguments passed to the fitter
        """
        GeV = 1e-3  # MeV --> GeV conversion
        q2 = np.asarray(q2)

        # Set up and run the fit
        x = {'z': gv.mean(self.conformal_map.z(q2/GeV**2))}  # conformal_map expects values in MeV
        y = {'fplus': self.fplus_data(q2), 'f0': self.f0_data(q2)}
        pole_vector = self.get_pole('vector', use_poles.vector) * GeV
        pole_scalar = self.get_pole('scalar', use_poles.scalar) * GeV

        if not np.isnan(pole_vector):
            y['fplus'] = y['fplus'] * (1 - q2/pole_vector**2)
        if not np.isnan(pole_scalar):
            y['f0'] = y['f0'] * (1 - q2/pole_scalar**2)

        prior = {
            'c0': gv.gvar(0, 10),
            'a': [gv.gvar(0, 10) for _ in range(nterms-1)],
            'b': [gv.gvar(0, 10) for _ in range(nterms-1)],
        }
        fit = lsqfit.nonlinear_fit(data=(x, y), fcn=self.model_z, p0=gv.mean(prior), prior=prior, **kwargs)
        self.build_fit_splines(fit, use_poles)

        return serialize.SerializableNonlinearFit(fit)

class ZFitter3:
    """ foo """
    def __init__(self, process, fplus, fzero, t0='zero'):
        """
        Carries out fits to the model-independent "BCL" expansion due to
        Bourrely, Caprini, Lellouch
        "Model-independent description of B -> pi l nu decays and a determination of |V(ub)|"
        PRD 79 (2009) 013008, https://arxiv.org/pdf/0807.2722.pdf.

        For a recent use in similar lattice QCD, see the calculation
        J. Bailey et al. "|Vub| from B -> pi l nu decays and (2+1)-flavor lattice QCD"
        PRD 92 (2015) 1, 014024, https://arxiv.org/pdf/1503.07839.pdf

        Args:
            process: str, the name of the decay (e.g., 'D to pi')
            energy: array, the energy of the daughter hadron in w0 units
                (and assumed to be in the rest frame of the mother hadron)
            fplus, fzero: array, the form factor data
        """
        process = process.replace("2", " to ")
        if process not in ('D to pi', 'D to K', 'Ds to K'):
            raise NotImplementedError('Unrecognized process', process)
        self.process = process
        self.t0 = t0
        self.conformal_map = ConformalMap(self.process, self.t0)
        # Splines for data
        self.fplus_data = fplus
        self.f0_data = fzero
        # Splines for fits
        self.fplus_fit = None
        self.f0_fit = None

    def _model_fplus(self, x, p):
        """
        The constrained z-expansion function which imposes that
        Im f_+(q2) ~ (q2 + t_+)^{3/2} near the 2-particle threshold.
        See Eq (15), and the preceding discussion in BCL,
        https://arxiv.org/pdf/0807.2722.pdf.
        """
        z = x['z']
        a = p['a']
        c0 = p['c0']
        a = [c0] + list(a)
        N = len(a)
        result = np.zeros(z.shape)
        for n, an in enumerate(a):
            result = result + an * (z**n - (n/N)*(-1)**(n-N)*z**N)
        return result

    def _model_f0(self, x, p):
        z = x['z']
        b = p['b']
        c0 = p['c0']
        b = [c0] + list(b)
        result = np.zeros(z.shape)
        for n, bn in enumerate(b):
            result = result + bn * z**n
        return result

    def model_z(self, x, p):
        return {
            'fplus': self._model_fplus(x, p),
            'f0': self._model_f0(x, p),
        }

    def get_q2_bounds(self, units='MeV'):
        GeV = 1e-3  # MeV --> GeV conversion
        q2min, q2max = self.conformal_map.get_q2_bounds()
        if units == 'MeV':
            pass
        elif units == 'GeV':
            q2min *= GeV**2
            q2max *= GeV**2
        else:
            raise ValueError("Unrecognized units", units)
        return (q2min, q2max)

    def get_pole(self, channel, use_pole):
        """

        """
        if use_pole not in [True, False, "pdg"]:
            raise ValueError("Unrecognized treatment")
        if use_pole == "pdg":
            use_pole = self.conformal_map.has_subthreshold_pole(channel)
        if use_pole:
            return self.conformal_map.get_pole(channel)
        return np.nan

    def build_fit_splines(self, fit, use_poles):
        """
        Builds the splines for fplus and f0 using a joint fit to the z-expansion.
        """
        GeV = 1e-3  # MeV --> GeV conversion

        # Create splines for the fits
        z_knots = self.conformal_map.z(np.linspace(*self.get_q2_bounds('MeV')))
        q2_knots = np.linspace(*self.get_q2_bounds('GeV'))
        fit_knots = fit.fcn({'z': z_knots}, fit.p)

        fplus_knots = fit_knots['fplus']
        pole_vector = self.get_pole('vector', use_poles.vector) * GeV
        if not np.isnan(pole_vector):
            fplus_knots = fplus_knots/(1-q2_knots/pole_vector**2)
        self.fplus_fit = gv.cspline.CSpline(q2_knots, fplus_knots)

        f0_knots = fit_knots['f0']
        pole_scalar = self.get_pole('scalar', use_poles.scalar) * GeV
        if not np.isnan(pole_scalar):
            f0_knots = f0_knots/(1-q2_knots/pole_scalar**2)
        self.f0_fit = gv.cspline.CSpline(q2_knots, f0_knots)

    def __call__(self, q2, nterms, use_poles=UsePoles("pdg", "pdg"), **kwargs):
        """
        Runs a simultaneous correlated fit to the z-expansion for f0 and f+.
        Args:
            q2: array, q2 points at which to generate synthetic data. Assumed to be in GeV^2.
            nterms: int, the number of terms to include in the fit to z-expansion
            use_poles: namedtuple, specifying how to handle possible subthreshold poles
            kwargs: keyword arguments passed to the fitter
        """
        GeV = 1e-3  # GeV --> MeV conversion
        q2 = np.asarray(q2)

        # Set up and run the fit
        x = {'z': gv.mean(self.conformal_map.z(q2/GeV**2))}  # conformal_map expects values in MeV
        y = {'fplus': self.fplus_data(q2), 'f0': self.f0_data(q2)}
        pole_vector = self.get_pole('vector', use_poles.vector) * GeV
        pole_scalar = self.get_pole('scalar', use_poles.scalar) * GeV

        if not np.isnan(pole_vector):
            y['fplus'] = y['fplus'] * (1 - q2/pole_vector**2)
        if not np.isnan(pole_scalar):
            y['f0'] = y['f0'] * (1 - q2/pole_scalar**2)

        prior = {
            'c0': gv.gvar(0, 1),
            'a': [gv.gvar(0, 1) for _ in range(nterms-1)],
            'b': [gv.gvar(0, 1) for _ in range(nterms-1)],
        }
        fit = lsqfit.nonlinear_fit(data=(x, y), fcn=self.model_z, p0=gv.mean(prior), prior=prior, **kwargs)
        self.build_fit_splines(fit, use_poles)

        return serialize.SerializableNonlinearFit(fit)



#####################################################################
# Too flexible. Can probably delete once the analysis settles down. #
#####################################################################

# class ZFitter3:
#     def __init__(self, process, energy, fparallel, fperp, f0, t0='zero'):
#         """
#         Carries out fits to the model-independent "BCL" expansion due to
#         Bourrely, Caprini, Lellouch
#         "Model-independent description of B -> pi l nu decays and a determination of |V(ub)|"
#         PRD 79 (2009) 013008, https://arxiv.org/pdf/0807.2722.pdf.

#         For a recent use in similar lattice QCD, see the calculation
#         J. Bailey et al. "|Vub| from B -> pi l nu decays and (2+1)-flavor lattice QCD"
#         PRD 92 (2015) 1, 014024, https://arxiv.org/pdf/1503.07839.pdf

#         Args:
#             process: str, the name of the decay (e.g., 'D to pi')
#             energy: array, the energy of the daughter hadron in w0 units
#                 (and assumed to be in the rest frame of the mother hadron)
#             f_parallel, f_perp: array, the form factor data in w0 units.
#         """

#         if process not in ('D to pi', 'D to K', 'Ds to K'):
#             raise NotImplementedError('Unrecognized process', process)
#         self.process = process
#         self.t0 = t0
#         self.conformal_map = ConformalMap(self.process, self.t0)
#         # Input data
#         self.energy = energy
#         self.data = {'fparallel': fparallel, 'fperp': fperp, 'f0': f0,}

#         # Splines for data
#         self.splines_data = {}
#         self.build_data_splines()

#         # Splines for fits
#         self.splines_fit = {}


#     def _model_fplus(self, x, p):
#         """
#         The constrained z-expansion function which imposes that
#         Im f_+(q2) ~ (q2 + t_+)^{3/2} near the 2-particle threshold.
#         See Eq (15), and the preceding discussion in BCL,
#         https://arxiv.org/pdf/0807.2722.pdf.
#         """
#         z = x['z']
#         a = p['a']
#         N = len(a)
#         result = np.zeros(z.shape)
#         for n, an in enumerate(a):
#             result = result + an * (z**n - (n/N)*(-1)**(n-N)*z**N)
#         return result

#     def _model_f0(self, x, p):
#         z = x['z']
#         b = p['b']
#         result = np.zeros(z.shape)
#         for n, bn in enumerate(b):
#             result = result + bn * z**n
#         return result

#     def model_z(self, x, p):
#         return {
#             'fplus': self._model_fplus(x, p),
#             'f0': self._model_f0(x, p),
#         }

#     def get_q2_bounds(self, units='MeV'):
#         GeV = 1e-3  # MeV --> GeV conversion
#         q2min, q2max = self.conformal_map.get_q2_bounds()
#         if units == 'MeV':
#             pass
#         elif units == 'GeV':
#             q2min *= GeV**2
#             q2max *= GeV**2
#         else:
#             raise ValueError("Unrecognized units", units)
#         return (q2min, q2max)

#     def get_pole(self, channel, use_pole):
#         """

#         """
#         if use_pole not in [True, False, "pdg"]:
#             raise ValueError("Unrecognized treatment")
#         if use_pole == "pdg":
#             use_pole = self.conformal_map.has_subthreshold_pole(channel)
#         if use_pole:
#             return self.conformal_map.get_pole(channel)
#         return np.nan

#     def build_data_splines(self):
#         """
#         Builds splines for fplus and f0 using data for f_parallel and f_perp.
#         """
#         GeV = 1e-3  # MeV --> GeV conversion
#         ctm = data_tables.ContinuumConstants()
#         scale = data_tables.ScaleSetting()
#         w0 = gv.mean(scale.w0_fm / ctm.hbarc)  # w0 in units of 1/MeV

#         # Construct f+ and f0 in different ways
#         # f_perp and f_parallel are assumed to be w0 units, so the masses must be too.
#         ML = gv.mean(self.conformal_map.daughter * w0)
#         MH = gv.mean(self.conformal_map.mother * w0)
#         EL = gv.mean(self.energy)
#         q2 = MH**2 + ML**2 -2*MH*EL

#         fparallel = self.data['fparallel']
#         fperp = self.data['fperp']
#         f0 = self.data['f0']

#         f_knots = {}
#         # "fplus ~ V4 + Vi"
#         f_knots[("fplus", "V4:Vi")] = (fparallel + (MH - EL) * fperp)/np.sqrt(2*MH)
#         # "fplus ~ Vi + S"
#         f_knots[("fplus", "Vi:S")] =\
#             (MH**2 - ML**2)/(2*MH)/(MH-EL) * f0\
#             + (MH - EL)/np.sqrt(2*MH)*(1 - (EL**2 - ML**2)/(MH - EL)**2)*fperp
#         # "fplus ~ V4 + S"
#         f_knots[("fplus", "V4:S")] =\
#             (np.sqrt(2*MH) * q2 * fparallel - (MH**2 - ML**2)*(MH - EL) * f0)\
#             /(q2*(MH - EL) - (MH**2 - ML**2)*(MH - EL))
#         # "f0 ~ V4 + Vi"
#         f_knots[("f0", "V4:Vi")] = np.sqrt(2*MH) / (MH**2 - ML**2) *\
#             ((MH-EL) * fparallel + (EL**2 - ML**2)*fperp)
#         # "f0 ~ S"
#         f_knots[("f0", "S")] = f0

#         # fplus and f0 are dimensionless
#         # Carry out the z-expansion in physical units
#         # Input the energy in MeV, but use q2 in GeV^2
#         # so that typical values are O(1) and not O(10^6)
#         q2_knots = gv.mean(self.conformal_map.q2(EL/w0) * GeV**2)

#         # Create splines for data
#         for key, y_knots in f_knots.items():
#             self.splines_data[key] =\
#                 gv.cspline.CSpline(q2_knots, y_knots, warn=True)

#     def build_fit_splines(self, fit, use_poles, key):
#         """
#         Builds the splines for fplus and f0 using a joint fit to the z-expansion.
#         """
#         GeV = 1e-3  # MeV --> GeV conversion

#         # Create splines for the fits
#         z_knots = self.conformal_map.z(np.linspace(*self.get_q2_bounds('MeV')))
#         q2_knots = np.linspace(*self.get_q2_bounds('GeV'))
#         fit_knots = fit.fcn({'z': z_knots}, fit.p)

#         fplus_knots = fit_knots['fplus']
#         pole_vector = self.get_pole('vector', use_poles.vector) * GeV
#         if not np.isnan(pole_vector):
#             fplus_knots = fplus_knots/(1-q2_knots/pole_vector**2)
#         fplus = gv.cspline.CSpline(q2_knots, fplus_knots)

#         f0_knots = fit_knots['f0']
#         pole_scalar = self.get_pole('scalar', use_poles.scalar) * GeV
#         if not np.isnan(pole_scalar):
#             f0_knots = f0_knots/(1-q2_knots/pole_scalar**2)
#         f0 = gv.cspline.CSpline(q2_knots, f0_knots)
#         self.splines_fit[key] = {'f0': f0, 'fplus': fplus}


#     def __call__(self, q2, nterms, use_poles=UsePoles("pdg", "pdg"), **kwargs):
#         """
#         Runs a simultaneous correlated fit to the z-expansion for f0 and f+.
#         Args:
#             q2: array, q2 points at which to generate synthetic data. Assumed to be in GeV^2.
#             nterms: int, the number of terms to include in the fit to z-expansion
#             use_poles: namedtuple, specifying how to handle possible subthreshold poles
#             kwargs: keyword arguments passed to the fitter
#         """
#         GeV = 1e-3  # MeV --> GeV conversion
#         q2 = np.asarray(q2)
#         x = {'z': gv.mean(self.conformal_map.z(q2/GeV**2))}  # conformal_map expects values in MeV
#         pole_vector = self.get_pole('vector', use_poles.vector) * GeV
#         pole_scalar = self.get_pole('scalar', use_poles.scalar) * GeV
#         prior = {
#                 'a': [gv.gvar(0, 100) for _ in range(nterms)],
#                 'b': [gv.gvar(0, 100) for _ in range(nterms)],
#             }
#         combos = itertools.product(["V4:Vi", "V4:S", "Vi:S"],["V4:Vi", "S"])
#         fits = {}
#         for key1 , key2 in combos:
#             fplus = self.splines_data[("fplus", key1)]
#             f0 = self.splines_data[("f0", key2)]
#             y = { 'fplus': fplus(q2), 'f0': f0(q2)}
#             if not np.isnan(pole_vector):
#                 y['fplus'] = y['fplus'] * (1 - q2/pole_vector**2)
#             if not np.isnan(pole_scalar):
#                 y['f0'] = y['f0'] * (1 - q2/pole_scalar**2)
#             fit = lsqfit.nonlinear_fit(data=(x, y), fcn=self.model_z, p0=gv.mean(prior), prior=prior, **kwargs)
#             fit = serialize.SerializableNonlinearFit(fit)
#             key = ("fplus", key1, "fzero", key2)
#             self.build_fit_splines(fit, use_poles, key=key)
#             fits[key] = fit
#         return fits
#         # Set up and run the fit

#         # y = {'fplus': self.fplus_data(q2), 'f0': self.f0_data(q2)}
#         # pole_vector = self.get_pole('vector', use_poles.vector) * GeV
#         # pole_scalar = self.get_pole('scalar', use_poles.scalar) * GeV

#         # if not np.isnan(pole_vector):
#         #     y['fplus'] = y['fplus'] * (1 - q2/pole_vector**2)
#         # if not np.isnan(pole_scalar):
#         #     y['f0'] = y['f0'] * (1 - q2/pole_scalar**2)

#         # prior = {
#         #     'a': [gv.gvar(0, 100) for _ in range(nterms)],
#         #     'b': [gv.gvar(0, 100) for _ in range(nterms)],
#         # }
#         # fit = lsqfit.nonlinear_fit(data=(x, y), fcn=self.model_z, p0=gv.mean(prior), prior=prior, **kwargs)
#         # self.build_fit_splines(fit, use_poles)
#         # return serialize.SerializableNonlinearFit(fit)
