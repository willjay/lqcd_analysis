"""
Form factors using SU(3) chiral effective theory.
"""
import numpy as np
from . import chipt


class BaseSU3Model(chipt.ChiralModel):
    """
    Base model for SU(3) EFT description of form factors.
    """

    def __init__(self, form_factor_name, process, lam, continuum=False, continuum_logs=False):
        super().__init__(form_factor_name, process, lam, continuum)
        self.model_type = 'BaseSU3Model'
        self.continuum_logs = continuum_logs

    def model(self, *args):
        """
        Compute the model functions
        Args:
            Accepts either a single positional argument 'params' or two
            positional arguments ('x', 'params'). This non-standard interface
            is designed to work well with lsqfit.nonlinear_fit, which accepts
            functions with either interface, depending on whether or not the
            "independent data" have errors.
        Returns:
            array with form factor data
        """
        # Unpackage x, params
        if len(args) not in (1, 2):
            raise TypeError(
                "Please specify either (x, params) or params as arguments."
            )
        x, params = args if len(args) == 2 else ({}, args[0])
        dict_list = [x, params]
        # Extract values from inputs
        leading = chipt.get_value(dict_list, 'leading')
        # c_0 = chipt.get_value(dict_list, 'c0')
        gpi = chipt.get_value(dict_list, 'g')
        fpi = chipt.get_value(dict_list, 'fpi')
        energy = chipt.get_value(dict_list, 'E')
        delta = chipt.get_value(dict_list, 'delta_pole')
        # Collect Goldstone bosons: pions, kaons, and "strangeons"
        goldstones = chipt.GoldstoneBosons(
            chipt.StaggeredPions(x, params, 'mpi5', (self.continuum or self.continuum_logs)),
            chipt.StaggeredPions(x, params, 'mK5', (self.continuum or self.continuum_logs)),
            chipt.StaggeredPions(x, params, 'mS5', (self.continuum or self.continuum_logs)))
        # Get the chiral logarithms
        logs = self.delta_logs(fpi, gpi, goldstones)
        sigma = self.self_energy()
        # Get the analytic terms
        chi = chipt.ChiralExpansionParameters(x, params)
        analytic = chipt.analytic_terms(chi, params, self.continuum)
        # Leading-order x (corrections )
        name = self.form_factor_name
        tree = chipt.form_factor_tree_level(leading, energy, delta, sigma, name)
        return tree * (1 + logs + analytic)
        # return chipt.form_factor_tree_level(gpi, fpi, energy, delta, sigma)\
        #     * (c_0 * (1 + logs) + analytic)

    def breakdown(self, *args):
        # Unpackage x, params
        if len(args) not in (1, 2):
            raise TypeError(
                "Please specify either (x, params) or params as arguments."
            )
        x, params = args if len(args) == 2 else ({}, args[0])
        dict_list = [x, params]
        # Extract values from inputs
        leading = chipt.get_value(dict_list, 'leading')
        # c_0 = chipt.get_value(dict_list, 'c0')
        gpi = chipt.get_value(dict_list, 'g')
        fpi = chipt.get_value(dict_list, 'fpi')
        energy = chipt.get_value(dict_list, 'E')
        delta = chipt.get_value(dict_list, 'delta_pole')
        # Collect Goldstone bosons: pions, kaons, and "strangeons"
        goldstones = chipt.GoldstoneBosons(
            chipt.StaggeredPions(x, params, 'mpi5', (self.continuum or self.continuum_logs)),
            chipt.StaggeredPions(x, params, 'mK5', (self.continuum or self.continuum_logs)),
            chipt.StaggeredPions(x, params, 'mS5', (self.continuum or self.continuum_logs)))
        # Get the chiral logarithms
        logs = self.delta_logs(fpi, gpi, goldstones)
        sigma = self.self_energy()
        # Get the analytic terms
        chi = chipt.ChiralExpansionParameters(x, params)
        analytic = chipt.analytic_terms(chi, params, self.continuum)
        analytic_nlo = chipt.analytic_terms(chi, params, self.continuum, order='NLO')
        analytic_nnlo = chipt.analytic_terms(chi, params, self.continuum, order='NNLO+')        
        # Leading-order x (corrections )
        name = self.form_factor_name
        tree = chipt.form_factor_tree_level(leading, energy, delta, sigma, name)
        return {
            'tree': tree,
            'NLO_log_corrections': tree * logs,
            'NLO_analytic_corrections': tree * analytic_nlo,
            'full_analytic_corrections': tree * analytic,
            'LO': tree,
            'NLO': tree * (logs + analytic_nlo),
            'NNLO': tree * analytic_nnlo,
            'self_energy': self.self_energy(),
        }


class HardSU3Model(BaseSU3Model):
    """
    Model for form factor in hard SU(3) EFT.
    """

    def __init__(self, form_factor_name, process, lam, continuum=False, continuum_logs=False):
        super().__init__(form_factor_name, process, lam, continuum, continuum_logs)
        self.model_type = "HardSU3Model"

    def delta_logs(self, fpi, gpi, goldstones):
        """
        Computes the full combination of chiral logarithms.
        """
        name = self.form_factor_name
        if name in ('f_0', 'f_T'):
            raise NotImplementedError("Logs not yet implementd for ")

        if self.process in ('B to pi', 'D to pi'):
            if name in ('f_parallel', r'f_\parallel'):
                return self._log_b2pi_parallel(fpi, gpi, goldstones)
            if name in ('f_perp', r'f_\perp'):
                return self._log_b2pi_perp(fpi, gpi, goldstones)

        if self.process in ('B to K', 'D to K'):
            if name in ('f_parallel', r'f_\parallel'):
                return self._log_b2k_parallel(fpi, gpi, goldstones)
            if name in ('f_perp', r'f_\perp'):
                return self._log_b2k_perp(fpi, gpi, goldstones)

        if self.process in ('Bs to K', 'Ds to K'):
            raise NotImplementedError(
                f"Logs not yet implemented for {self.process}."
            )
        raise ValueError("Unrecognized process and/or form_factor_name")

    def self_energy(self, *args, **kwargs):
        """
        Compute the self-energy correction for B2X decays, which happens to
        vanish in hard-pion SU(3) effective theory. This result is a special
        case of Eq (A19) of the reference below, which follows after setting
        J1sub to zero, following the replacement specified after Eq (36).
        Reference:
            J. Bailey et al., PRD 93, 025026 (2016)
            "B -> Kl+l− decay form factors from three-flavor lattice QCD".
            [https://arxiv.org/1509.06235].
        """
        return 0.

    def _residue_combo(self, mass, mu):
        """
        Computes the combination of quantities involving the Euclidean
        residue functions
        """
        return (
            chipt.residue_r(mass, mu, 1) *
            chipt.chiral_log_i1(mass[0], self.lam)
            + chipt.residue_r(mass, mu, 2) *
            chipt.chiral_log_i1(mass[1], self.lam)
            + chipt.residue_r(mass, mu, 3) *
            chipt.chiral_log_i1(mass[2], self.lam)
        )

    def _log_b2k(self, goldstones, coefficients):
        """
        Computes the general form of the B2K chiral logarithms for vector form
        factors f_parallel and f_perp, given a set of coefficients. In the
        hard-kaon limit, both form factors have the same functional form up to
        coefficients which depend on fpi and gpi. The implemented equations are
        special cases of Eqs (A2) and (A22) the paper cited below. To obtain
        the equations implemented here, apply the replacements specified after
        Eq (A36) of the same paper to the general case.
        Reference:
            J. Bailey et al., PRD 93, 025026 (2016)
            "B -> Kl+l− decay form factors from three-flavor lattice QCD".
            [https://arxiv.org/1509.06235].
        Args:
            pions, kaons, strangeons: StaggeredPions objects
            coefficients: dict of coefficients
        Returns:
            float, the value of the chiral logarithm
        """
        pions = goldstones.pions
        kaons = goldstones.kaons
        strangeons = goldstones.strangeons
        # Taste-averaged terms
        result = coefficients['taste-averaged:pions'] *\
            chipt.taste_average_i1(pions, self.lam)
        result += coefficients['taste-averaged:kaons'] *\
            chipt.taste_average_i1(kaons, self.lam)
        result += coefficients['taste-averaged:strangeons'] *\
            chipt.taste_average_i1(strangeons, self.lam)
        # Scalar pion terms
        result += coefficients['scalar:pions'] *\
            chipt.chiral_log_i1(pions.m_i, self.lam)
        result += coefficients['scalar:kaons'] *\
            chipt.chiral_log_i1(pions.meta_i, self.lam)
        result += coefficients['scalar:strangeons'] *\
            chipt.chiral_log_i1(strangeons.m_i, self.lam)
        # Vector pion and eta terms
        # Note: mu=m_{S,V}
        result += coefficients['vector:pions'] *\
            self._residue_combo(
                mass=[pions.m_v, pions.meta_v, strangeons.metaprime_v],
                mu=strangeons.m_v)
        # Note: mu=m_{pi,V}
        result += coefficients['vector:strangeons'] *\
            self._residue_combo(
                mass=[strangeons.m_v, pions.meta_v, strangeons.metaprime_v],
                mu=pions.m_v)
        # Axial pion and eta terms
        # Note: mu=m_{S,A}
        result += coefficients['axial:pions'] *\
            self._residue_combo(
                [pions.m_a, pions.meta_a, strangeons.metaprime_a],
                strangeons.m_a)
        # Note: mu=m_{pi,A}
        result += coefficients['vector:strangeons'] *\
            self._residue_combo(
                mass=[strangeons.m_a, pions.meta_a, strangeons.metaprime_a],
                mu=pions.m_a)
        # Normalization
        result *= coefficients['normalization']
        return result

    def _log_b2k_parallel(self, fpi, gpi, goldstones):
        """
        Computes the chiral logarithm associated with the form factor
        f_parallel for the decay B -> K. See _log_B2K(...) for more details.
        Args:
            fpi, gpi: float
            pions, kaons, strangeons: StaggeredPions objects
        Returns:
            float, the value of the chiral logarithm
        """
        g2 = gpi**2.
        coefficients = {
            'taste-averaged:pions': -3. * g2,
            'taste-averaged:kaons': -0.5 * (2. + 3. * g2),
            'taste-averaged:strangeons': -0.5,
            'scalar:pions': 0.75 * g2,
            'scalar:eta': -(8. + 3. * g2) / 12.,
            'scalar:strangeons': 0.5,
            'vector:pions': 1.5 * g2,
            'vector:strangeons': 0.5,
            'axial:pions': 1.5 * g2,
            'axial:strangeons': 0.5,
            'normalization': 1 / (4. * np.pi * fpi)**2.,
        }
        return self._log_b2k(goldstones, coefficients)

    def _log_b2k_perp(self, fpi, gpi, goldstones):
        """
        Computes the chiral logarithm associated with the form factor f_perp
        for the decay B -> K. See _log_B2K(...) for more details.
        Args:
            fpi, gpi: float
            pions, kaons, strangeons: StaggeredPions objects
        Returns:
            float, the value of the chiral logarithm
        """
        g2 = gpi**2.
        coefficients = {
            'taste-averaged:pions': -3. * g2,
            'taste-averaged:kaons': -0.5 * (2. + 3. * g2),
            'taste-averaged:strangeons': -0.5,
            'scalar:pions': 0.75 * g2,
            'scalar:eta': (4. + 3. * g2) / 12.,
            'scalar:strangeons': 0.5,
            'vector:pions': 1.5 * g2,
            'vector:strangeons': 0.5,
            'axial:pions': 1.5 * g2,
            'axial:strangeons': 0.5,
            'normalization': 1 / (4. * np.pi * fpi)**2.,
        }
        return self._log_b2k(goldstones, coefficients)

    def _log_b2pi(self, goldstones, coefficients):
        """
        Computes the general form of the B2pi chiral logarithms for vector form
        factors f_parallel and f_perp, given a set of coefficients. In the
        hard-kaon limit, both form factors have the same functional form up to
        coefficients which depend on fpi and gpi. The implemented equations are
        special cases of Eqs (A1) and (A20) the paper cited below. To obtain
        the equations implemented here, apply the replacements specified after
        Eq (A36) of the same paper to the general case.
        Implementation note:
        --------------------
        The "scalar pion" terms involve the scalar-taste eta meson. This meson
        is hybrid quantity whose mass is a weighted average of the light-quark
        pions (l-lbar) and strange-quark "strangeons" (s-sbar) via Eq (A13)
        m_{eta,I}^2 = 1/3 m_{U,I}^2 + 2/3 m_{S,I}^2. Because of its hybrid
        nature, we compute it locally in this function instead of extracting
        it from a StaggeredPions object.
        Reference:
            J. Bailey et al., PRD 93, 025026 (2016)
            "B -> Kl+l− decay form factors from three-flavor lattice QCD".
            [https://arxiv.org/1509.06235].
        Args:
            pions, kaons, strangeons: StaggeredPions objects
            coefficients: dict of coefficients
        Returns:
            float, the value of the chiral logarithm
        """
        pions = goldstones.pions
        kaons = goldstones.kaons
        strangeons = goldstones.strangeons
        meta_scalar = np.sqrt((pions.m_i**2. + 2. * strangeons.m_i**2.) / 3.)
        # Taste-averaged terms
        result = coefficients['taste-averaged'] * (
            2. * chipt.taste_average_i1(pions, self.lam)
            + chipt.taste_average_i1(kaons, self.lam)
        )
        # Scalar pion terms
        result += coefficients['scalar'] * (
            3. * chipt.chiral_log_i1(pions.m_i, self.lam)
            - chipt.chiral_log_i1(meta_scalar, self.lam)
        )
        # Vector pion and eta terms
        if not self.continuum:
            # These terms vanish identically in the continuum
            result += coefficients['vector'] * self._residue_combo(
                mass=[pions.m_v, pions.meta_v, strangeons.metaprime_v],
                mu=strangeons.m_v)

        # Axial pion and eta terms
        if not self.continuum:
            # These terms vanish identically in the continuum
            result += coefficients['axial'] * self._residue_combo(
                mass=[pions.m_a, pions.meta_a, strangeons.metaprime_a],
                mu=strangeons.m_a)
        # Normalization
        result *= coefficients['normalization']
        return result

    def _log_b2pi_parallel(self, fpi, gpi, goldstones):
        """
        Computes the chiral logarithm associated with the form factor
        f_parallel for the decay B -> pi See _log_B2pi(...) for more details.
        Args:
            fpi, gpi: float
            pions, kaons, strangeons: StaggeredPions objects
        Returns:
            float, the value of the chiral logarithm
        """
        g2 = gpi**2.
        if self.continuum:
            hairpin_v = 0
            hairpin_a = 0
        else:
            hairpin_v = goldstones.pions['Hairpin_V']
            hairpin_a = goldstones.pions['Hairpin_A']
        cofficients = {
            'taste-averaged': -(1. + 3. * g2) / 2.,
            'scalar': (1. + 3. * g2) / 12.,
            'vector': 1.5 * (1. + g2) * hairpin_v,
            'axial': 1.5 * (1. + g2) * hairpin_a,
            'normalization': 1 / (4. * np.pi * fpi)**2.,
        }
        return self._log_b2pi(goldstones, cofficients)

    def _log_b2pi_perp(self, fpi, gpi, goldstones):
        """
        Computes the chiral logarithm associated with the form factor f_perp
        for the decay B -> pi See _log_B2pi(...) for more details.
        Args:
            fpi, gpi: float
            pions, kaons, strangeons: StaggeredPions objects
        Returns:
            float, the value of the chiral logarithm
        """
        g2 = gpi**2.
        if self.continuum:
            hairpin_v = 0
            hairpin_a = 0
        else:
            hairpin_v = goldstones.pions['Hairpin_V']
            hairpin_a = goldstones.pions['Hairpin_A']
        cofficients = {
            'taste-averaged': -(1. + 3. * g2) / 2.,
            'scalar': (1. + 3. * g2) / 12.,
            'vector': (1. + 3. * g2) / 2. * hairpin_v,
            'axial': (1. + 3. * g2) / 2. * hairpin_a,
            'normalization': 1 / (4. * np.pi * fpi)**2.,
        }
        return self._log_b2pi(goldstones, cofficients)
