"""
Form factors using SU(2) chiral effective theory.
"""
import numpy as np
from . import chipt


class BaseSU2Model(chipt.ChiralModel):
    """
    Base model for SU(2) EFT description of form factors.
    """

    def __init__(self, form_factor_name, process, lam, continuum=False):
        super().__init__(form_factor_name, process, lam, continuum)
        self.model_type = 'BaseSU2Model'

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
        c_0 = chipt.get_value(dict_list, 'c0')
        gpi = chipt.get_value(dict_list, 'g')
        fpi = chipt.get_value(dict_list, 'fpi')
        energy = chipt.get_value(dict_list, 'E')
        delta = chipt.get_value(dict_list, 'delta_pole')
        # Get the chiral logarithms
        pions = chipt.StaggeredPions(x, params, continuum=self.continuum)
        logs = self.delta_logs(fpi, gpi, pions, energy)
        sigma = self.self_energy(fpi, gpi, pions, energy)
        # Get the analytic terms
        chi = chipt.ChiralExpansionParameters(x, params)
        analytic = chipt.analytic_terms(chi, params, self.continuum)
        # Leading-order x (corrections )
        return chipt.form_factor_tree_level(gpi, fpi, energy, delta, sigma)\
            * (c_0 * (1 + logs) + analytic)


class HardSU2Model(BaseSU2Model):
    """
    Model for form factor data in hard K/pi limit of SU(2) EFT.
    """

    def __init__(self, form_factor_name, process, lam, continuum=False):
        super().__init__(form_factor_name, process, lam, continuum)
        self.model_type = "HardSU2Model"

    def self_energy(self, *args, **kwargs):
        """
        Compute the self-energy correction, which vanishes in hard SU(2) EFT.
        """
        return 0.0

    def delta_logs(self, fpi, gpi, pions, *args):
        """
        Computes the full chiral logarithm for SU(2) hard K/pi EFT.
        Note:
        In general, the functional form of the chiral logarithms depend on both
        the process (e.g., 'D to pi') and the form factor (e.g., 'f_0').
        However, in the hard K/pi limit of SU(2) EFT is special because the
        functional form of the logarithms is "universal," up to an overall
        multiplicative constant.
        """
        g2 = gpi**2.

        def combo(mass):
            return chipt.chiral_log_i1(mass, self.lam)
        # Taste-averaged terms
        result = -1. * chipt.taste_average_i1(pions, self.lam)
        # Scalar pion terms
        result += 0.25 * chipt.chiral_log_i1(pions.m_i, self.lam)
        # Vector pion and eta terms
        result += combo(pions.m_v) - combo(pions.meta_v)
        # Axial pion and eta terms
        result += combo(pions.m_a) - combo(pions.meta_a)
        # Normalization
        result /= (4. * np.pi * fpi)**2.
        if self.process in ('B to pi', 'D to pi'):
            return (1. + 3. * g2) * result
        if self.process in ('B to K', 'D to K'):
            return 3. * g2 * result
        if self.process in ('Bs to K', 'Ds to K'):
            # TODO: Conjecture! Verify to be sure this is correct!
            return result
            # raise NotImplementedError(
            #     f"Logs not yet implemented for {self.process}."
            # )


class SU2Model(BaseSU2Model):
    """
    Model for form factors in general SU(2) EFT, away from the hard K/pi limit.
    """

    def __init__(self, form_factor_name, process, lam, continuum=False):
        super().__init__(form_factor_name, process, lam, continuum)
        self.model_type = "SU2Model"

    def delta_logs(self, fpi, gpi, pions, energy):
        """
        Computes the full combination of chiral logarithms.
        """
        name = self.form_factor_name
        if name in ('f_0'):
            raise NotImplementedError(f"Logs not yet implementd for {name}")

        if self.process in ('B to pi', 'D to pi'):
            if name in ('f_parallel', r'f_\parallel'):
                return self._log_b2pi_parallel(fpi, gpi, pions, energy)
            if name in ('f_perp', r'f_\perp', 'f_T'):
                return self._log_b2pi_perp(fpi, gpi, pions, energy)

        if self.process in ('B to K', 'D to K'):
            if name in ('f_parallel', r'f_\parallel'):
                return self._log_b2k_parallel(fpi, gpi, pions)
            if name in ('f_perp', r'f_\perp', 'f_T'):
                return self._log_b2k_perp(fpi, gpi, pions)

        if self.process in ('Bs to K', 'Ds to K'):
            raise NotImplementedError(
                f"Logs not yet implemented for {self.process}."
            )
        raise ValueError("Unrecognized process and/or form_factor_name")

    def self_energy(self, fpi, gpi, pions, energy):
        """
        Computes the self-energy corrections.
        """
        if self.process in ('B to pi', 'D to pi'):
            return self._self_energy_b2pi(fpi, gpi, pions, energy)
        if self.process in ('B to K', 'D to K'):
            return self._self_energy_b2k()
        if self.process in ('Bs to K', 'Ds to K'):
            raise NotImplementedError(
                f"Self energy not yet implemented for {self.process}."
            )
        raise ValueError("Unrecognized process and/or form_factor_name")

    def _log_b2pi_parallel(self, fpi, gpi, pions, energy):
        """
        Computes the chiral logarithm associated with the form factor
        f_parallel for B to pi semileptonic decays. See Eq. (A27) in
        J. Bailey et al.,
        "B -> Kl+l- Decay Form Factors from Three-Flavor Lattice QCD"
        Phys.Rev. D93 (2016) no.2, 025026 [arXiv:1509.06235]
        """
        g2 = gpi**2.

        def combo(mass):
            return (
                3. * (g2 - 1.) * chipt.chiral_log_i1(mass, self.lam)
                - 4. * chipt.chiral_log_i2(mass, energy, self.lam)
            )
        # Taste-averaged terms
        result = (1. - 3. * g2) * chipt.taste_average_i1(pions, self.lam)
        result += 2. * chipt.taste_average_i2(pions, energy, self.lam)
        # Scalar pion terms
        result += (1. + 3. * g2) / 4. * \
            chipt.chiral_log_i1(pions.m_i, self.lam)
        # Vector pion and eta terms
        result += combo(pions.m_v) - combo(pions.meta_v)
        # Axial pion and eta terms
        result += combo(pions.m_a) - combo(pions.meta_a)
        # Normalization
        result /= (4. * np.pi * fpi)**2.
        return result

    def _log_b2k_parallel(self, fpi, gpi, pions):
        """
        Computes the chiral logarithm associated with the form factor
        f_parallel for B to K semileptonic decays. See Eq. (A28) in
        J. Bailey et al.,
        "B -> Kl+l- Decay Form Factors from Three-Flavor Lattice QCD"
        Phys.Rev. D93 (2016) no.2, 025026 [arXiv:1509.06235]
        """
        g2 = gpi**2.

        def combo(mass):
            return 3. * g2 * chipt.chiral_log_i1(mass, self.lam)
        # Taste-averaged terms
        result = -3. * g2 * chipt.taste_average_i1(pions, self.lam)
        # Scalar pion terms
        result += 3. * g2 / 4. * chipt.chiral_log_i1(pions.m_i, self.lam)
        # Vector pion and eta terms
        result += combo(pions.m_v) - combo(pions.meta_v)
        # Axial pion and eta terms
        result += combo(pions.m_a) - combo(pions.meta_a)
        # Normalization
        result /= (4. * np.pi * fpi)**2.
        return result

    def _log_b2pi_perp(self, fpi, gpi, pions, energy):
        """
        Computes the chiral logarithm associated with the form factor f_perp
        for B to pi semileptonic decays. See Eq. (A32) in J. Bailey et al
        "B -> Kl+l- Decay Form Factors from Three-Flavor Lattice QCD"
        Phys.Rev. D93 (2016) no.2, 025026 [arXiv:1509.06235]
        """
        g2 = gpi**2.

        def combo(mass):
            return (
                2. * g2 * chipt.chiral_log_j1sub(mass, energy, self.lam)
                + (1. + 3. * g2) * chipt.chiral_log_i1(mass, self.lam)
            )
        # Taste-averaged terms
        result = -(1. + 3. * g2) * chipt.taste_average_i1(pions, self.lam)
        # Scalar pion terms
        result -= 0.5 * g2 * \
            chipt.chiral_log_j1sub(pions.m_i, energy, self.lam)
        result += (1. + 3. * g2) / 4. * \
            chipt.chiral_log_i1(pions.m_i, self.lam)
        # Vector pion and eta terms
        result += combo(pions.m_v) - combo(pions.meta_v)
        # Axial pion and eta terms
        result += combo(pions.m_a) - combo(pions.meta_a)
        # Normalization
        result *= 3 * g2 * energy / (4. * np.pi * fpi)**2.
        return result

    def _log_b2k_perp(self, fpi, gpi, pions):
        """
        Computes the chiral logarithm associated with the form factor f_perp
        for B to K semileptonic decays. See Eq. (A34) in J. Bailey et al
        "B -> Kl+l- Decay Form Factors from Three-Flavor Lattice QCD"
        Phys.Rev. D93 (2016) no.2, 025026 [arXiv:1509.6235]
        """
        g2 = gpi**2.

        def combo(mass):
            return 3. * g2 * chipt.chiral_log_i1(mass, self.lam)
        # Taste-averaged terms
        result = -3. * g2 * chipt.taste_average_i1(pions, self.lam)
        # Scalar pion terms
        result += 3. * g2 / 4. * chipt.chiral_log_i1(pions.m_i, self.lam)
        # Vector pion and eta terms
        result += combo(pions.m_v) - combo(pions.meta_v)
        # Axial pion and eta terms
        result += combo(pions.m_a) - combo(pions.meta_a)
        # Normalization
        result /= (4. * np.pi * fpi)**2.
        return result

    def _self_energy_b2pi(self, fpi, gpi, pions, energy):
        g2 = gpi**2.

        def combo(mass):
            return 2. * chipt.chiral_log_j1sub(mass, energy, self.lam)
        # Taste-averaged terms
        result = 2. * chipt.taste_average_j1sub(pions, energy, self.lam)
        # Scalar pion terms
        result -= 0.5 * chipt.chiral_log_j1sub(pions.m_i, energy, self.lam)
        # Vector pion and eta terms
        result -= 2. * (combo(pions.m_v) - combo(pions.meta_v))
        # Axial pion and eta terms
        result -= 2. * (combo(pions.m_a) - combo(pions.meta_a))
        # Normalization
        result *= -3. * g2 * energy / (4. * np.pi * fpi)**2.
        return result

    def _self_energy_b2k(self):
        """
        Computes the self-energy contribution for B to K semileptonic decays,
        which happens to vanish in SU(2) EFT. See Eq. (A33) in J. Bailey et al
        "B -> Kl+l- Decay Form Factors from Three-Flavor Lattice QCD"
        Phys.Rev. D93 (2016) no.2, 025026 [arXiv:1509.6235]
        """
        return 0.
