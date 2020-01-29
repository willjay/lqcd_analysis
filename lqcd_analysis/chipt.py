"""
Functions and classes for modeling semi-leptonic decays of B- and D-mesons using
heavy meson (rooted staggered) chiral perturbation theory.
"""
import re
import numpy as np

def get_value(dict_list, key):
    """
    Gets a value from a list of dicts.
    """
    for adict in dict_list:
        value = adict.get(key)
        if value is not None:
            return value
    raise KeyError("key not found within dict_list.")


def check_duplicate_keys(dict_list):
    """
    Checks for duplicate keys in a list of dicts.
    Raises:
        ValueError, when non-unique keys are found
    """
    all_keys = [list(adict.keys()) for adict in dict_list]  # list of lists
    flat_keys = [key for sublist in all_keys for key in sublist]  # flatten
    unique_keys = np.unique(flat_keys)
    if len(unique_keys) != len(flat_keys):
        raise ValueError("Non-unique keys found in dict_list")


def valid_name(coefficient_name):
    """
    Checks for a valid coefficient name for an "analytic term."
    Some examples of valid coefficients are: c_l, c_h2, c_leha4
    Args:
        coefficient_name: str
    Returns: 
        bool, whether the name is valid
    """
    return bool(re.match(r"c_([lhEa]\d{0,})+", coefficient_name))


def parse_name(coefficient_name):
    """
    Parses the name for a generic coefficient of an "analytic term" into a list
    of (name, power) tuples.
    a2lhE4 --> [('a',2), ('l', 1), ('h', 1), ('E', 4)]
    Args:
        coefficient_name: str,
    Returns:
        list of tuples
    """
    matches = re.findall(r'[lhEa]\d{0,}', coefficient_name)
    result = []
    for match in matches:
        power = 1
        if len(match) > 1:
            power = int(match[1:])
        result.append((match[0], power))
    return result


def analytic_terms(chi, params):
    """
    Computes the sum of analytic terms. Each term is a product of the form:
    coefficient * (chi_l**m) * (chi_h**n) * (chi_a**p) * (chi_E**q).
    Note:
    The expansion parameter chi_E is often an array, so the full some of terms
    will generally be an array, too.
    Args:
        chi: ChiralExpansionParameters
        params: dict of parameters. Parameters that are invalid coefficient 
            names for analytic terms are simply skipped.
    Returns:
        float or array with the result
    """
    result = 0.0
    for name, value in params.items():
        if valid_name(name):
            term = value
            for subscript, power in parse_name(name):
                term *= chi[subscript]**power  # Subscript is l, h, a, or E
            result += term
    return result


def chiral_log_I1(mass, lam):
    """
    Computes the chiral logarithm function I_1.
    """
    lam2 = lam**2.0
    mass2 = mass**2.0
    return mass2 * np.log(mass2 / lam2)


def form_factor_tree_level(gpi, fpi, energy, delta):
    """
    Computes the tree-level expression for the form factor.

    This expression appears (with inconsistent notation) throughout the
    literature. For example,

    Eqs. (3.20) and (3.21) of [https://arxiv.org/abs/1509.06235], J. Bailey et
    al., PRD 93, 025026 (2016) "B -> Kl+l− decay form factors from three-flavor
    lattice QCD".

    Eq. (A2) of [https://arxiv.org/abs/1901.02561], A. Bazavov et al.,
    PRD 100 (2019) no.3, 034501 "Bs -> Klnu decay from lattice QCD.

    Eq. 29 of [https://arxiv.org/abs/0704.0795], Aubin and Bernard,
    PRD 76 (2007) 014002  "Heavy-Light Semileptonic Decays in Staggered
    Chiral Perturbation Theory".

    Eqs. (42) and (43) of [https://arxiv.org/abs/0811.3640] J. Bailey et al.
    PRD 79 (2009) 054507 "B -> pilnu semileptonic form factor from three-flavor
    lattice QCD: A model-independent determination of |Vub|".

    A cautionary word on notation and units:
    The mass dimensions of the parameters in this expression are
    [gpi] = 0
    [fpi] = [energy] = [delta] = +1
    Since [f_parallel] = +1/2 and [f_perp] = -1/2, the formula
    f_J = gpi / (fpi * (energy + delta)) is clearly nonsense as written!
    An additional _dimensionful_ constant is necessary to restore dimensional
    consistency. Typically this extra factor takes the form "Sum_i C_i Chi_i",
    with _dimensionful_ fit parameters C_i. The "Chi_i" factors are always
    dimensionless in the literature.
    """
    return gpi / (fpi * (energy + delta))


class StaggeredPions:
    """
    Wrapper class for collecting the masses of the Goldstone bosons
    and eta mesons of different tastes. Designed for use with results
    on a single ensemble.
    Args:
        mpi5: mass of the pseudoscalar taste "pion", mpi5 =  M_{pi,5}.
        splittings: dict of the staggered taste splittings,
            Delta_xi and Hairpin_{V(A)}
    """

    def __init__(self, mpi5, params):
        self._params = params
        self.mpi_I = self._mpi(mpi5, taste='I')
        self.mpi_P = self._mpi(mpi5, taste='P')
        self.mpi_V = self._mpi(mpi5, taste='V')
        self.mpi_A = self._mpi(mpi5, taste='A')
        self.mpi_T = self._mpi(mpi5, taste='T')
        self.meta_V = self._meta(mpi5, taste='V')
        self.meta_A = self._meta(mpi5, taste='A')

    def _mpi(self, mpi5, taste):
        """
        Computes the pion mass in a different taste, given
        the mass of the pseuoscalar taste pion mpi5 = M_{pi,5}.
        """
        delta = self._params[f'Delta_{taste}']
        return np.sqrt(mpi5**2.0 + delta)

    def _meta(self, mpi5, taste):
        """
        Computes the eta mass in a different taste, given
        the mass of the pseudoscalar taste pion mpi5 = M_{pi},5}.
        """
        mpi = self._mpi(mpi5, taste)
        hairpin = self._params[f'Hairpin_{taste}']
        return np.sqrt(mpi**2.0 + 0.5 * hairpin)

    def __str__(self):
        return (
            "StaggeredPions(\n"
            "    mpi_P  = {mpi_P},\n"
            "    mpi_I  = {mpi_I},\n"
            "    mpi_V  = {mpi_V},\n"
            "    mpi_A  = {mpi_A},\n"
            "    mpi_T  = {mpi_T},\n"
            "    meta_V = {meta_V},\n"
            "    meta_A = {meta_A})".format(**self.__dict__)
        )


class ChiralExpansionParameters:
    """
    Wrapper class for the dimensionless chiral expansion parameters "chi."
    Args:
        x, params: dicts with keys 'm_light', 'm_heavy', 'E', 'mu', 'fpi', and
        'DeltaBar'
    """

    def __init__(self, x, params):
        self.x = x
        self.params = params
        self._validate()

    def _validate(self):
        keys = ['m_light', 'm_heavy', 'E', 'mu', 'fpi', 'DeltaBar']
        for key in keys:
            test = (int(key in self.x) + int(key in self.params))
            if test == 1:
                continue
            elif test == 0:
                raise KeyError(f"Missing key {key}")
            else:
                raise KeyError(f"Key {key} present in x and params.")

    def _get(self, key):
        value = self.x.get(key)
        if value is not None:
            return value
        return self.params.get(key)

    @property
    def light(self):
        """
        Computes the dimenionless expansion parameter defined in Eq (3.22) of
        [https://arxiv.org/abs/1509.06235], J. Bailey et al., PRD 93, 025026
        (2016) "B -> Kl+l− decay form factors from three-flavor lattice QCD."
        """
        mu = self._get('mu')
        fpi = self._get('fpi')
        m_light = self._get('m_light')
        return 3.0 * (2.0 * mu * m_light) / (8.0 * np.pi**2.0 * fpi**2.0)

    @property
    def heavy(self):
        """
        Computes the dimenionless expansion parameter defined in Eq (3.23) of
        [https://arxiv.org/abs/1509.06235], J. Bailey et al., PRD 93, 025026
        (2016) "B -> Kl+l− decay form factors from three-flavor lattice QCD."
        """
        mu = self._get('mu')
        fpi = self._get('fpi')
        m_heavy = self._get('m_heavy')
        return 2.0 * mu * m_heavy / (8.0 * np.pi**2.0 * fpi**2.0)

    @property
    def energy(self):
        """
        Computes the dimenionless expansion parameter defined in Eq (3.24) of
        [https://arxiv.org/abs/1509.06235], J. Bailey et al., PRD 93, 025026
        (2016) "B -> Kl+l− decay form factors from three-flavor lattice QCD."
        """
        fpi = self._get('fpi')
        energy = self._get('E')
        return np.sqrt(2.0) * energy / (4.0 * np.pi * fpi)

    @property
    def a2(self):
        """
        Computes the dimenionless expansion parameter defined in Eq (3.24) of
        [https://arxiv.org/abs/1509.06235], J. Bailey et al., PRD 93, 025026
        (2016) "B -> Kl+l− decay form factors from three-flavor lattice QCD."
        """
        fpi = self._get('fpi')
        delta_bar = self._get('DeltaBar')
        return delta_bar / (8.0 * np.pi**2.0 * fpi**2.0)

    def __getitem__(self, key):
        if key in ('l', 'light'):
            return self.light
        elif key in ('h', 'heavy'):
            return self.heavy
        elif key in ('E', 'energy'):
            return self.energy
        elif key == 'a2':
            return self.a2
        elif key == 'a':
            return np.sqrt(self.a2)
        else:
            raise KeyError(f"Invalid key: {key}")

    def __str__(self):
        return (
            f"ChiralExpansionParamters(\n"
            f"    chi_light  = {self.light}\n"
            f"    chi_heavy  = {self.heavy}\n"
            f"    chi_a2     = {self.a2}\n"
            f"    chi_energy = {self.energy})"
        )


class Scale:
    """
    Wrapper class for scale-setting parameters. By definition, the scale factor
    is the approriate power of physical quantity with mass dimension unity:
    [scale_factor] = -1. Here are some examples with common scale-settings
    parameters:
    [w0] = -1       --> scale_factor = w0**(-1/-1) = w0
    [t0] = -2       --> scale_factor = t0**(-1/-2) = sqrt(t0)
    [sqrt(t0)] = -1 --> scale_factor = sqrt(t0)**(-1/-1) = sqrt(t0)
    [fps] = +1      --> scale_factor = fps**(-1/1) = 1/fps
    """
    def __init__(self, name, value, dim=-1):
        self.name = name
        self.value = value
        self.dim = dim
        self.scale_factor = value**(-1.0 / dim)

    def __str__(self):
        return (
            f"Scale(name='{self.name}', "
            f"value={self.value}, "
            f"dim={self.dim}, "
            f"scale_factor={self.scale_factor})"
        )


class FormFactorData:
    """
    Wrapper class for form factor data.
    """
    def __init__(self, ydata, name, ns, quark_masses):
        self.ydata = ydata
        self.name = name
        self.ns = ns
        self.quark_masses = quark_masses

    def __str__(self):
        return (
            "FormFactorData("
            f"name='{self.name}', "
            f"ns={self.ns}, "
            f"mq={self.quark_masses})"
        )


class ChiralModel:
    """
    General chiral model for semileptonic decays of B and D mesons.
    """
    def __init__(self, form_factor_name, process, lam):
        valid_names = ['f_parallel', 'f_perp', 'f_0', 'f_T']
        valid_processes = [
            'D to pi', 'B to pi',
            'D to K', 'B to K',
            'Ds to K', 'Bs to K']
        if form_factor_name not in valid_names:
            raise ValueError(f"Name must be in {valid_names}")
        if process not in valid_processes:
            raise ValueError(f"Name must be in {valid_processes}")
        self.model_type = "ChiralModel"
        self.form_factor_name = form_factor_name
        self.process = process
        self.lam = lam

    def model(self, *args):
        raise NotImplementedError(
            "model not implemented for generic chiral model"
        )

    def __call__(self, x, params):
        return self.model(x, params)

    def __str__(self):
        return (
            f"{self.model_type}({self.process}; "
            f"{self.form_factor_name}; "
            f"Lambda_UV={self.lam})"
        )


class HardSU2Model(ChiralModel):
    """
    Model for form factor data in hard K/pi limit of SU(2) EFT.
    """
    def __init__(self, form_factor_name, process, lam):
        super().__init__(self, form_factor_name, process, lam)
        self.model_type = "HardSU2Model"

    def delta_logs(self, fpi, pions):
        """
        Computes the full chiral logarithm for SU(2) hard K/pi EFT.
        Note:
        In general, the functional form of the chiral logarithms depend on both
        the process (e.g., 'D to pi') and the form factor (e.g., 'f_0').
        However, in the hard K/pi limit of SU(2) EFT is special because the
        functional form of the logarithms is "universal.""
        """
        result = (
            -1.0 * chiral_log_I1(pions.mpi_I, self.lam)
            - 4.0 * chiral_log_I1(pions.mpi_V, self.lam)
            - 6.0 * chiral_log_I1(pions.mpi_T, self.lam)
            - 4.0 * chiral_log_I1(pions.mpi_A, self.lam)
            - 1.0 * chiral_log_I1(pions.mpi_A, self.lam)
        ) / 16.0
        result += (
            0.25 * chiral_log_I1(pions.mpi_I, self.lam)
            + chiral_log_I1(pions.mpi_V, self.lam)
            + chiral_log_I1(pions.meta_V, self.lam)
            + chiral_log_I1(pions.mpi_A, self.lam)
            + chiral_log_I1(pions.meta_A, self.lam)
        )
        result /= (4.0 * np.pi * fpi)**2.0
        return result

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
        x, params = args if len(args) == 2 else ({}, args)
        dict_list = [x, params]
        # Extract values from inputs
        c_0 = get_value(dict_list, 'c0')
        gpi = get_value(dict_list, 'g')
        fpi = get_value(dict_list, 'fpi')
        mpi5 = get_value(dict_list, 'mpi5')
        energy = get_value(dict_list, 'E')
        delta = get_value(dict_list, 'delta_pole')
        # Get the chiral logarithms
        pions = StaggeredPions(mpi5, params)
        logs = self.delta_logs(fpi, pions)
        # Get the analytic terms
        chi = ChiralExpansionParameters(x, params)
        analytic = analytic_terms(chi, params)
        # Leading-order x (corrections )
        return form_factor_tree_level(gpi, fpi, energy, delta)\
            * (c_0 * (1 + logs) + analytic)
