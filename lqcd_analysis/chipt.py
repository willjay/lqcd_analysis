"""
Functions and classes for modeling semi-leptonic decays of B- and D-mesons using
heavy meson (rooted staggered) chiral perturbation theory.
"""
import re
import numpy as np
from . import analysis

# pylint: disable=invalid-name


def get_value(dict_list, key):
    """
    Gets a value from a list of dicts.
    """
    for adict in dict_list:
        value = adict.get(key)
        if value is not None:
            return value
    raise KeyError(f"Key '{key}' not found within dict_list.")


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


def valid_name(coefficient_name, continuum=False):
    """
    Checks for a valid coefficient name for an "analytic term."
    Some examples of valid coefficients are: c_l, c_h2, c_leha4
    Args:
        coefficient_name: str
        continuum: whether or not to accept terms with 'a' for lattice spacing
    Returns:
        bool, whether the name is valid
    """
    if continuum:
        return bool(re.match(r"c_([lhE]\d{0,})+", coefficient_name))
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


def analytic_terms(chi, params, continuum=False):
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
        continuum: bool, whether or not to skip terms involving chi_a
    Returns:
        float or array with the result
    """
    result = 0.0
    for name, value in params.items():
        if valid_name(name, continuum):
            term = value
            for subscript, power in parse_name(name):
                term *= chi[subscript]**power  # Subscript is l, h, a, or E
            result += term
    return result


def chiral_log_I1(mass, lam):
    """
    Computes the chiral logarithm function I_1.
    See Eq. 46 of Aubin and Bernard
    "Heavy-light semileptonic decays in staggered chiral perturbation theory"
    Phys.Rev. D76 (2007) 014002 [arXiv:0704.0795]
    """
    lam2 = lam**2.0
    mass2 = mass**2.0
    return mass2 * np.log(mass2 / lam2)


def chiral_log_I2(mass, delta, lam):
    """
    Computes the chiral logarithm function I_2.
    See Eq. 47 of Aubin and Bernard,
    "Heavy-light semileptonic decays in staggered chiral perturbation theory"
    Phys.Rev. D76 (2007) 014002 [arXiv:0704.0795]
    """
    lam2 = lam**2.0
    mass2 = mass**2.0
    delta2 = delta**2.0
    x = mass / delta
    return 2.0 * delta2 * (1.0 - np.log(mass2 / lam2) - 2.0 * chiral_log_F(x))


def chiral_log_J1(mass, delta, lam):
    """
    Computes the chiral logarithm function J_1.
    See Eq. 48 of Aubin and Bernard,
    "Heavy-light semileptonic decays in staggered chiral perturbation theory"
    Phys.Rev. D76 (2007) 014002 [arXiv:0704.0795]
    """
    mass2 = mass**2.0
    delta2 = delta**2.0
    lam2 = lam**2.0
    x = mass / delta
    return (-mass2 + 2. / 3. * delta2) * np.log(mass2 / lam2)\
        + (4. / 3.) * (delta2 - mass2) * chiral_log_F(x)\
        - (10. / 9.) * delta2\
        + (4. / 3.) * mass2


def chiral_log_F(x):
    """
    Computes the function F(x) appearing alongside chiral logarithms.
    See Eq. 49 of Aubin and Bernard,
    "Heavy-light semileptonic decays in staggered chiral perturbation theory"
    Phys.Rev. D76 (2007) 014002 [arXiv:0704.0795]
    """
    if x < 0:
        raise ValueError("chiral_log_F(x) needs x >= 0.")
    if x > 1:
        root = np.sqrt(x**2.0 - 1)
        return -1.0 * root * np.arctan(root)  # Note: Usual inverse tangent
    else:  # x in [0,1]
        root = np.sqrt(1.0 - x**2.0)
        return root * np.arctanh(root)  # Note: hyperbolic inverse tangent


def chiral_log_J1sub(mass, delta, lam):
    """
    Computes the subtracted chiral logarithm function J_1, which cancels the
    singularity when delta --> 0.
    See Eq. 51 of Aubin and Bernard,
    "Heavy-light semileptonic decays in staggered chiral perturbation theory"
    Phys.Rev. D76 (2007) 014002 [arXiv:0704.0795]
    """
    factor = 2.0 * np.pi / 3.0
    mass3 = mass**3.0
    return chiral_log_J1(mass, delta, lam) - factor * mass3 / delta


def residue_R(mass, mu, j):
    """
    Computes the Euclidean residue function R^[n,k]_j({mass},{mu}).
    See Eq. (B4) of Aubin and Bernard,
    "Heavy-light semileptonic decays in staggered chiral perturbation theory"
    Phys.Rev. D76 (2007) 014002 [arXiv:0704.0795]
    """
    n = len(mass)
    k = len(mu) if hasattr(mu, '__len__') else 1
    if n <= k:
        raise ValueError(f"residue_r requires n > k, found n={n}, k={k}.")
    if (j < 1) or (j > n):
        raise ValueError(f"residue_R requires j in 0, 1, ..., n, found j={j}")
    idx_j = j - 1  # Convert from physics 1-indexing to python 0-indexing
    mu2 = np.array(mu)**2.0
    mass2 = np.array(mass)**2.0
    mass2hat = np.delete(mass2, idx_j)  # mass2 but without the jth element
    upper = np.product(mu2 - mass2[idx_j])
    lower = np.product(mass2hat - mass2[idx_j])
    if np.isclose(lower, 0.0, rtol=1e-6):
        raise ValueError(f"Singular residue_R with j={j}")
    return upper / lower


def taste_average(fcn, pions, *args, **kwargs):
    """
    Computes the taste-averaged evaluation of the specified function.
    Args:
        fcn: function
        pions: StaggeredPions object
        args, kwargs: any positional or keyword arguments to pass to fcn
    Returns:
        taste-averaged evaulation of fcn
    """
    return (
        - 1.0 * fcn(pions.mpi_I, *args, **kwargs)
        - 4.0 * fcn(pions.mpi_V, *args, **kwargs)
        - 6.0 * fcn(pions.mpi_T, *args, **kwargs)
        - 4.0 * fcn(pions.mpi_A, *args, **kwargs)
        - 1.0 * fcn(pions.mpi_A, *args, **kwargs)
    ) / 16.0


def taste_average_I1(pions, lam):
    """
    Computes the taste-averaged evaluation of I1.
    """
    return taste_average(chiral_log_I1, pions, lam)


def taste_average_I2(pions, delta, lam):
    """
    Computes the taste-averaged evaluation of I2.
    """
    return taste_average(chiral_log_I2, pions, delta, lam)


def taste_average_J1sub(pions, delta, lam):
    """
    Computes the taste-averaged evaluation of J1sub.
    """
    return taste_average(chiral_log_J1sub, pions, delta, lam)


def form_factor_tree_level(gpi, fpi, energy, delta, self_energy=0.0):
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
    return gpi / (fpi * (energy + delta + self_energy))


class StaggeredPions:
    """
    Wrapper class for collecting the masses of the Goldstone bosons
    and eta mesons of different tastes. Designed for use with results
    on a single ensemble.
    Args:
        mpi5: mass of the pseudoscalar taste "pion", mpi5 =  M_{pi,5}.
        splittings: dict of the staggered taste splittings,
            Delta_xi and Hairpin_{V(A)}
        continuum: bool, whether or not to use continuum expressions (where the
            taste splittings vanish)
    """

    def __init__(self, x, params, continuum=False):
        self.dict_list = [x, params]
        self.continuum = continuum
        self.mpi_I = self._mpi(taste='I')
        self.mpi_P = self._mpi(taste='P')
        self.mpi_V = self._mpi(taste='V')
        self.mpi_A = self._mpi(taste='A')
        self.mpi_T = self._mpi(taste='T')
        self.meta_V = self._meta(taste='V')
        self.meta_A = self._meta(taste='A')

    def _mpi(self, taste):
        """
        Computes the pion mass in a different taste, given the mass of the
        "pion" with pseuoscalar taste:
        m^2_{ij,xi} = mu (m_i + m_j) + a^2 Delta_xi
        where
        mu (m_i + m_j) = "pion with pseudoscalar taste" = m^2_{ij,5}.

        Eq. (A8) of [https://arxiv.org/abs/1901.02561], A. Bazavov et al.,
        PRD 100 (2019) no.3, 034501 "Bs -> Klnu decay from lattice QCD."
        """
        mpi5 = get_value(self.dict_list, f'mpi5')
        if self.continuum:
            return mpi5
        delta = get_value(self.dict_list, f'Delta_{taste}')
        arg = mpi5**2.0 + delta
        if arg < 0:
            raise ValueError(
                f"Negative argument encountered using 'Delta_{taste}'."
            )
        return np.sqrt(arg)

    def _meta(self, taste):
        """
        Computes the eta mass in a different taste, given the mass of the "pion"
        with pseudoscalar taste.
        m^2_{eta,V(A)} = m^2_{uu,V(A)} + 1/2 a^2 delta^prime_{V(A)},
        where
        "a^2 delta^prime_{V(A)}" = "Hairpin_V(A)".
        Eq. (A5b) of [https://arxiv.org/abs/1901.02561], A. Bazavov et al.,
        PRD 100 (2019) no.3, 034501 "Bs -> Klnu decay from lattice QCD."
        """
        mpi = self._mpi(taste)
        if self.continuum:
            return mpi
        hairpin = get_value(self.dict_list, f'Hairpin_{taste}')
        arg = mpi**2.0 + 0.5 * hairpin
        if arg < 0:
            raise ValueError(
                f"Negative argument encountered using 'Hairpin_{taste}'."
            )
        return np.sqrt(arg)

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

    def unpackage_quark_masses(self):
        """
        Unpackages a pair of quark masses associated with the daughter meson (a
        K or pi). When the daughter is a pion, the masses are equal, and we set
        m_heavy to zero, since the chiral fit should NOT include any analytic
        dependence on the heavy "strange quark mass."
        """
        (m_light, m_heavy) = np.sort(self.quark_masses)
        if m_heavy == m_light:
            m_heavy = 0.0
        return m_light, m_heavy

    def unpackage_ydata(self, m_daughter):
        """
        Unpackages ydata stored as a dict into arrays of x and y data.
        Example: ydata = {'p000': value0, 'p100': value1, ...}
        Args:
            m_daughter: float, mass of the daughter K/pi
        Returns:
            x,y : (boosted energy of the daughter, form factor values)
        """
        y = np.zeros(len(self.ydata), dtype=object)
        x = np.zeros(len(self.ydata), dtype=object)
        for idx, (ptag, form_factor) in enumerate(self.ydata.items()):
            p2 = analysis.p2(ptag, self.ns)
            energy = np.sqrt(m_daughter**2.0 + p2)
            y[idx] = form_factor
            x[idx] = energy
        idxs = np.argsort(x)
        return x[idxs], y[idxs]


class ChiralModel:
    """
    General chiral model for semileptonic decays of B and D mesons.
    """

    def __init__(self, form_factor_name, process, lam, continuum=False):
        valid_names = [
            'f_parallel', r'f_\parallel', 'f_perp', r'f_\perp', 'f_0', 'f_T'
        ]
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
        self.continuum = continuum

    def model(self, *args):
        """
        Computes the model function
        """
        raise NotImplementedError(
            "model not implemented for generic chiral model"
        )

    def __call__(self, *args):
        return self.model(*args)

    def __str__(self):
        return (
            f"{self.model_type}({self.process}; "
            f"{self.form_factor_name}; "
            f"Lambda_UV={np.round(self.lam, decimals=4)}; "
            f"continuum={self.continuum})"
        )


class HardSU2Model(ChiralModel):
    """
    Model for form factor data in hard K/pi limit of SU(2) EFT.
    """

    def __init__(self, form_factor_name, process, lam, continuum=False):
        super().__init__(form_factor_name, process, lam, continuum)
        self.model_type = "HardSU2Model"

    def delta_logs(self, fpi, pions):
        """
        Computes the full chiral logarithm for SU(2) hard K/pi EFT.
        Note:
        In general, the functional form of the chiral logarithms depend on both
        the process (e.g., 'D to pi') and the form factor (e.g., 'f_0').
        However, in the hard K/pi limit of SU(2) EFT is special because the
        functional form of the logarithms is "universal."
        """
        def combo(mass):
            return chiral_log_I1(mass, self.lam)
        # Taste-averaged terms
        result = taste_average_I1(pions, self.lam)
        # Scalar pion terms
        result += 0.25 * chiral_log_I1(pions.mpi_I, self.lam)
        # Vector pion and eta terms
        result += combo(pions.mpi_V) - combo(pions.meta_V)
        # Axial pion and eta terms
        result += combo(pions.mpi_A) - combo(pions.meta_A)
        # Normalization
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
        x, params = args if len(args) == 2 else ({}, args[0])
        dict_list = [x, params]
        # Extract values from inputs
        c_0 = get_value(dict_list, 'c0')
        gpi = get_value(dict_list, 'g')
        fpi = get_value(dict_list, 'fpi')
        energy = get_value(dict_list, 'E')
        delta = get_value(dict_list, 'delta_pole')
        # Get the chiral logarithms
        pions = StaggeredPions(x, params, self.continuum)
        logs = self.delta_logs(fpi, pions)
        # Get the analytic terms
        chi = ChiralExpansionParameters(x, params)
        analytic = analytic_terms(chi, params, self.continuum)
        # Leading-order x (corrections )
        return form_factor_tree_level(gpi, fpi, energy, delta)\
            * (c_0 * (1 + logs) + analytic)


class SU2Model(ChiralModel):
    """
    Model for form factor data in hard K/pi limit of SU(2) EFT.
    """

    def __init__(self, form_factor_name, process, lam, continuum=False):
        super().__init__(form_factor_name, process, lam, continuum)
        self.model_type = "SU2Model"

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
        c_0 = get_value(dict_list, 'c0')
        gpi = get_value(dict_list, 'g')
        fpi = get_value(dict_list, 'fpi')
        energy = get_value(dict_list, 'E')
        delta = get_value(dict_list, 'delta_pole')
        # Get the chiral logarithms
        pions = StaggeredPions(x, params, self.continuum)
        logs = self.delta_logs(fpi, gpi, pions, energy)
        self_energy = self.self_energy(fpi, gpi, pions, energy)
        # Get the analytic terms
        chi = ChiralExpansionParameters(x, params)
        analytic = analytic_terms(chi, params, self.continuum)
        # Leading-order x (corrections )
        return form_factor_tree_level(gpi, fpi, energy, delta, self_energy)\
            * (c_0 * (1 + logs) + analytic)

    def delta_logs(self, fpi, gpi, pions, energy):
        """
        Computes the full combination of chiral logarithms.
        """
        name = self.form_factor_name
        if name in ('f_0', 'f_T'):
            raise NotImplementedError("Logs not yet implementd for ")

        if self.process in ('B to pi', 'D to pi'):
            if name in ('f_parallel', r'f_\parallel'):
                return self._log_B2pi_parallel(fpi, gpi, pions, energy)
            if name in ('f_perp', r'f_\perp'):
                return self._log_B2pi_perp(fpi, gpi, pions, energy)

        if self.process in ('B to K', 'D to K'):
            if name in ('f_parallel', r'f_\parallel'):
                return self._log_B2K_parallel(fpi, gpi, pions)
            if name in ('f_perp', r'f_\perp'):
                return self._log_B2K_perp(fpi, gpi, pions)

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
            return self._self_energy_B2pi(fpi, gpi, pions, energy)
        if self.process in ('B to K', 'D to K'):
            return self._self_energy_B2K()
        if self.process in ('Bs to K', 'Ds to K'):
            raise NotImplementedError(
                f"Self energy not yet implemented for {self.process}."
            )
        raise ValueError("Unrecognized process and/or form_factor_name")

    def _log_B2pi_parallel(self, fpi, gpi, pions, energy):
        """
        Computes the chiral logarithm associated with the form factor f_parallel
        for B to pi semileptonic decays. See Eq. (A27) in J. Bailey et al
        "B -> Kl+l- Decay Form Factors from Three-Flavor Lattice QCD"
        Phys.Rev. D93 (2016) no.2, 025026 [arXiv:1509.06235]
        """
        g2 = gpi**2.0

        def combo(mass):
            return (
                3.0 * (g2 - 1.0) * chiral_log_I1(mass, self.lam)
                - 4.0 * chiral_log_I2(mass, energy, self.lam)
            )
        # Taste-averaged terms
        result = (1.0 - 3.0 * g2) * taste_average_I1(pions, self.lam)
        result += 2.0 * taste_average_I2(pions, energy, self.lam)
        # Scalar pion terms
        result += (1.0 + 3.0 * g2) / 4.0 * chiral_log_I1(pions.mpi_I, self.lam)
        # Vector pion and eta terms
        result += combo(pions.mpi_V) - combo(pions.meta_V)
        # Axial pion and eta terms
        result += combo(pions.mpi_A) - combo(pions.meta_A)
        # Normalization
        result /= (4.0 * np.pi * fpi)**2.0
        return result

    def _log_B2K_parallel(self, fpi, gpi, pions):
        """
        Computes the chiral logarithm associated with the form factor f_parallel
        for B to K semileptonic decays. See Eq. (A28) in J. Bailey et al
        "B -> Kl+l- Decay Form Factors from Three-Flavor Lattice QCD"
        Phys.Rev. D93 (2016) no.2, 025026 [arXiv:1509.06235]
        """
        g2 = gpi**2.0

        def combo(mass):
            return 3.0 * g2 * chiral_log_I1(mass, self.lam)
        # Taste-averaged terms
        result = -3.0 * g2 * taste_average_I1(pions, self.lam)
        # Scalar pion terms
        result += 3.0 * g2 / 4.0 * chiral_log_I1(pions.mpi_I, self.lam)
        # Vector pion and eta terms
        result += combo(pions.mpi_V) - combo(pions.meta_V)
        # Axial pion and eta terms
        result += combo(pions.mpi_A) - combo(pions.meta_A)
        # Normalization
        result /= (4.0 * np.pi * fpi)**2.0
        return result

    def _log_B2pi_perp(self, fpi, gpi, pions, energy):
        """
        Computes the chiral logarithm associated with the form factor f_perp
        for B to pi semileptonic decays. See Eq. (A32) in J. Bailey et al
        "B -> Kl+l- Decay Form Factors from Three-Flavor Lattice QCD"
        Phys.Rev. D93 (2016) no.2, 025026 [arXiv:1509.06235]
        """
        g2 = gpi**2.0

        def combo(mass):
            return (
                2.0 * g2 * chiral_log_J1sub(mass, energy, self.lam)
                + (1.0 + 3.0 * g2) * chiral_log_I1(mass, self.lam)
            )
        # Taste-averaged terms
        result = -1.0 * (1.0 + 3.0 * g2) * taste_average_I1(pions, self.lam)
        # Scalar pion terms
        result -= 0.5 * g2 * chiral_log_J1sub(pions.mpi_I, energy, self.lam)
        result += (1.0 + 3.0 * g2) / 4.0 * chiral_log_I1(pions.mpi_I, self.lam)
        # Vector pion and eta terms
        result += combo(pions.mpi_V) - combo(pions.meta_V)
        # Axial pion and eta terms
        result += combo(pions.mpi_A) - combo(pions.meta_A)
        # Normalization
        result *= 3 * g2 * energy / (4.0 * np.pi * fpi)**2.0
        return result

    def _log_B2K_perp(self, fpi, gpi, pions):
        """
        Computes the chiral logarithm associated with the form factor f_perp
        for B to K semileptonic decays. See Eq. (A34) in J. Bailey et al
        "B -> Kl+l- Decay Form Factors from Three-Flavor Lattice QCD"
        Phys.Rev. D93 (2016) no.2, 025026 [arXiv:1509.06235]
        """
        g2 = gpi**2.0

        def combo(mass):
            return 3.0 * g2 * chiral_log_I1(mass, self.lam)
        # Taste-averaged terms
        result = -3.0 * g2 * taste_average_I1(pions, self.lam)
        # Scalar pion terms
        result += 3.0 * g2 / 4.0 * chiral_log_I1(pions.mpi_I, self.lam)
        # Vector pion and eta terms
        result += combo(pions.mpi_V) - combo(pions.meta_V)
        # Axial pion and eta terms
        result += combo(pions.mpi_A) - combo(pions.meta_A)
        # Normalization
        result /= (4.0 * np.pi * fpi)**2.0
        return result

    def _self_energy_B2pi(self, fpi, gpi, pions, energy):
        g2 = gpi**2.0

        def combo(mass):
            return 2.0 * chiral_log_J1sub(mass, energy, self.lam)
        # Taste-averaged terms
        result = 2.0 * taste_average_J1sub(pions, energy, self.lam)
        # Scalar pion terms
        result -= 0.5 * chiral_log_J1sub(pions.mpi_I, energy, self.lam)
        # Vector pion and eta terms
        result -= 2.0 * (combo(pions.mpi_V) - combo(pions.meta_V))
        # Axial pion and eta terms
        result -= 2.0 * (combo(pions.mpi_A) - combo(pions.meta_A))
        # Normalization
        result *= -3.0 * g2 * energy / (4.0 * np.pi * fpi)**2.0
        return result

    def _self_energy_B2K(self):
        """
        Computes the self-energy contribution for B to K semileptonic decays,
        which happens to vanish in SU(2) EFT. See Eq. (A33) in J. Bailey et al
        "B -> Kl+l- Decay Form Factors from Three-Flavor Lattice QCD"
        Phys.Rev. D93 (2016) no.2, 025026 [arXiv:1509.06235]
        """
        return 0.0
# pylint: enable=invalid-name
