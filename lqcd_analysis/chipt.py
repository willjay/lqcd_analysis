"""
Functions and classes for modeling semi-leptonic decays of B- and D-mesons
using chiral effective theory.
"""
from abc import abstractmethod
from collections import namedtuple
import re
import numpy as np
import gvar as gv
from . import analysis
from . import finite_volume


fv_shift = finite_volume.table_interpolator()


GoldstoneBosons = namedtuple(
    'GoldstoneBosons',
    ['pions', 'kaons', 'strangeons'],
    defaults=[None, None, None])


def get_value(dict_list, key, extend=False, default=None):
    """
    Gets a value from a list of dicts.
    """
    for adict in dict_list:
        value = adict.get(key)
        if value is not None:
            return value
        value = adict.get(f"log({key})")
        if value is not None:
            return np.exp(value)
    if extend:
        return default
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
        return bool(re.match(r"c_([lshHE]\d{0,})+", coefficient_name))
    return bool(re.match(r"c_([lshHEa]\d{0,})+", coefficient_name))


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
    matches = re.findall(r'[lshHEa]\d{0,}', coefficient_name)
    result = []
    for match in matches:
        power = 1
        if len(match) > 1:
            power = int(match[1:])
        result.append((match[0], power))
    return result


def analytic_terms(chi, params, continuum=False, order=None):
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
    re_log = re.compile(r'^log\((\S+)\)$')
    result = 0.
    alpha_s = chi['alpha_s']
    # Approximate ratio v/c for D mesons
    # See https://arxiv.org/pdf/hep-lat/0610092.pdf, especially discussion around Table III.
    v_by_c = np.sqrt(0.1)
    for name, value in params.items():
        # Handle log priors
        match = re_log.match(name)
        if match:
            name, = match.groups()
            value = np.exp(value)
        if valid_name(name, continuum):
            term = value
            if order is not None:
                if order == 'NLO':
                    # Restrict to these coefficients
                    if name not in ('c_l', 'c_s', 'c_h', 'c_H', 'c_E', 'c_E2'):
                        continue
                elif order == 'NNLO+':
                    # Take anything but these coefficients
                    if name in ('c_l', 'c_s', 'c_h', 'c_H', 'c_E', 'c_E2', 'c_a2'):
                        continue
                    print("NNLO+", name)
            for subscript, power in parse_name(name):
                term *= chi[subscript]**power  # Subscript is l, h, H, a, or E
                if subscript in ('h', ):
                    # Leading generic and heavy-quark discretization errors begin at
                    # order alpha_s (a Lamba)^2 and alpha_s (a m_h)**2, respectively.
                    # See discussion around Eqs (5.15 and 5.16) in
                    # [https://arxiv.org/pdf/1712.09262.pdf]
                    if power <= 2:
                        term *= (alpha_s * v_by_c)**int(power/2)
                    else:
                        term *= v_by_c
            result += term
    return result


def chiral_log_i1(mass, lam, mpiL=None):
    """
    Computes the chiral logarithm function I_1.
    See Eq. 46 of Aubin and Bernard
    "Heavy-light semileptonic decays in staggered chiral perturbation theory"
    Phys.Rev. D76 (2007) 014002 [arXiv:0704.0795]
    """
    lam2 = lam**2.
    mass2 = mass**2.
    if mpiL is None:
        return mass2 * np.log(mass2 / lam2)
    else:
        return mass2 * (np.log(mass2 / lam2) + 4 * fv_shift(mpiL))


def chiral_log_i2(mass, delta, lam):
    """
    Computes the chiral logarithm function I_2.
    See Eq. 47 of Aubin and Bernard,
    "Heavy-light semileptonic decays in staggered chiral perturbation theory"
    Phys.Rev. D76 (2007) 014002 [arXiv:0704.0795]
    """
    lam2 = lam**2.
    mass2 = mass**2.
    delta2 = delta**2.
    x = mass / delta
    return 2. * delta2 * (1. - np.log(mass2 / lam2) - 2. * chiral_log_f(x))


def chiral_log_j1(mass, delta, lam):
    """
    Computes the chiral logarithm function J_1.
    See Eq. 48 of Aubin and Bernard,
    "Heavy-light semileptonic decays in staggered chiral perturbation theory"
    Phys.Rev. D76 (2007) 014002 [arXiv:0704.0795]
    """
    mass2 = mass**2.
    delta2 = delta**2.
    lam2 = lam**2.
    x = mass / delta
    return (-mass2 + 2. / 3. * delta2) * np.log(mass2 / lam2)\
        + (4. / 3.) * (delta2 - mass2) * chiral_log_f(x)\
        - (10. / 9.) * delta2\
        + (4. / 3.) * mass2


def chiral_log_f(x):
    """
    Computes the function F(x) appearing alongside chiral logarithms.
    See Eq. 49 of Aubin and Bernard,
    "Heavy-light semileptonic decays in staggered chiral perturbation theory"
    Phys.Rev. D76 (2007) 014002 [arXiv:0704.0795]
    """
    if np.any(x) < 0:
        raise ValueError("chiral_log_f(x) needs x >= 0.")
    result = np.zeros(x.shape, dtype=object)
    # Region 1: x > 1
    # Note the usual inverse tangent
    region_1 = (x > 1)  # x > 1
    root = np.sqrt(x[region_1]**2. - 1)
    result[region_1] = -1. * root * np.arctan(root)
    # Region 2: x in [0, 1]
    # Since x is positive, region 2 is the complement of region 1.
    # Note: hyperbolic inverse tangent
    region_2 = ~region_1
    try:
        root = np.sqrt(1. - x[region_2]**2.0)
    except ZeroDivisionError:
        eps = 1e-6
        root = np.sqrt(1. + eps - x[region_2]**2.0)
    result[region_2] = root * np.arctanh(root)
    return result


def chiral_log_j1sub(mass, delta, lam):
    """
    Computes the subtracted chiral logarithm function J_1, which cancels the
    singularity when delta --> 0.
    See Eq. 51 of Aubin and Bernard,
    "Heavy-light semileptonic decays in staggered chiral perturbation theory"
    Phys.Rev. D76 (2007) 014002 [arXiv:0704.0795]
    """
    factor = 2. * np.pi / 3.
    mass3 = mass**3.
    return chiral_log_j1(mass, delta, lam) - factor * mass3 / delta


def residue_r(mass, mu, j):
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
    mu2 = np.array(mu)**2.
    mass2 = np.array(mass)**2.
    mass2hat = np.delete(mass2, idx_j)  # mass2 but without the jth element
    upper = np.product(mu2 - mass2[idx_j])
    lower = np.product(mass2hat - mass2[idx_j])
    if np.isclose(gv.mean(lower), 0., rtol=1e-6):
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
        + 1. * fcn(pions.m_i, *args, **kwargs)
        + 4. * fcn(pions.m_v, *args, **kwargs)
        + 6. * fcn(pions.m_t, *args, **kwargs)
        + 4. * fcn(pions.m_a, *args, **kwargs)
        + 1. * fcn(pions.m_p, *args, **kwargs)
    ) / 16.


def taste_average_i1(pions, lam, mpiL=None):
    """
    Computes the taste-averaged evaluation of I1.
    """
    return taste_average(chiral_log_i1, pions, lam, mpiL=mpiL)


def taste_average_i2(pions, delta, lam):
    """
    Computes the taste-averaged evaluation of I2.
    """
    return taste_average(chiral_log_i2, pions, delta, lam)


def taste_average_j1sub(pions, delta, lam):
    """
    Computes the taste-averaged evaluation of J1sub.
    """
    return taste_average(chiral_log_j1sub, pions, delta, lam)


# def form_factor_tree_level(phi, gpi, fpi, energy, delta, self_energy=0., form_factor_name='f_perp'):
def form_factor_tree_level(c_leading, energy, delta, self_energy=0., form_factor_name='f_perp'):
    """
    Computes the tree-level expression for the form factor.

    This expression appears (with inconsistent notation) throughout the
    literature. For example,

    Eqs. (3.20) and (3.21) of [https://arxiv.org/abs/1509.06235], J. Bailey et
    al., PRD 93, 025026 (2016) "B -> Kl+l- decay form factors from three-flavor
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
    # return gpi / (fpi * (energy + delta + self_energy))
    if form_factor_name in ('f_perp', 'f_parallel', 'f_0', 'f_plus'):
        # In physical terms, the leading constant c_leading = phi * g / fpi
        # return (phi * gpi) / (fpi * (energy + delta + self_energy))
        return c_leading / (energy + delta + self_energy)
        # return c_leading / (1 + energy / delta)
    # elif form_factor_name == 'f_parallel':
    #     return phi / fpi
    elif form_factor_name == 'f_T':
        raise NotImplementedError(f"{form_factor_name} not implemented yet")
    else:
        raise ValueError(f"Unrecognized form_factor_name: {form_factor_name}")


def form_factor_tree_level_pole(c_leading, q2, Mpole2):
    """
    The exact form of the pole is 1/(Mpole**2 - q2).
    Here Mpole is the mass of the vector (1-) or scalar (0+).
    """
    return c_leading / (1 - q2/Mpole2)


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
    def __init__(self, x, params, base='mpi5', continuum=False):
        if base not in ['mpi5', 'mK5', 'mS5']:
            raise ValueError(
                f"Invalid base: {base}. Please specify 'mpi5', 'mK5', or mS5.")
        self.base = base
        self.dict_list = [x, params]
        self.continuum = continuum
        self.m_i = self._mpi(taste='I')
        self.m_p = self._mpi(taste='P')
        self.m_v = self._mpi(taste='V')
        self.m_a = self._mpi(taste='A')
        self.m_t = self._mpi(taste='T')
        self.meta_v = self._meta(taste='V')
        self.meta_a = self._meta(taste='A')
        if base == 'mS5':
            self.metaprime_v = self._metaprime(taste='V')
            self.metaprime_a = self._metaprime(taste='A')

    def _mpi(self, taste):
        """
        Computes the "pion" mass in a different taste, given the mass of the
        "pion" with pseuoscalar taste:
        m^2_{ij,xi} = mu (m_i + m_j) + a^2 Delta_xi
        where
        mu (m_i + m_j) = "pion with pseudoscalar taste" = m^2_{ij,5}.

        Eq. (A8) of [https://arxiv.org/abs/1901.02561], A. Bazavov et al.,
        PRD 100 (2019) no.3, 034501 "Bs -> Klnu decay from lattice QCD."
        """
        m_ij5 = self.__getitem__(self.base)
        if self.continuum:
            return m_ij5
        delta = self.__getitem__(f'Delta_{taste}')
        arg = m_ij5**2. + delta
        if arg < 0:
            raise ValueError(
                f"Negative argument encountered using 'Delta_{taste}'."
            )
        return np.sqrt(arg)

    def _metaprime(self, taste):
        """
        Compute the etaprime mass in a different taste, given the mass of the
        "pion" with pseudoscalar taste.
        """
        if self.base in ('mpi5', 'mK5'):
            raise ValueError(
                "SU(2) theory does not have an etaprime state. "
                "Please use base='mS5'."
            )
        return self._meta(taste, factor=0.25)

    def _meta(self, taste, factor=0.5):
        """
        Computes the eta mass in a different taste, given the mass of the
        "pion" with pseudoscalar taste.
        m^2_{eta,V(A)} = m^2_{uu,V(A)} + 1/2 a^2 delta^prime_{V(A)},
        where
        "a^2 delta^prime_{V(A)}" = "Hairpin_V(A)".
        Eq. (A5b) of [https://arxiv.org/abs/1901.02561], A. Bazavov et al.,
        PRD 100 (2019) no.3, 034501 "Bs -> Klnu decay from lattice QCD."
        """
        m_ij_xi = self._mpi(taste)
        if self.continuum:
            return m_ij_xi
        hairpin = self.__getitem__(f'Hairpin_{taste}')
        arg = m_ij_xi**2. + factor * hairpin
        if arg < 0:
            raise ValueError(
                f"Negative argument encountered using 'Hairpin_{taste}'."
            )
        return np.sqrt(arg)

    def __getitem__(self, key):
        return get_value(self.dict_list, key)

    def __str__(self):
        return (
            "StaggeredPions(\n"
            "    m_P  = {m_P},\n"
            "    m_I  = {m_I},\n"
            "    m_V  = {m_V},\n"
            "    m_A  = {m_A},\n"
            "    m_T  = {m_T},\n"
            "    meta_V = {meta_V},\n"
            "    meta_A = {meta_A})".format(**self.__dict__)
        )


class ChiralExpansionParameters:
    """
    Wrapper class for the dimensionless chiral expansion parameters "chi."
    Args:
        x, params: dicts with keys 'm_light', 'm_heavy', 'E', 'mu', 'fpi', and 'DeltaBar'
    """

    def __init__(self, x, params):
        self.x = x
        self.params = params
        self._validate()

    def _validate(self):
        # keys = ['m_light', 'm_strange', 'm_heavy', 'dm_heavy', 'E', 'mu', 'fpi', 'DeltaBar', 'alpha_s']
        # keys = ['m_light', 'm_strange', 'm_heavy', 'dMH2', 'E', 'mu', 'fpi', 'DeltaBar', 'alpha_s']
        keys = ['mpi5', 'dMH2', 'E', 'fpi', 'DeltaBar', 'alpha_s']
        for key in keys:
            test = (int(key in self.x) + int(key in self.params))
            if test == 1:
                continue
            if test == 0:
                raise KeyError(f"Missing key {key}")
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
        mpi = self._get('mpi5')
        # mu = self._get('mu')
        fpi = self._get('fpi')
        # m_light = self._get('m_light')
        # return (2. * mu * m_light) / (8. * np.pi**2. * fpi**2.)
        return mpi**2 / (8. * np.pi**2. * fpi**2.)

    @property
    def strange(self):
        """
        Computes a dimensionless expansion parameter for use with the strange quark.
        This term is similar to those of defined in Eq (3.22) or (3.23) of
        [https://arxiv.org/abs/1509.06235], J. Bailey et al., PRD 93, 025026
        (2016) "B -> Kl+l− decay form factors from three-flavor lattice QCD."
        """
        mu = self._get('mu')
        fpi = self._get('fpi')
        m_strange = self._get('m_strange')
        return 2. * mu * m_strange / (8. * np.pi**2. * fpi**2.)

    @property
    def dstrange(self):
        """
        Computes a dimensionless "mistuning" paramter for the kaon.
        """
        fpi = self._get('fpi')
        # dm_heavy = self._get('dm_heavy')
        # return 2. * mu * dm_heavy / (8. * np.pi**2. * fpi**2.)
        dMK2 = self._get('dMK2')
        return dMK2 / (8. * np.pi**2. * fpi**2.)

    @property
    def heavy(self):
        """
        Computes the expansion parameter "x_h" for HQET discretization errors
        defined in the text preceding
        Eq. (5.14) of
        [https://arxiv.org/abs/1712.09262] A. Bazavov et al., PRD 98 (2018) 7, 074512
        "B- and D-meson leptonic decay constants from four-flavor lattice QCD".
        """
        m_heavy = self._get('m_heavy')
        return 2. * m_heavy / np.pi

    @property
    def dheavy(self):
        """
        Computes a dimenionless expansion parameter in the mistuning of the
        heavy quark mass, similar to the one defined in Eq (3.23) of
        [https://arxiv.org/abs/1509.06235], J. Bailey et al., PRD 93, 025026
        (2016) "B -> Kl+l− decay form factors from three-flavor lattice QCD."
        """
        # mu = self._get('mu')
        fpi = self._get('fpi')
        # dm_heavy = self._get('dm_heavy')
        # return 2. * mu * dm_heavy / (8. * np.pi**2. * fpi**2.)
        dMH2 = self._get('dMH2')
        return dMH2 / (8. * np.pi**2. * fpi**2.)

    @property
    def energy(self):
        """
        Computes the dimenionless expansion parameter defined in Eq (3.24) of
        [https://arxiv.org/abs/1509.06235], J. Bailey et al., PRD 93, 025026
        (2016) "B -> Kl+l− decay form factors from three-flavor lattice QCD."
        """
        fpi = self._get('fpi')
        energy = self._get('E')
        return np.sqrt(2.) * energy / (4. * np.pi * fpi)

    @property
    def a2(self):
        """
        Computes the dimenionless expansion parameter defined in Eq (3.24) of
        [https://arxiv.org/abs/1509.06235], J. Bailey et al., PRD 93, 025026
        (2016) "B -> Kl+l− decay form factors from three-flavor lattice QCD."
        """
        fpi = self._get('fpi')
        delta_bar = self._get('DeltaBar')
        return delta_bar / (8. * np.pi**2. * fpi**2.)

    def __getitem__(self, key):
        if key in ('l', 'light'):
            return self.light
        if key in ('s', 'strange'):
            return self.dstrange
        if key in ('h', 'heavy'):
            return self.heavy
        if key in ('H', 'dheavy'):
            return self.dheavy
        if key in ('E', 'energy'):
            return self.energy
        if key == 'a2':
            return self.a2
        if key == 'a':
            return np.sqrt(self.a2)
        if key == 'alpha_s':
            return self._get('alpha_s')
        raise KeyError(f"Invalid key: {key}")

    def __str__(self):
        return (
            f"ChiralExpansionParamters(\n"
            f"    chi_light  = {self.light}\n"
            f"    chi_strange = {self.dstrange}\n"
            f"    chi_heavy  = {self.heavy}\n"
            f"    chi_dheavy = {self.dheavy}\n"
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
        self.scale_factor = value**(-1. / dim)

    def __str__(self):
        return (
            f"Scale(name='{self.name}', "
            f"value={self.value}, "
            f"dim={self.dim}, "
            f"scale_factor={self.scale_factor})"
        )


class ChiralModel:
    """
    General chiral model for semileptonic decays of B and D mesons.
    """
    def __init__(self, form_factor_name, process, lam, continuum=False):
        valid_names = [
            'f_parallel', r'f_\parallel', 'f_perp', r'f_\perp', 'f_0', 'f_T',
            'f_plus',
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

    @abstractmethod
    def model(self, *args):
        """Computes the model function."""
        raise NotImplementedError()

    @abstractmethod
    def delta_logs(self, *args, **kwargs):
        """Computes the chiral logarithms."""
        return NotImplementedError()

    @abstractmethod
    def self_energy(self, *args, **kwargs):
        """Computes the self-energy correction."""
        return NotImplementedError()

    def __call__(self, *args):
        return self.model(*args)

    def __str__(self):
        return (
            f"{self.model_type}({self.process}; "
            f"{self.form_factor_name}; "
            f"Lambda_UV={np.round(self.lam, decimals=4)}; "
            f"continuum={self.continuum})"
        )


def convert_to_w0_units(xdict):
    """
    Converts input data in physical units to dimensionless units of w0.
    Assumes that all input data are in consistent units (e.g., MeV) and that
    w0 is given in the same units (e.g., 1/MeV).
    This function is useful for constructing final error budgets, where it's
    important to maintain separation between errors coming from the scale w0
    and the other input parameters (like the quark masses).
    """
    w0 = xdict['w0']
    dim0 = ['alpha_s', 'units', 'Mpi*L']
    dim1 = ['fpi', 'm_light', 'm_strange', 'm_heavy', 'dm_heavy',
            'mpi5', 'mK5', 'mS5', 'mu', 'M_mother', 'M_daughter', 'E']
    dim2 = ['dMH2', 'DeltaBar', 'dMK2', 'q2']
    result = {}
    for key, value in xdict.items():
        if (key in dim0) or (key == 'w0'):
            result[key] = value
        elif (key in dim1):
            result[key] = value * w0
        elif (key in dim2):
            result[key] = value * w0**2
        else:
            raise ValueError("Unexpected key", key)
    return result


class LogLessModel(ChiralModel):
    """
    Base model for log-less description of form factors.

    Sometimes fits to functions from proper effective theory have the feature that contributions from chiral
    logarithms are numerically small. This observation motivates one to consider a model which simply neglects
    any non-analytic functions, supposing that the full behavior can be captured by analytic terms. This treatment
    is probably only reasonable for chiral interpolations (and NOT extrapolations). For interpolation, one might be
    able to expand:
    m = m0 + dm
    m*log(m)
        = (m0+dm)*log(m0+dm)
        = m0*log(m0) + (1+log(m0))*dm + dm^2 / (2m0) + ...
        = c0 + c1*dm + c2*dm^2 + ...
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
        if ('units' in x) and (x['units'] == 'MeV'):
            print("Converting input data to w0 units.")
            _x = convert_to_w0_units(x)
        else:
            _x = x
        dict_list = [_x, params]

        # Extract values from inputs
        leading = get_value(dict_list, 'leading')
        energy = get_value(dict_list, 'E')
        # q2 = get_value(dict_list, 'q2')
        delta = get_value(dict_list, 'delta_pole')
        # pole = get_value(dict_list, 'pole')
        sigma = 0  # Take the self energy to vanish

        # Get the analytic terms
        chi = ChiralExpansionParameters(x, params)
        analytic = analytic_terms(chi, params, self.continuum)

        # Leading-order x (corrections )
        name = self.form_factor_name
        tree = form_factor_tree_level(leading, energy, delta, sigma, name)
        # tree = form_factor_tree_level_pole(leading, q2, pole)
        return tree * (1 + analytic)

    def breakdown(self, *args):
        # Unpackage x, params
        if len(args) not in (1, 2):
            raise TypeError(
                "Please specify either (x, params) or params as arguments."
            )
        x, params = args if len(args) == 2 else ({}, args[0])
        dict_list = [x, params]

        # Extract values from inputs
        leading = get_value(dict_list, 'leading')
        energy = get_value(dict_list, 'E')
        # q2 = get_value(dict_list, 'q2')
        delta = get_value(dict_list, 'delta_pole')
        # pole = get_value(dict_list, 'pole')
        sigma = 0  # Take the self energy to vanish

        # Get the analytic terms
        chi = ChiralExpansionParameters(x, params)
        analytic = analytic_terms(chi, params, self.continuum)
        analytic_nlo = analytic_terms(chi, params, self.continuum, order='NLO')
        analytic_nnlo = analytic_terms(chi, params, self.continuum, order='NNLO+')

        # Leading-order x (corrections )
        name = self.form_factor_name
        tree = form_factor_tree_level(leading, energy, delta, sigma, name)
        # tree = form_factor_tree_level_pole(leading, q2, pole)
        # return tree * (1 + analytic)
        return {
            'tree': tree,
            'NLO_log_corrections': 0,
            'NLO_analytic_corrections': tree * analytic_nlo,
            'full_analtyic_corrections': tree * analytic,
            'LO': tree,
            'NLO': tree * analytic_nlo,
            'NNLO': tree * analytic_nnlo,
            'self_energy': 0,
        }