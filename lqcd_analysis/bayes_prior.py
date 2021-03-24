"""
BasePrior
 * MesonPrior
 * FormFactorPrior
 * GoldstonePrior
"""
import numpy as np
import gvar as gv
import re

def main():
    """TODO: Add main function."""


def inflate(params, frac):
    """
    Inflates the width on the priors to frac*mean, unless the existing width is
    already wider.
    """
    for key, value in params.items():
        mean = gv.mean(value)
        sdev = np.maximum(frac*np.abs(mean), gv.sdev(value))
        params[key] = gv.gvar(mean, sdev)
    return params


def _is_log(key):
    """Check if the key has the form 'log(*)'."""
# TODO: delete
#     if key.startswith('log(') and key.endswith(')'):
#         return True
    pattern = re.compile(r"^log\((.*)\)$")
    match = re.match(pattern, key)
    if match:
        if re.match(pattern, match[1]):
            raise ValueError(f"Cannot have 'log(log(*))' keys, found {key}")
        return True
    return False


def _check_duplicate_keys(keys):
    """Avoid key duplication during initialization."""
    log_keys = [key for key in keys if _is_log(key)]
    for log_key in log_keys:
        # log(<astring>) --> <astring>
        exp_key = re.match(r"^log\((.*)\)$", log_key)[1]
# TODO: delete
#         exp_key = log_key[4:-1]
        if exp_key in keys:
            pass
            # msg = "Cannot have keys '{0}' and '{1}' together.".\
            #     format(log_key, exp_key)
            # msg += " Pick one or the other."
            # print(msg)
            # raise ValueError(msg)


def _sanitize_mapping(mapping):
    """Replace log keys/values with 'exp' versions in the internal dict."""
    log_keys = [key for key in mapping.keys() if _is_log(key)]
    for log_key in log_keys:
        exp_value = np.exp(mapping.pop(log_key))
        exp_key = log_key[4:-1]
        mapping[exp_key] = exp_value
    return mapping


class BasePrior(object):
    """
    Basic class for priors, which basically extends the functionality
    of dict to account for possible "log priors" with keys like "log(key)"
    without having to compute the logarithms by hand.
    Args:
        mapping: dict, the prior's key-value pairs
        extend: bool, whether or not to treat handle energies
            'dE' as 'log' priors
    Returns:
        BasePrior object
    """

    def __init__(self, mapping, extend=True):
        _check_duplicate_keys(mapping.keys())
        self.extend = extend
        self.dict = dict(_sanitize_mapping(mapping))
        self._verify_keys()

    def _verify_keys(self):
        """Check for spurious log keys, e.g, from 'log(log(key))'."""
        if self.extend:
            for key in self.dict.keys():
                if _is_log(key):
                    msg = "Invalid key encountered {0}.".format(key)
                    msg += " Perhaps from log({0})?".format(key)
                    raise ValueError(msg)

    def __getitem__(self, key):
        """Get value corresponding to key, allowing for 'log' terms."""
        if self.extend and _is_log(key):
            return np.log(self.dict.__getitem__(key[4:-1]))

        return self.dict.__getitem__(key)

    def __setitem__(self, key, value):
        """Set value corresponding to key, allowing for 'log' terms."""
        if self.extend and _is_log(key):
            self.dict.__setitem__(key[4:-1], np.exp(value))
        else:
            self.dict.__setitem__(key, value)

    def __len__(self):
        return self.dict.__len__()

    def __iter__(self):
        for key in self.keys():
            yield key

    def __repr__(self):
        string = '{'
        str_tmp = []
        for key, val in self.items():
            str_tmp.append("{0} : {1},".format(key.__repr__(), val.__repr__()))
        string += '\n'.join(str_tmp)
        string += '}'
        return string

    def __str__(self):
        string = '{'
        str_tmp = []
        for key, val in self.items():
            str_tmp.append("{0} : {1},".format(key.__str__(), val.__str__()))
        string += '\n'.join(str_tmp)
        string += '}'
        return string

    def items(self):
        """Overrides items to handle 'logs' gracefully."""
        for key in self.keys():
            yield key, self.__getitem__(key)

    def _keys(self):
        for key in self.dict.keys():
            # nota bene: this enforces that these keys use log priors!
            # if ('dE' in key) or ('fluctuation' in key):
            if ('dE' in key) or (':a' in key) or ('fluctuation' in key):
                yield 'log({0})'.format(key)
            else:
                yield key

    def keys(self):
        """Override keys to handle 'logs' gracefully."""
        if self.extend:
            return self._keys()
        return self.dict.keys()

    def values(self):
        """Override values to handle 'logs' gracefully."""
        for key in self.keys():
            yield self.__getitem__(key)

    def update(self, update_with, width=None, fractional_width=False):
        """Update keys in prior with dict 'update_with'."""
        keys = self.keys()
        for key in update_with:
            if key in keys:
                value = update_with[key]
                if width:
                    if not hasattr(value, '__len__'):
                        value = [value]
                    if fractional_width:
                        value = [gv.gvar(gv.mean(val), gv.mean(val) * width) for val in value]
                    else:
                        value = [gv.gvar(gv.mean(val), width) for val in value]
                self.__setitem__(key, value)

    @property
    def p0(self):
        """Get central values for initial guesses"""
        return {key: gv.mean(val) for key, val in self.items()}


class MesonPrior(BasePrior):
    """
    Prior for mesonic two-point correlation function.
    MesonPrior is basically an extension of dict
    which handles log priors gracefully.
    Args:
        n: int, the number of decaying states
        no: int, the number of oscillating states
        amps: list of strings specifying the source and
            sink amplitudes. Default is ['a','b','ao','bo'].
        tag: str giving a tag/name, often something like
            'light-light' or 'heavy-light' This can be useful
            to distinguish different correlators in a joint fit.
        ffit: corrfitter.fastfit object for estimating masses and
            amplitudes. Default is None.
    """

    def __init__(self, n=1, no=0, amps=None, tag=None, ffit=None, **kwargs):
        if n < 1:
            raise ValueError("Must have n_decay >=1.")
        if no < 0:
            raise ValueError("Must have n_oscillate > 0.")
        if amps is None:
            amps = ['a', 'b', 'ao', 'bo']
        super(MesonPrior, self).\
            __init__(MesonPrior._build(n, no, amps, tag, ffit), **kwargs)

    @staticmethod
    def _build(n_decay, n_oscillate, amps, tag=None, ffit=None):
        """Build the prior dict."""
        prior = {}
        # Decaying energies and amplitudes
        n = range(n_decay)
        prior['dE'] = [gv.gvar('1.0(1.0)')] +\
            [gv.gvar('0.6(0.6)') for _ in range(1, n_decay)]
        if 'a' in amps:
            prior['a'] = [gv.gvar('0.1(1.0)') for _ in n]
        if 'b' in amps:
            prior['b'] = [gv.gvar('0.1(1.0)') for _ in n]

        # Oscillating eneriges and amplitudes
        if n_oscillate > 0:
            no = range(0, n_oscillate)
            prior['dEo'] = [gv.gvar('1.65(50)')] +\
                           [gv.gvar('0.6(0.6)') for _ in range(1, n_oscillate)]
            if 'ao' in amps:
                prior['ao'] = [gv.gvar('0.1(1.0)') for _ in no]
            if 'bo' in amps:
                prior['bo'] = [gv.gvar('0.1(1.0)') for _ in no]

        # Extract guesses for the ground-state energy and amplitude
        if ffit is not None:
            dE_guess = gv.mean(ffit.E)
            amp_guess = gv.mean(ffit.ampl)
            prior['dE'][0] = gv.gvar(dE_guess, 0.5 * dE_guess)
            if 'a' in amps:
                prior['a'][0] = gv.gvar(amp_guess, 2.0 * amp_guess)
            elif 'b' in amps:
                prior['b'][0] = gv.gvar(amp_guess, 2.0 * amp_guess)
            else:
                msg = "Error: Unrecognized amplitude structure?"
                raise ValueError(msg)

        # Convert to arrays
        keys = list(prior.keys())
        if tag is None:
            # Just convert to arrays
            for key in keys:
                prior[key] = np.asarray(prior[key])
        else:
            # Prepend keys with 'tag:' and then convert
            for key in keys:
                new_key = "{0}:{1}".format(tag, key)
                prior[new_key] = np.asarray(prior.pop(key))

        return prior


class FormFactorPrior(BasePrior):
    """
    Prior for joint fits to extract form factors.
    FormFactorPrior is basically an extension of dict
    which handles log priors gracefully.
    Args:
        nstates: dict specifiying the number of states to use in
            the prior and which should have a form like, e.g.,
            {
                'light-light' : (n, no),
                'heavy-light' : (m, mo),
            }
            where n, no, m, and mo are all integers.
        ds: FormFactorDataset, for estimating masses, amplitudes,
            and the matrix element which gives the form factor.
            Default is None.
    Returns:
        FormFactorPrior object
    """

    def __init__(self, nstates, ds=None, positive_ff=True, pedestal=None, **kwargs):
        if ds is None:
            ds = {}
        else:
            FormFactorPrior._verify_tags(ds.tags)
        self.positive_ff = positive_ff
        self.pedestal = pedestal
        super(FormFactorPrior, self).__init__(
                mapping=self._build(nstates, ds),
                **kwargs)

    @staticmethod
    def _verify_tags(tags):
        """Verify that the tags (from nstates) are supported."""
        tags = set(tags)
        valid_tags = [
            ['light-light'],
            ['heavy-light'],
            ['heavy-heavy'],
            ['light-light', 'heavy-light'],
        ]
        for possibility in valid_tags:
            if tags == set(possibility):
                return
        raise ValueError("Unrecognized tags in FormFactorPrior")

    def _build(self, nstates, ds):
        """Build the prior dict."""
        prior = FormFactorPrior._make_meson_prior(nstates, ds)
        tmp = self._make_vmatrix_prior(nstates, ds)
        for key in tmp:
            prior[key] = tmp[key]
        return prior

    @staticmethod
    def _make_meson_prior(nstates, ds):
        """Build prior associated with the meson two-point functions."""
        tags = ds.tags
        meson_priors = [
            MesonPrior(nstates.n, nstates.no,
                       tag=tags.src, ffit=ds.c2_src.fastfit),
            MesonPrior(nstates.m, nstates.mo,
                       tag=tags.snk, ffit=ds.c2_snk.fastfit),
        ]
        prior = {}
        for meson_prior in meson_priors:
            for key, value in meson_prior.items():
                prior[key] = value
        return prior

    def _make_vmatrix_prior(self, nstates, ds):
        """Build prior for the 'mixing matrices' Vnn, Vno, Von, and Voo."""
        def _abs(val):
            return val * np.sign(val)
        mass = None
        n = nstates.n
        no = nstates.no
        m = nstates.m
        mo = nstates.mo
        mass = ds[ds.tags.src].fastfit.E

        # General guesses
        tmp_prior = {}
        tmp_prior['Vnn'] = gv.gvar(n * [m * ['0.1(10.0)']])
        tmp_prior['Vno'] = gv.gvar(n * [mo * ['0.1(10.0)']])
        tmp_prior['Von'] = gv.gvar(no * [m * ['0.1(10.0)']])
        tmp_prior['Voo'] = gv.gvar(no * [mo * ['0.1(10.0)']])
        # v = gv.mean(ds.v_guess)
        # verr = 0.5 * _abs(v)
        # tmp_prior['Vnn'][0, 0] = gv.gvar(v, verr)

        if self.pedestal is not None:
            raise ValueError("Pedestal is untrustworthy -- don't use.")
            if np.sign(self.pedestal) != np.sign(v):
                raise ValueError(
                    "Sign of the specified pedestal disagrees with the sign"
                    " inferred from the data. Unable to proceed.")
            sign = np.sign(v)
            v = self.pedestal
            verr = 0.5 * _abs(v)
            # Start with a 10% fluctuation on top of the pedestal
            fluctuation = 0.1 * _abs(v)
            # Start the matrix element at the pedestal with large uncertainty
            tmp_prior['Vnn'][0, 0] = gv.gvar(v + sign * fluctuation, verr)
            # In the model function, the pedestal is a pure number without error
            # so the fluctuation should carry all the uncertainty.
            tmp_prior['log(fluctuation)'] = np.log(gv.gvar(fluctuation, verr))

        return tmp_prior


def vmatrix(nstates):
    """
    Get the prior for matrices of matrix elements Vnn, Vno, Von, and Voo in
    lattice units.
    """
    n = nstates.n
    no = nstates.no
    m = nstates.m
    mo = nstates.mo
    # General guesses
    prior = {}
    prior['Vnn'] = gv.gvar(n * [m * ['0.1(10.0)']])
    prior['Vno'] = gv.gvar(n * [mo * ['0.1(10.0)']])
    prior['Von'] = gv.gvar(no * [m * ['0.1(10.0)']])
    prior['Voo'] = gv.gvar(no * [mo * ['0.1(10.0)']])
    return prior


def pion_energy(n):
    """
    Get the energy of the nth excited pion in MeV.
    """
    if n == 0:
        return 0.
    if n == 1:
        return gv.gvar(135, 50)
    if n == 2:
        return (gv.gvar(1300, 400))
    return gv.gvar(1300 + 400*(n-2), 400)


def pion_osc_energy(n):
    """
    Get the energy of the nth excited opposite-parity pion in MeV.
    """
    if n == 0:
        return 0.
    if n == 1:
        return gv.gvar(500, 300)
    return gv.gvar(500 + 400*(n-1), 800)


def d_energy(n):
    """ Get the energy of the nth excited D meson in MeV. """
    if n == 0:
        return 0.
    if n == 1:
        return gv.gvar(1865, 200)
    return gv.gvar(1865 + 700*(n-1), 700)


def d_osc_energy(n):
    """ Get the energy of the nth excted opposite-parity D meson in MeV. """
    if n == 0:
        return 0.
    if n == 1:
        return gv.gvar(2300, 700)
    return gv.gvar(2300 + 700*(n-1), 700)


def b_energy(n):
    """ Get the energy of the nth excited B meson in MeV. """
    if n == 0:
        return 0.
    if n == 1:
        # Use the measured value from the PDG: M_B = 5280 MeV
        return gv.gvar(5280, 200)
    return gv.gvar(5280 + 1000*(n-1), 1000)


def b_osc_energy(n):
    """ Get the energy of the nth excted opposite-parity D meson in MeV. """
    if n == 0:
        return 0.
    if n == 1:
        # Use a guess for the 1/2(0+) state, which has not yet been identified
        # experimentally. I expect this state to be somewhat heavier than the
        # B0, albeit with a large uncertainty on the precise value.
        return gv.gvar(5600, 1000)
    return gv.gvar(5600 + 1000*(n-1), 1000)

def boost(dE, p2):
    """
    Boost the energy splittings with the squared momentum using the
    relativistic dispersion relation. First sums the splittings dE to get
    masses / energies. Then boosts and converts back to splittings.
    Assumes dE and p2 are in the same units, e.g., lattice units.
    Does *not* do any unit conversion.
    """
    e2 = np.cumsum(dE)**2.0
    boosted = np.sqrt(e2 + p2)
    # "inverse" of cumsum
    dE_boosted = np.ediff1d(boosted, to_begin=boosted[0])
    # maintain the fractional uncertainty
    tmp = []
    for idx, (dEi, dEi_boosted) in enumerate(zip(dE, dE_boosted)):
        if idx == 0:
            frac = 0.75
        else:
            frac = gv.sdev(dEi)/gv.mean(dEi)
        mean = gv.mean(dEi_boosted)
        err  = frac * mean
        tmp.append(gv.gvar(mean, err))
    dE_boosted = np.array(tmp)
    return dE_boosted


class PhysicalSplittings():
    """
    Class for handling splittings inspired by physical results in the PDG.
    Both lattice units and MeV are supported.
    """
    def __init__(self, state):
        state = str(state).lower()
        if state not in ['pion', 'pion_osc', 'd', 'd_osc', 'b', 'b_osc']:
            raise ValueError(f"Unrecognized state. Found state={state}")
        self.state = state

    def energy(self, n):
        if self.state == 'pion':
            return pion_energy(n)
        elif self.state == 'pion_osc':
            return pion_osc_energy(n)
        elif self.state == 'd':
            return d_energy(n)
        elif self.state == 'd_osc':
            return d_osc_energy(n)
        elif self.state == 'b':
            return b_energy(n)
        elif self.state == 'b_osc':
            return b_osc_energy(n)
        else:
            raise ValueError("Unrecognized state")

    def __call__(self, n, a_fm=None, scale=1.0):
        """
        Get the energy splittings for n states in MeV (when a_fm is None) or
        lattice units (when a_fm is specified).
        """
        # Get energies in MeV
        energies = np.array([self.energy(ni) for ni in range(n+1)])
        # Apply scaling factor
        energies = energies * scale
        # Convert to energy splittings
        dE = energies[1:] - energies[:-1]
        if a_fm is None:
            return dE
        # Convert MeV to lattice units
        dE = dE * a_fm / 197
        return dE


def decay_amplitudes(n, a0="0.50(20)", ai="0.1(0.3)"):
    """
    Get basic amplitude guesses in lattice units for n total decaying states.
    """
    def _amplitude(n):
        if n == 0:
            return gv.gvar(a0)
        else:
            return gv.gvar(ai)
    return np.array([_amplitude(ni) for ni in range(n)])


def osc_amplitudes(n, ai="0.1(1.0)"):
    """
    Get basic amplitude guesses in lattice units for n total oscillating states.
    """
    return np.array([gv.gvar(ai) for _ in range(n)])


class FormFactorPriorD2Pi(BasePrior):
    """
    Class for building priors for D to pi form factor analyses inspired by
    physical results in the PDG.
    """
    def __init__(self, nstates, ds=None, a_fm=None, **kwargs):
        prior = {}
        # Decaying states
        prior['light-light:dE'] = PhysicalSplittings('pion')(nstates.n, a_fm)
        prior['light-light:a'] = decay_amplitudes(nstates.n)
        prior['heavy-light:dE'] = PhysicalSplittings('d')(nstates.m, a_fm)
        prior['heavy-light:a'] = decay_amplitudes(nstates.m)
        # Oscillating states
        if nstates.no:
            prior['light-light:dEo'] = PhysicalSplittings('pion_osc')(nstates.no, a_fm)
            prior['light-light:ao'] = osc_amplitudes(nstates.no)
        if nstates.mo:
            prior['heavy-light:dEo'] = PhysicalSplittings('d_osc')(nstates.mo, a_fm)
            prior['heavy-light:ao'] = osc_amplitudes(nstates.mo)

        # Matrix elements Vnn
        for key, value in vmatrix(nstates).items():
            if value.size:  # skip empty matrices
                prior[key] = value

        # Make informed guesses for ground states and the form factor Vnn[0,0].
        # Estimate central values as well as possible, but keep wide priors.
        if ds is not None:
            for tag in ['light-light', 'heavy-light']:
                mean = gv.mean(ds[tag].mass)  # Central value from "meff"
                err = gv.sdev(prior[f"{tag}:dE"][0])
                prior[f"{tag}:dE"][0] = gv.gvar(mean, err)
            mean = gv.mean(ds.v_guess)  # Central value from ratio R
            err = 0.5 * mean
            prior['Vnn'][0,0] = gv.gvar(mean, err)
        super(FormFactorPriorD2Pi, self).__init__(mapping=prior, **kwargs)

# NOTE for time being I am 'overriding' the heavy-light and light-light
# tags relevant for D to pi for use in B to D. Eventually will want to
# correct this.
class FormFactorPriorH2D(BasePrior):
    """
    Class for building priors for B to D form factor analyses.
    """
    def __init__(self, nstates, amh, ds=None, a_fm=None, **kwargs):
        prior = {}
        amp = "0.1(0.4)"  # Generic amplitude prior.
        hbarc = 0.197327  # GeV-fm.
        mc = 0.99  # MSbar at 3 GeV.
        mb = 4.2 # MSbar at mb.
        sf = (amh*hbarc/(a_fm*mb))**0.7  # Scale factor (see pdg.py).
        print('sf:', sf)
        
        # Decaying states
        prior['light-light:dE'] = PhysicalSplittings('d')(nstates.n, a_fm)
        prior['light-light:a'] = decay_amplitudes(nstates.n, amp, amp)
        prior['heavy-light:dE'] = PhysicalSplittings('b')(nstates.m, a_fm,
                                                           scale=sf)       
        prior['heavy-light:a'] = decay_amplitudes(nstates.m, amp, amp)
        # Oscillating states
        if nstates.no:
            prior['light-light:dEo'] = PhysicalSplittings('d_osc')(nstates.no, a_fm)
            prior['light-light:ao'] = osc_amplitudes(nstates.no, amp)
        if nstates.mo:
            prior['heavy-light:dEo'] = PhysicalSplittings('b_osc')(nstates.mo, 
                                                 a_fm, scale=sf)
            prior['heavy-light:ao'] = osc_amplitudes(nstates.mo, amp)

        # Matrix elements Vnn
        for key, value in vmatrix(nstates).items():
            if value.size:  # skip empty matrices
                prior[key] = value

        # Make informed guesses for ground states and the form factor Vnn[0,0].
        # Estimate central values as well as possible, but keep wide priors.
        if ds is not None:
            for tag in ['light-light', 'heavy-light']:
                mean = gv.mean(ds[tag].mass)  # Central value from "meff"
                err = gv.sdev(prior[f"{tag}:dE"][0])
                prior[f"{tag}:dE"][0] = gv.gvar(mean, err)
            mean = gv.mean(ds.v_guess)  # Central value from ratio R
            err = 0.5 * mean
            prior['Vnn'][0,0] = gv.gvar(mean, err)
        super(FormFactorPriorH2D, self).__init__(mapping=prior, **kwargs)

class FormFactorPriorD2D(BasePrior):
    """
    Class for building priors for generic form factor analyses inspired by
    physical results in the PDG.
    """
    def __init__(self, nstates, ds=None, a_fm=None, heavy_factor=1.0, **kwargs):
        prior = {}
        # Decaying states
        amp = "0.1(0.4)"  # Generic amplitude prior.
        prior['light-light:dE'] = PhysicalSplittings('d')(nstates.n, a_fm)
        prior['light-light:a'] = decay_amplitudes(nstates.n, amp, amp)
        prior['heavy-light:dE'] = PhysicalSplittings('d')(nstates.m, a_fm)
        prior['heavy-light:a'] = decay_amplitudes(nstates.m, amp, amp)
        
        # Oscillating states
        if nstates.no:
            prior['light-light:dEo'] = PhysicalSplittings('d_osc')(nstates.no, a_fm)
            prior['light-light:ao'] = osc_amplitudes(nstates.no, amp)
        if nstates.mo:
            prior['heavy-light:dEo'] = PhysicalSplittings('d_osc')(nstates.mo, a_fm)
            prior['heavy-light:ao'] = osc_amplitudes(nstates.mo, amp)
        # Scale up the ground-state mass of the "heavy D-meson".
        # Usually the "heavy D-meson" contains a heavier-than-physical "charm"
        # quark with a mass like "1.4 m_charm".
        prior['heavy-light:dE'][0] = heavy_factor * prior['heavy-light:dE'][0]
        prior['heavy-light:dEo'][0] = heavy_factor * prior['heavy-light:dEo'][0]

        # Matrix elements Vnn
        for key, value in vmatrix(nstates).items():
            if value.size:  # skip empty matrices
                prior[key] = value

        # Make informed guesses for ground states and the form factor Vnn[0,0].
        # Estimate central values as well as possible, but keep wide priors.
        if ds is not None:
            for tag in ['light-light', 'heavy-light']:
                mean = gv.mean(ds[tag].mass)  # Central value from "meff"
                err = gv.sdev(prior[f"{tag}:dE"][0])
                prior[f"{tag}:dE"][0] = gv.gvar(mean, err)
            mean = gv.mean(ds.v_guess)  # Central value from ratio R
            err = 0.5 * mean
            prior['Vnn'][0,0] = gv.gvar(mean, err)
        super(FormFactorPriorD2D, self).__init__(mapping=prior, **kwargs)


class MesonPriorPDG(BasePrior):
    """
    Class for building priors for analysis of pion 2pt functions inspired by
    physical results in the PDG.
    Args:
        nstates: namedtuple
        tag: str, the name of the state
        a_fm: float, the lattice spacing in fm
        scale: float, amount by which to scale the spectrum with respect to the
            PDG value(s). This option is useful, e.g., for lighter-than-physical
            b-quarks. For instance, the physical b quark as mass
            m_b ~ 4.2 m_charm, but a simulation might use 4.0 m_charm. In this
            case, one might choose scale = 4.0 / 4.2. Default is unity.
        kwargs: key-word arguments passed to constructor for BasePrior`
    Returns:
        MesonPriorPDG, a dict-like object containing the prior
    """
    def __init__(self, nstates, tag, a_fm=None, scale=1.0, **kwargs):
        prior = {}
        # Decaying states
        prior[f"{tag}:dE"] = PhysicalSplittings(tag)(nstates.n, a_fm, scale)
        prior[f"{tag}:a"] = decay_amplitudes(nstates.n)
        # Oscillating states
        if nstates.no:
            prior[f"{tag}:dEo"] = PhysicalSplittings(f"{tag}_osc")(nstates.no, a_fm, scale)
            prior[f"{tag}:ao"] = osc_amplitudes(nstates.no)
        super(MesonPriorPDG, self).__init__(mapping=prior, **kwargs)


if __name__ == '__main__':
    main()
