"""
BasePrior
 * MesonPrior
 * FormFactorPrior
 * GoldstonePrior
"""
import numpy as np
import gvar as gv


def main():
    """TODO: Add main function."""


def _is_log(key):
    """Check if the key has the form 'log(*)'."""
    if key.startswith('log(') and key.endswith(')'):
        return True
    return False


def _check_duplicate_keys(keys):
    """Avoid key duplication during initialization."""
    log_keys = [key for key in keys if _is_log(key)]
    for log_key in log_keys:
        exp_key = log_key[4:-1]
        if exp_key in keys:
            msg = "Cannot have keys '{0}' and '{1}' together.".\
                format(log_key, exp_key)
            msg += " Pick one or the other."
            raise ValueError(msg)


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
            if 'dE' in key:
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
        prior['dE'] = [gv.gvar('0.5(1.5)') for _ in n]
        if 'a' in amps:
            prior['a'] = [gv.gvar('1(1)') for _ in n]
        if 'b' in amps:
            prior['b'] = [gv.gvar('1(1)') for _ in n]

        # Oscillating eneriges and amplitudes
        if n_oscillate > 0:
            no = range(n_oscillate)
            prior['dEo'] = [gv.gvar('1.0(1.5)') for _ in no]
            if 'ao' in amps:
                prior['ao'] = [gv.gvar('1(1)') for _ in no]
            if 'bo' in amps:
                prior['bo'] = [gv.gvar('1(1)') for _ in no]

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

    def __init__(self, nstates, ds=None, positive_ff=True, **kwargs):
        if ds is None:
            ds = {}
        else:
            FormFactorPrior._verify_tags(ds._tags)
        self.positive_ff = positive_ff
        super(FormFactorPrior, self).__init__(
                FormFactorPrior._build(nstates, ds, positive_ff), **kwargs
            )

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

    @staticmethod
    def _build(nstates, ds, positive_ff):
        """Build the prior dict."""
        prior = FormFactorPrior._make_meson_prior(nstates, ds)
        tmp = FormFactorPrior._make_vmatrix_prior(nstates, ds, positive_ff)
        for key in tmp:
            prior[key] = tmp[key]
        return prior

    @staticmethod
    def _make_meson_prior(nstates, ds):
        """Build prior associated with the meson two-point functions."""
        tags = ds._tags
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

    @staticmethod
    def _make_vmatrix_prior(nstates, ds, positive_ff):
        """Build prior for the 'mixing matrices' Vnn, Vno, Von, and Voo."""
        mass = None

        n = nstates.n
        no = nstates.no
        m = nstates.m
        mo = nstates.mo
        mass = ds[ds._tags.src].fastfit.E

        # General guesses
        tmp_prior = {}
        tmp_prior['Vnn'] = gv.gvar(n * [m * ['0.1(2.0)']])
        tmp_prior['Vno'] = gv.gvar(n * [mo * ['0.1(2.0)']])
        tmp_prior['Von'] = gv.gvar(no * [m * ['0.1(2.0)']])
        tmp_prior['Voo'] = gv.gvar(no * [mo * ['0.1(2.0)']])

        if mass is not None:
            if positive_ff:
                sign = 1
            else:
                sign = -1
            r = FormFactorPrior._estimate_r(ds, sign)
            v = FormFactorPrior._r_to_v(r, mass)
            tmp_prior['Vnn'][0, 0] = v

        return tmp_prior

    @staticmethod
    def _estimate_r(ds, sign=1):
        """
        Estimate the 'plateau' value from ratios of 2- and 3-point functions.
        Args:
            ds: FormFactorDatset
        Returns:
            float, an estimate of the plateau
        """
        maxes = []
        for T_sink in ds.rbar:
            val = ds.rbar[T_sink]
            local_max = max(sign * gv.mean(val[:T_sink - 2]))
            maxes.append(local_max)
        r_guess = max(maxes) * 1.05
        return r_guess

    @staticmethod
    def _r_to_v(r_plateau, mass):
        """
        Compute the matrix element 'Vnn' from the plateau value 'R'.
        This matrix element is the form factor, up to some conventional
        normalization factors. The formula is $V_{nn} = R / \\sqrt{2 M_L}$,
        where $M_L$ is the mass of the 'light' state. For heavy-to-light
        transitions like D to K/pi, $M_L$ would be kaon or pion mass.
        For "degenerate" heavy-heavy transitions, $M_L$ is simply the mass.
        Args:
            r: the value for the plateau
            mass: the mass of the 'light' state
        Returns:
            gvar, the estimate of the matrix element 'Vnn'
        """
        vnn = gv.mean(r_plateau / np.sqrt(2.0 * mass))
        width = np.abs(0.5 * vnn)
        return gv.gvar(vnn, width)


if __name__ == '__main__':
    main()
