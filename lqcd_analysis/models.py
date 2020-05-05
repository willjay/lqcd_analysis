"""
This module contains additional models for three-point correlations functions.
The models are subclasses of those in Lepage's corrfitter package.
"""
import numpy as np
import gvar as _gvar
from corrfitter import Corr3 as _Corr3

import logging
LOGGER = logging.getLogger(__name__)

def _abs(val):
    return val * np.sign(val)

def _parse_param(p, default=None):
    " Parse fit-parameter label "
    return p if isinstance(p, tuple) else (p, default)

class Corr3(_Corr3):
    """
    A subclass extending the functionality of corrfitter.Corr3.
    """
    def __init__(self, *args, **kwargs):
        self.pedestal = kwargs.get('pedestal')
        if hasattr(self.pedestal, 'sdev'):
            raise ValueError(f"pedestal cannot be gvar, found {self.pedestal}")
        _ = kwargs.pop('pedestal', None)
        super(Corr3, self).__init__(*args, **kwargs)

    def buildprior(self, prior, mopt=None, nterm=None):
        new_prior = super().buildprior(prior, mopt, nterm)
        if self.pedestal is not None:
            try:
                new_prior['log(fluctuation)'] = prior['log(fluctuation)']
            except:
                new_prior['fluctuation'] = prior['fluctuation']
        return new_prior
        
    def _buildprop(self, times, params, choice):
        """
        Builds an array of "propagators".
        The array has the form prop[i][j][t], where i = n or o (i.e., decaying
        or oscillating), j = oscillation level, and t is time.
        """
        # Get inputs
        if choice == 'a':
            signs, amps, dEs = self.sa, self.a, self.dEa
        elif choice == 'b':
            signs, amps, dEs = self.sb, self.b, self.dEb
        else:
            raise ValueError("_buildprop: choice must be 'a' or 'b'.")

        # Get minus-sign coefficients
        ofacs = (
            # decaying sign factor
            None if signs[0] == 0.0 else signs[0],
            # oscillating sign factor
            None if signs[1] == 0.0 else signs[1]*(-1)**times[None, :]
            )

        # Compute propagator
        prop = []
        for amp, dE, ofac in zip(amps, dEs, ofacs):
            if (amp is None) or (ofac is None):
                prop.append(None)
                continue
            energies = np.cumsum(params[dE])
            prop.append(
                ofac
                * params[amp][:, None]
                * np.exp(-times[None, :] * energies[:, None])
                )
        return np.array(prop)

    def _get_vertices(self, params, idx_i, idx_j):
        """
        Gets the matrix of vertices associated with the indices i and j.
        Recall that V is a matrix of tags, usually of the form
        V = [['Vnn', 'Vno'], ['Von', 'Voo']]
        """
        tag_ij = self.V[idx_i][idx_j]
        # By default, grab the matrix from params
        if tag_ij is not None:
            return params[tag_ij]
        # tag_ij is missing, try checking the transpose
        if (idx_i != idx_j) and self.symmetric_V:
            # Von is Vno.T or vice versa
            tag_ji = self.V[idx_j][idx_i]
            if tag_ji is None:
                return None
            return params[tag_ij].T
        # There's nowhere else to look. The matrix isn't there.
        return None

    def _unpack_vertices(self, vertices, size):
        """
        Unpacks symmetric matrix of vertices, assumed to be square.
        """
        iterator = iter(vertices)  # does copy
        vertices = np.empty((size, size), dtype=vertices.dtype)
        for idx_k in range(size):
            for idx_l in range(idx_k, size):
                vertices[idx_k, idx_l] = next(iterator)
                if idx_k != idx_l:
                    vertices[idx_l, idx_k] = vertices[idx_k, idx_l]
        return vertices

    def _apply_pedestal(self, vertices, params):
        """
        Treats V[0,0] using fluctuation upon a pedestal, i.e., 
        A[n][0] * Vnn[0,0] * B[n][0] -->
            A[n][0] * (pedestal +/- <fluctuation>) * B[n][0].
        The sign of the fluctuation is taken from the sign of V[0,0]. This 
        convention means that a log prior for 'fluctuation' can only only push
        the value of the matrix element *away* from the origin.
        """
        if self.pedestal is None:
            return vertices
        v_copy = np.array(vertices)      
        sign = np.sign(self.pedestal)
        v_copy[0, 0] = self.pedestal + sign * _abs(params['fluctuation'])
        return v_copy
    
    def _bind_with_vertices(self, aprop, bprop, params):
        """
        Binds propagators together with vertices according to
        A[n]*Vnn*B[n] + A[n]*Vno*B[o] + A[o]*Von*B[n] + A[o]*Voo*B[o],
        where A and B denote aprop and bprop, repsectively.
        """
        ans = 0.0
        for idx_i, apropi in enumerate(aprop):
            if apropi is None:
                continue
            for idx_j, bpropj in enumerate(bprop):
                if bpropj is None:
                    continue
                # Extract and unpack the matrix of vertices
                vertices = self._get_vertices(params, idx_i, idx_j)
                if vertices is None:
                    continue
                if (idx_i == idx_j) and self.symmetric_V:
                    vertices = self._unpack_vertices(vertices, len(apropi))
                if self.transpose_V:
                    vertices = vertices.T
                if (idx_i == 0) and (idx_j == 0):
                    vertices = self._apply_pedestal(vertices, params)
                # Accumulate the product of propagators and vertices
                if min(len(apropi), len(vertices), len(bpropj)) != 0:
                    ans += np.sum(apropi * np.dot(vertices, bpropj), axis=0)
        return ans

    def fitfcn(self, p, t=None):
        """
        Overrides default fitfcn of corrfitter.Corr3.
        """
        # setup
        if t is None:
            t = self.tfit
        ta = np.asarray(t)
        if len(ta.shape) != 1:
            raise ValueError('t must be 1-d array')
        tb = self.T - ta
        # initial propagators
        # aprop[i][j][t] where i = n or o and j = excitation level
        aprop = self._buildprop(ta, p, 'a')
        # final propagators
        # bprop[i][j][t] where i = n or o and j = excitation level
        bprop = self._buildprop(tb, p, 'b')
        # combine propagators with vertices
        return self._bind_with_vertices(aprop, bprop, p)

class ConstrainedCorr3(Corr3):
    """
    A subclass extending the functionality of corrfitter.Corr3.
    This subclass gives a "constrained" version of the three-point function
    where only the matrix elements 'Vnn', 'Vno', 'Von', and 'Voo' are allowed
    to vary in the fit. The amplitudes ('a' and 'b') and energy splittings 
    ('dE' and 'dEo') are fixed by the *central values* specified in the fit
    prior, discarding errors and correlations associated with these parameters.
    """
    def __init__(self, *args, **kwargs):
        LOGGER.info("Constrained 3pt model -- fixing 'a', 'b', 'dE', 'dEo'.")
        if kwargs.get('pedestal') is not None:
            raise ValueError("'pedestal' not supported for ConstrainedCorr3.")
        self.pfixed = None
        super(ConstrainedCorr3, self).__init__(*args, **kwargs)

    def fitfcn(self, p, t=None):
        """
        Overrides default fitfcn of corrfitter.Corr3.
        """
        # setup
        if t is None:
            t = self.tfit
        ta = np.asarray(t)
        if len(ta.shape) != 1:
            raise ValueError('t must be 1-d array')
        tb = self.T - ta
        # amplitudes, energies from fixed parameters in propagators
        # initial propagators
        # aprop[i][j][t] where i = n or o and j = excitation level
        aprop = self._buildprop(ta, 'a')
        # final propagators
        # bprop[i][j][t] where i = n or o and j = excitation level
        bprop = self._buildprop(tb, 'b')
        # combine propagators with vertices
        return self._bind_with_vertices(aprop, bprop, p)

    def buildpfixed(self, prior):
        """
        Isolates fixed parameters 'a', 'b', 'dE', and 'dEo' from the prior.
        """
        pfixed = {}
        for key in [self.a, self.b, self.dEa, self.dEb]:
            for subkey in key:  # basically, key = (decay, osc)
                if subkey is None:
                    continue
                value = prior[subkey]
                value = _gvar.mean(value)  # restrict to mean
                pfixed[subkey] = value
        return pfixed

    def buildprior(self, prior, mopt=None, nterm=None):
        """
        Builds a prior, removing and setting the fixed parameters beforehand.
        """
        self.pfixed = self.buildpfixed(prior)  # extract fixed parameters
        _prior = {key: prior[key] for key in prior if key not in self.pfixed}
        new_prior = self._buildprior(_prior, mopt, nterm)
        return new_prior

    def _buildprop(self, times, choice):
        """
        Builds an array of "propagators" using the fixed parameters
        """
        return super()._buildprop(times, self.pfixed, choice)

    def _buildprior(self, prior, mopt=None, nterm=None):
        """
        Based on corrfitter.py:698-804. Skips building the fixed parameters.
        """
        if nterm is None:
            nterm = mopt
        nterm = _parse_param(nterm, None)
        def resize_sym(Vii, ntermi):
            # N = size of Vii; ntermi is new max. dimension
            N = int(numpy.round((((8. * len(Vii) + 1.) ** 0.5 - 1.) / 2.)))
            if ntermi is None or N == ntermi:
                return Vii
            ans = []
            iterV = iter(Vii)
            for i in range(N):
                for j in range(i, N):
                    v = next(iterV)
                    if j < ntermi:
                        ans.append(v)
            return numpy.array(ans)
        ans = _gvar.BufferDict()

        # i,j range from n to o
        for i in range(2):
            for j in range(2):
                vij = self.V[i][j]
                if vij is None:
                    continue
                vij = _gvar.dictkey(prior, vij)
                if i == j and self.symmetric_V:
                    ans[vij] = (
                        prior[vij] if nterm[i] is None else
                        resize_sym(prior[vij], nterm[i])
                        )
                else:
                    if self.transpose_V:
                        ans[vij] = prior[vij][None:nterm[j], None:nterm[i]]
                    else:
                        ans[vij] = prior[vij][None:nterm[i], None:nterm[j]]

        # verify dimensions
        for ai, dEai in zip(self.a, self.dEa):
            if ai is None:
                continue
            ai, dEai = _gvar.get_dictkeys(self.pfixed, [ai, dEai])
            if len(self.pfixed[ai]) != len(self.pfixed[dEai]):
                raise ValueError(
                    'length mismatch between a and dEa for '
                    + str(self.datatag)
                    )
        for bj, dEbj in zip(self.b, self.dEb):
            if bj is None or dEbj is None:
                continue
            bj, dEbj = _gvar.get_dictkeys(self.pfixed, [bj, dEbj])
            if len(self.pfixed[bj]) != len(self.pfixed[dEbj]):
                raise ValueError(
                    'length mismatch between b and dEb for '
                    + str(self.datatag)
                    )
        for i in range(2):
            for j in range(2):
                Vij = self.V[i][j]
                if Vij is None:
                    continue
                ai, bj =  _gvar.get_dictkeys(self.pfixed,
                                             [self.a[i], self.b[j]])
                Vij = _gvar.dictkey(prior, Vij)
                if i == j and self.symmetric_V:
                    N = self.pfixed[ai].shape[0]
                    if self.pfixed[bj].shape[0] != N:
                        raise ValueError(
                            'length mismatch between a, b, and V for '
                            + str(self.datatag)
                            )
                    if len(ans[Vij].shape) != 1:
                        raise ValueError(
                            'symmetric_V=True => Vnn, Voo = 1-d arrays for '
                            + str(self.datatag)
                            )
                    if ans[Vij].shape[0] !=  (N * (N+1)) / 2:
                        raise ValueError(
                            'length mismatch between a, b, and V for '
                            + str(self.datatag)
                            )
                else:
                    ai, bj =  _gvar.get_dictkeys(self.pfixed,
                                                 [self.a[i], self.b[j]])
                    Vij = _gvar.dictkey(prior, Vij)
                    Vij_shape = (
                        ans[Vij].shape[::-1] if self.transpose_V else
                        ans[Vij].shape
                        )
                    if self.pfixed[ai].shape[0] != Vij_shape[0]:
                        raise ValueError(
                            'length mismatch between a and V for '
                            + str(self.datatag)
                            )
                    elif self.pfixed[bj].shape[0] != Vij_shape[1]:
                        raise ValueError(
                            'length mismatch between b and V for '
                            + str(self.datatag)
                            )
        return ans        