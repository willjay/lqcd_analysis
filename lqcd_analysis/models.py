"""
This module contains additional models for three-point correlations functions.
The models are subclasses of those in Lepage's corrfitter package.
"""
import numpy as np
import corrfitter

class Corr3(corrfitter.Corr3):
    """
    A subclass extending the functionality of corrfitter.Corr3.
    """
    def __init__(self, *args, **kwargs):
        self.pedestal = kwargs.pop('pedestal')
        super(Corr3, self).__init__(*args, **kwargs)

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

    def _apply_pedestal(self, vertices):
        """
        Reinterprets V[0,0] as log prior for a positive fluctuation upon a
        pedestal. In other words
        A[n][0] * Vnn[0,0] * B[n][0] -->
            A[n][0] * (pedestal + <postive fluctuation>) * B[n][0],
        where <positive fluctuation> = exp(Vnn[0,0])
        """
        if self.pedestal is not None:
            vertices[0, 0] = self.pedestal + np.exp(vertices[0, 0])

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
                    self._apply_pedestal(vertices)
                # Accumulate the product of propagators and vertices
                if min(len(apropi), len(vertices), len(bpropj)) != 0:
                    ans += np.sum(apropi * np.dot(vertices, bpropj), axis=0)
        return ans

    def fitfcn(self, params, t=None):
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
        aprop = self._buildprop(ta, params, 'a')
        # final propagators
        # bprop[i][j][t] where i = n or o and j = excitation level
        bprop = self._buildprop(tb, params, 'b')
        # combine propagators with vertices
        return self._bind_with_vertices(aprop, bprop, params)
