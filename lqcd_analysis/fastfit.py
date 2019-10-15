"""
FastFit -- for two-point correlation functions
FormFactorFastFit -- for form factors
"""
import collections
import logging
import itertools
import numpy as np
import lsqfit
import gvar as gv

LOGGER = logging.getLogger(__name__)


class FastFit(object):
    """
    Gab(t) = sn * sum_i an[i]*bn[i] * fn(En[i], t)
               + so * sum_i ao[i]*bo[i] * fo(Eo[i], t)

    where ``(sn, so)`` is typically ``(1, -1)`` and ::

        fn(E, t) =  exp(-E*t) + exp(-E*(tp-t)) # tp>0 -- periodic
               or   exp(-E*t) - exp(-E*(-tp-t))# tp<0 -- anti-periodic
               or   exp(-E*t)                  # if tp is None (nonperiodic)

        fo(E, t) = (-1)**t * fn(E, t)
    """
    # This code closely mirrors that of Lepage's. For ease of future comparison
    # we avoid changing all of the variable names.
    # pylint: disable=invalid-name
    def __init__(self, data, ampl='0(1)', dE='1(1)', E=None, s=(1, -1),
                 tp=None, tmin=6, svdcut=1e-6, osc=False, nterm=10):
        """
        Args:
            data: list / array of correlator data
            ampl: str, the prior for the amplitude. Default is '0(1)'
            dE: str, the prior for the energy splitting. Default is '1(1)'
            E0: str, the prior for the ground state energy. Default is None,
                which corresponds to taking the value for dE for the ground
                state as well
            s: tuple (s, so) with the signs for the towers of decay and
                oscillating states, repsectively. Default is (1, -1)
            tp: int, the periodicity of the data. Negative tp denotes
                antiperiodic "sinh-like" data. Default is None, corresponding
                to exponential decay
            tmin: int, the smallest tmin to include in the "averaging"
            svdcut: float, the desired svd cut for the "averaging"
            osc: bool, whether to estimate the lowest-lying oscillating state
                instead of the decay state
            nterms: the number of terms to include in the towers of decaying
                and oscillating states.
        Raises:
            RuntimeError: Can't estimate energy when cosh(E) < 1.
        """
        self.osc = osc
        self.nterm = nterm
        s = self._to_tuple(s)
        a, ao = self._build(ampl)
        dE, dEo = self._build(dE, E)
        s, so = s
        tmax = None

        if tp is None:
            # Data is not periodic
            # Model data with exponential decay
            def g(E, t):
                return gv.exp(-E * t)

        elif tp > 0:
            # Data is periodic
            # Model with a cosh
            def g(E, t):
                return gv.exp(-E * t) + gv.exp(-E * (tp - t))

            if tmin > 1:
                tmax = -tmin + 1
            else:
                tmax = None
        elif tp < 0:
            # Data is antiperiodic
            # Model with a sinh
            def g(E, t):
                return gv.exp(-E * t) - gv.exp(-E * (-tp - t))
            # Reflect and fold the data around the midpoint
            tmid = int((-tp + 1) // 2)
            data_rest = lsqfit.wavg(
                [data[1:tmid], -data[-1:-tmid:-1]], svdcut=svdcut)
            data = np.array([data[0]] + list(data_rest))

        else:
            raise ValueError('FastFit: bad tp')

        t = np.arange(len(data))[tmin:tmax]
        data = data[tmin:tmax]

        if not t.size:
            raise ValueError('FastFit: tmin too large; no t values left')
        self.tmin = tmin
        self.tmax = tmax

        if osc:
            data *= (-1) ** t * so
            a, ao = ao, a
            dE, dEo = dEo, dE
            s, so = so, s

        d_data = 0.
        E = np.cumsum(dE)

        # Sum the tower of decaying states, excluding the ground state
        for aj, Ej in list(zip(a, E))[1:]:
            d_data += s * aj * g(Ej, t)

        # Sum the full tower of oscillating states
        if ao is not None and dEo is not None:
            Eo = np.cumsum(dEo)
            for aj, Ej in zip(ao, Eo):
                d_data += so * aj * g(Ej, t) * (-1) ** t

        # Marginalize over the exicted states
        data = data - d_data
        self.marginalized_data = data
        # Average over the remaining plateau
        meff = 0.5 * (data[2:] + data[:-2]) / data[1:-1]
        ratio = lsqfit.wavg(meff, prior=gv.cosh(E[0]), svdcut=svdcut)

        if ratio >= 1:
            self.E = type(ratio)(gv.arccosh(ratio), ratio.fit)
            self.ampl = lsqfit.wavg(data / g(self.E, t) / s,
                                    svdcut=svdcut, prior=a[0])
        else:
            LOGGER.warn(
                'Cannot estiamte energy in FastFit: cosh(E) = %s', 
                str(ratio)
            )
            self.E = None
            self.ampl = None

    def _to_tuple(self, val):
        """Convert val to tuple."""
        if isinstance(val, tuple):
            return val
        if self.osc:
            return (None, val)
        return (val, None)

    def _build(self, x, x0=(None, None)):
        x = self._to_tuple(x)
        x0 = self._to_tuple(x0)
        return (self._build_prior(x[0], x0[0]),
                self._build_prior(x[1], x0[1]))

    def _build_prior(self, x, x0):
        if x is None:
            return x
        x = gv.gvar(x)
        dx = 0 if abs(x.mean) > 0.1 * x.sdev else 0.2 * x.sdev
        xmean = x.mean
        xsdev = x.sdev

        if x0 is None:
            first_x = x
        else:
            first_x = gv.gvar(x0)

        return (
            [first_x + dx] +
            [gv.gvar(xmean + dx, xsdev) for i in range(self.nterm - 1)]
        )

    def __str__(self):
        return (
            "FastFit("
            "E: {} ampl: {} chi2/dof [dof]: {:.1f} {:.1f} [{}] "
            "Q: E:{:.1f} ampl:{:.1f} "
            "(tmin,tmax)=({},{}))"
        ).format(
            self.E, self.ampl, self.E.chi2 / self.E.dof,
            self.ampl.chi2 / self.ampl.dof, self.E.dof, self.E.Q, self.ampl.Q,
            self.tmin, self.tmax
        )
    # pylint: enable=invalid-name,protected-access 
    def to_dict(self):
        return {
            'energy': str(self.E),
            'ampl': str(self.ampl),
            'tmin': self.tmin,
            'tmax': self.tmax,
            'nterm': self.nterm,
            'osc': self.osc,
        }
            

FFRatioPrior = collections.namedtuple(
    'FFRatioPrior',
    field_names=['m_src', 'm_snk', 'r_guess'],
    defaults=[gv.gvar("0.25(0.25)")]
)


FFRatioData = collections.namedtuple(
    'FFRatioData',
    field_names=['rdata', 'tdata', 'tfit']
)


class FastFitRatio(object):
    """
    FastFitRatio: A quick and dirty Bayesian estimation of the 'plateau' of a
    ratio of two- and three-point correlation functions, which usually (up to
    normalization factors) corresponds to a matrix element or a form factor.

    A standard mesonic 3-point correlation function (using staggered fermions)
    has the form of Eq. 31 of Bailey et al PRD 79, 054507 (2009)
    [https://arxiv.org/abs/0811.3640]. In other words, it is a sum of
    exponentials with some phase factors. In Eq. 39 of the same paper, they
    consider a ratio of two- and three-point functions which has the form
    R(t;T) = plateau + <exponentially decaying terms>. Fig. 6 of the same paper
    shows the relatively flat plateaus even arise in practice.

    This class furnishes an estimate of the "plateau" by estimating the size of
    the "exponentially decaying terms" and removing them from data for R(t;t).

    This class should be useful for both for the ratio of "averaged"
    correlators of Eq. 39 or a raw/unaveraged version of Eq. 39 with the "bars"
    removed from all the terms.
    """
    def __init__(self, t_snk, data, prior,
                 ampl="0(1)", dE="0.25(0.25)", svdcut=1e-6, nterm=10):
        """
        Estimates the plateau by marginalizing over excited states.
        Args:
            t_snk: int, the sink time "T"
            data: namedtuple with attributes 'tdata','rdata', and 'tfit'
            prior: namedtuple with attributes 'm_src', 'm_snk', 'r_guess'
            ampl: str, estimate for the amplitudes in amplitude matrix.
                Default is "0(1)"
            dE: str, estimate for the energy splittings.
                Default is "0.25(0.25)"
            svdcut: float, size of the svdcut passed to lsqfit.wavg
            nterm: int, number of terms to include in the model. Default is 10.
        """
        self.t_snk = t_snk
        self.rdata = data.rdata
        self.tdata = np.array(data.tdata, dtype=float)
        self.tfit = data.tfit
        self.nterm = nterm
        # Estimates for the plateau and ground-states at the source and sink
        self.m_src = prior.m_src
        self.m_snk = prior.m_snk
        self.r_guess = prior.r_guess
        # Estimates for the tower of splittings at source and sink and for the
        # matrix of amplitudes A[m, n]
        dE_src = self._energies_above_ground_state(self.m_src, gv.gvar(dE))
        dE_snk = self._energies_above_ground_state(self.m_snk, gv.gvar(dE))
        amplitude = np.array([gv.gvar(ampl) for _ in range(nterm**2)])
        amplitude = amplitude.reshape(nterm, nterm)
        self.marginalized_data = self._marginalize(dE_src, dE_snk, amplitude)
        self.plateau = self._fit_plateau(svdcut)

    def _marginalize(self, dE_src, dE_snk, amplitude):
        """
        Marginalizes over the excited states using Bayesian priors.
        More precisely, we use our priors for the excited-state masses and
        amplitudes to subtract their contribution from the input data. This
        subtraction amounts to 'marginalization.'
        """
        # Isolate data up to the sink location
        t_snk = self.t_snk
        t = self.tdata[:t_snk]
        data = self.rdata[:t_snk]
        # Estimate contributions from the tower of excited states
        # pylint: disable=protected-access
        tower = np.zeros((t_snk, ), dtype=gv._gvarcore.GVar)
        # pylint: enable=protected-access
        for m, n in itertools.product(range(self.nterm), repeat=2):
            if (m == 0) and (n == 0):
                continue
            dE_m0 = dE_src[m]
            dE_n0 = dE_snk[n]
            phase = (-1)**(m*t) * (-1)**(n*(t_snk-t))
            tower +=\
                phase\
                * amplitude[m, n]\
                * self._model(t, t_snk, dE_m0, dE_n0)
        # Marginalize over the excited states
        return data - tower

    def _fit_plateau(self, svdcut):
        """Fits the marginalized data to a constant"""
        if self.tfit is not None:
            tfit = self.tfit
        else:
            tfit = np.arange(self.t_snk)
        # Average over the remaining plateau
        yfit = self.marginalized_data[tfit]
        return lsqfit.wavg(yfit, prior=self.r_guess, svdcut=svdcut)

    def _model(self, t, t_snk, dE_src, dE_snk):
        """
        Computes the 'model function' containing the exponential decay away
        from the source and sink operators.
        model(t; t_snk) = exp(-dE_src * t) * exp(-dE_snk * (t_snk - t))
        """
        return np.exp(-dE_src * t) * np.exp(-dE_snk * (t_snk - t))

    def _energies_above_ground_state(self, E0, splitting):
        """
        Computes the energies splittings above the ground state.
        For the nth state, the splitting is Delta E_{n,0} = E_n - E_0.
        """
        # splittings
        dE = [E0] + [splitting for _ in range(self.nterm - 1)]
        # energies
        E = np.cumsum(dE)
        # energy of nth state above ground state
        dE_n0 = E - E[0]
        return dE_n0

    def __str__(self):
        return (
            "FastFitRatio("
            "t_snk: {}, "
            "plateau: {}, "
            "chi2/dof [dof]: {:.1f} [{}], "
            "Q: {:.1f})"
        ).format(
            self.t_snk,
            self.plateau,
            self.plateau.chi2 / self.plateau.dof, self.plateau.dof,
            self.plateau.Q
        )
