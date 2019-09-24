"""
FastFit -- for two-point correlation functions
FormFactorFastFit -- for form factors
"""
import numpy as np
import lsqfit
import gvar as gv


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
            raise ValueError('bad tp')

        t = np.arange(len(data))[tmin:tmax]
        data = data[tmin:tmax]

        if not t:
            raise ValueError('tmin too large; not t values left')

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

        # Average over the remaining plateau
        meff = 0.5 * (data[2:] + data[:-2]) / data[1:-1]
        ratio = lsqfit.wavg(meff, prior=gv.cosh(E[0]), svdcut=svdcut)

        if ratio >= 1:
            self.E = type(ratio)(gv.arccosh(ratio), ratio.fit)
        else:
            msg = "can't estimate energy: cosh(E) = {}".format(ratio)
            raise RuntimeError(msg)

        self.ampl = lsqfit.wavg(data / g(self.E, t) / s,
                                svdcut=svdcut, prior=a[0])

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
            "Q: {:.1f} {:.1f})"
        ).format(
            self.E, self.ampl, self.E.chi2 / self.E.dof,
            self.ampl.chi2 / self.ampl.dof, self.E.dof, self.E.Q, self.ampl.Q,
        )
    # pylint: enable=invalid-name,protected-access