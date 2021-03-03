"""
Implements a serializable extension of the various "nonlinear_fit" objects
appearing in lsqfit.
"""
import sys
import logging
import numpy as np
import gvar as gv 
import sys
import datetime as datetime
from . import statistics

LOGGER = logging.getLogger(__name__)

def _to_text(adict):
    """ Wrapper for converting dicts to text for postgres"""
    new_dict = {}
    for key, val in sorted(adict.items()):
        new_dict[key] = str(val)
    return '$delim${{{0}}}$delim$'.format(str(new_dict))

class SerializableNonlinearFit:
    def __init__(self, fit):
        # Copy over attributes to serializable instance
        for attr in dir(fit):
            if attr in ['__class__', '__weakref__', 'p', 'format']:
                continue
            self.__setattr__(attr, fit.__getattribute__(attr))
        if np.isnan(fit.chi2) or np.isinf(fit.chi2):
            self.failed = True
        else:
            self.failed = False
        stats = statistics.FitStats(fit)
        for attr in ['chi2','chi2_aug','nparams','ndata','q_value','p_value',
                     'aic','model_probability']:    
            self.__setattr__(attr, stats.__getattribute__(attr))    

    @property
    def p(self):
        return self._getp()
    
    def serialize(self, rawtext=True):
        payload = {
            'prior': _to_text(self.prior) if rawtext else self.prior,
            'params': _to_text(self.p) if rawtext else self.p,
            'q': self.Q,
            'chi2_aug': self.chi2_aug,
            'chi2': self.chi2,
            'chi2_per_dof': self.chi2_aug/self.dof,
            'dof': self.dof,
            'q_value': self.q_value,
            'p_value': self.p_value,
            'nparams': self.nparams,
            'npoints': self.ndata,
            'calcdate': datetime.datetime.now(),
            'aic': self.aic,
            'model_probability': self.model_probability,
        }
        return payload

    def format(self, maxline=0, pstyle='v', nline=None, extend=True):
        """ Formats fit output details into a string for printing.

        The output tabulates the ``chi**2`` per degree of freedom of the fit
        (``chi2/dof``), the number of degrees of freedom, the ``Q``  value of
        the fit (ie, p-value), and the logarithm of the Gaussian Bayes Factor
        for the fit (``logGBF``). At the end it lists the SVD cut, the number
        of eigenmodes modified by the SVD cut, the tolerances used in the fit,
        and the time in seconds needed to do the fit. The tolerance used to
        terminate the fit is marked with an asterisk. It also lists
        information about the fitter used if it is other than the standard
        choice.

        Optionally, ``format`` will also list the best-fit values
        for the fit parameters together with the prior for each (in ``[]`` on
        each line). Lines for parameters that deviate from their prior by more
        than one (prior) standard deviation are marked with asterisks, with
        the number of asterisks equal to the number of standard deviations (up
        to five). Lines for parameters designated as linear (see ``linear``
        keyword) are marked with a minus sign after their prior.

        ``format`` can also list all of the data and the corresponding values
        from the fit, again with asterisks on lines  where there is a
        significant discrepancy.

        Args:
            maxline (int or bool): Maximum number of data points for which
                fit results and input data are tabulated. ``maxline<0``
                implies that only ``chi2``, ``Q``, ``logGBF``, and ``itns``
                are tabulated; no parameter values are included. Setting
                ``maxline=True`` prints all data points; setting it
                equal ``None`` or ``False`` is the same as setting
                it equal to ``-1``. Default is ``maxline=0``.
            pstyle (str or None): Style used for parameter list. Supported
                values are 'vv' for very verbose, 'v' for verbose, and 'm' for
                minimal. When 'm' is set, only parameters whose values differ
                from their prior values are listed. Setting ``pstyle=None``
                implies no parameters are listed.
            extend (bool): If ``True``, extend the parameter list to
                include values derived from log-normal or other
                non-Gaussian parameters. So values for fit parameter
                ``p['log(a)']``, for example, are listed together with
                values ``p['a']`` for the exponential of the fit parameter.
                Setting ``extend=False`` means that only the value
                for ``p['log(a)']`` is listed. Default is ``True``.

        Returns:
            String containing detailed information about fit.
        """
        # unpack arguments
        if nline is not None and maxline == 0:
            maxline = nline         # for legacy code (old name)
        if maxline is True:
            # print all data
            maxline = sys.maxsize
        if maxline is False or maxline is None:
            maxline = -1
        if pstyle is not None:
            if pstyle[:2] == 'vv':
                pstyle = 'vv'
            elif pstyle[:1] == 'v':
                pstyle = 'v'
            elif pstyle[:1] == 'm':
                pstyle = 'm'
            else:
                raise ValueError("Invalid pstyle: "+str(pstyle))

        def collect(v1, v2, style='v', stride=1, extend=False):
            """ Collect data from v1 and v2 into table.

            Returns list of [label,v1fmt,v2fmt]s for each entry in v1 and
            v2. Here v1fmt and v2fmt are strings representing entries in v1
            and v2, while label is assembled from the key/index of the
            entry.
            """
            def nstar(v1, v2):
                sdev = max(v1.sdev, v2.sdev)
                if sdev == 0:
                    nstar = 5
                else:
                    try: 
                        nstar = int(abs(v1.mean - v2.mean) / sdev)
                    except:
                        return '  ! (Error)'
                if nstar > 5:
                    nstar = 5
                elif nstar < 1:
                    nstar = 0
                return '  ' + nstar * '*'
            ct = 0
            ans = []
            width = [0,0,0]
            stars = []
            if v1.shape is None:
                # BufferDict
                keys = list(v1.keys())
                if extend:
                    v1 = gv.BufferDict(v1)
                    v2 = gv.BufferDict(v2)
                    ekeys = v1.extension_keys()
                    if len(ekeys) > 0:
                        first_ekey = ekeys[0]
                        keys += ekeys
                    else:
                        extend = False
                for k in keys:
                    if extend and k == first_ekey:
                        # marker indicating beginning of extra keys
                        stars.append(None)
                        ans.append(None)
                    ktag = str(k)
                    if np.shape(v1[k]) == ():
                        if ct%stride != 0:
                            ct += 1
                            continue
                        if style in ['v','m']:
                            v1fmt = v1[k].fmt(sep=' ')
                            v2fmt = v2[k].fmt(sep=' ')
                        else:
                            v1fmt = v1[k].fmt(-1)
                            v2fmt = v2[k].fmt(-1)
                        if style == 'm' and v1fmt == v2fmt:
                            ct += 1
                            continue
                        stars.append(nstar(v1[k], v2[k]))
                        ans.append([ktag, v1fmt, v2fmt])
                        w = [len(ai) for ai in ans[-1]]
                        for i, (wo, wn) in enumerate(zip(width, w)):
                            if wn > wo:
                                width[i] = wn
                        ct += 1
                    else:
                        ktag = ktag + " "
                        for i in np.ndindex(v1[k].shape):
                            if ct%stride != 0:
                                ct += 1
                                continue
                            ifmt = (len(i)*"%d,")[:-1] % i
                            if style in ['v','m']:
                                v1fmt = v1[k][i].fmt(sep=' ')
                                v2fmt = v2[k][i].fmt(sep=' ')
                            else:
                                v1fmt = v1[k][i].fmt(-1)
                                v2fmt = v2[k][i].fmt(-1)
                            if style == 'm' and v1fmt == v2fmt:
                                ct += 1
                                continue
                            stars.append(nstar(v1[k][i], v2[k][i]))
                            ans.append([ktag+ifmt, v1fmt, v2fmt])
                            w = [len(ai) for ai in ans[-1]]
                            for i, (wo, wn) in enumerate(zip(width, w)):
                                if wn > wo:
                                    width[i] = wn
                            ct += 1
                            ktag = ""
            else:
                # np array
                v2 = np.asarray(v2)
                for k in np.ndindex(v1.shape):
                    # convert array(GVar) to GVar
                    v1k = v1[k] if hasattr(v1[k], 'fmt') else v1[k].flat[0]
                    v2k = v2[k] if hasattr(v2[k], 'fmt') else v2[k].flat[0]
                    if ct%stride != 0:
                        ct += 1
                        continue
                    kfmt = (len(k) * "%d,")[:-1] % k
                    if style in ['v','m']:
                        v1fmt = v1k.fmt(sep=' ')
                        v2fmt = v2k.fmt(sep=' ')
                    else:
                        v1fmt = v1k.fmt(-1)
                        v2fmt = v2k.fmt(-1)
                    if style == 'm' and v1fmt == v2fmt:
                        ct += 1
                        continue
                    stars.append(nstar(v1k, v2k)) ###
                    ans.append([kfmt, v1fmt, v2fmt])
                    w = [len(ai) for ai in ans[-1]]
                    for i, (wo, wn) in enumerate(zip(width, w)):
                        if wn > wo:
                            width[i] = wn
                    ct += 1

            collect.width = width
            collect.stars = stars
            return ans

        # build header
        # Bayesian statistics
        dof = self.dof
        if dof > 0:
            chi2_dof = self.chi2_aug/self.dof
        else:
            chi2_dof = self.chi2_aug
        try:
            Q = 'Q = %.2g' % self.Q
        except:
            Q = ''
        try:
            logGBF = 'logGBF = %.5g' % self.logGBF
        except:
            logGBF = ''
        # frequentist statistics
        dof_freq = self.ndata - self.nparams
        if dof_freq > 0:
            chi2_dof_freq = self.chi2/dof_freq
        else:
            chi2_dof_freq = self.chi2

        if self.prior is None:
            descr = ' (no prior)'
        else:
            descr = ''
        # table = ('Least Square Fit%s:\n  chi2/dof [dof] = %.2g [%d]    %s'
        #             '    %s\n' % (descr, chi2_dof, dof, Q, logGBF))
        table = (
            f"{'#' * 80}\n"
            f"Least Square Fit{descr}:\n"
            "Bayesian summary:\n"
            "  Counting using dof = ndata\n"
            "  Counting using the augmented chi2 function\n"
            f"  chi2_aug/dof [dof] = {chi2_dof:.2f} [{dof}]    {Q}    {logGBF}\n"
            "Frequentist summary:\n"
            "  Counting dof = (ndata - nparams)\n"
            "  Counting using the correlated chi2 function only\n"
            f"  chi2/dof [dof] = {chi2_dof_freq:.2f} [{dof_freq}]"
            f"    p={self.p_value:.2f}\n"
        )
        if maxline < 0:
            return table

        # create parameter table
        if pstyle is not None:
            table = table + '\nParameters:\n'
            prior = self.prior
            if prior is None:
                if self.p0.shape is None:
                    prior = gv.BufferDict(
                        self.p0, buf=self.p0.flatten() + gv.gvar(0,float('inf')))
                else:
                    prior = self.p0 + gv.gvar(0,float('inf'))
            data = collect(self.palt, prior, style=pstyle, stride=1, extend=extend)
            w1, w2, w3 = collect.width
            fst = "%%%ds%s%%%ds%s[ %%%ds ]" % (
                max(w1, 15), 3 * ' ',
                max(w2, 10), int(max(w2,10)/2) * ' ', max(w3,10)
                )
            if len(self.linear) > 0:
                spacer = [' ', '-']
            else:
                spacer = ['', '']
            for i, (di, stars) in enumerate(zip(data, collect.stars)):
                if di is None:
                    # marker for boundary between true fit parameters and derived parameters
                    ndashes = (
                        max(w1, 15) + 3 + max(w2, 10) + int(max(w2, 10)/2)
                        + 4 + max(w3, 10)
                        )
                    table += ndashes * '-' + '\n'
                    continue
                table += (
                    (fst % tuple(di)) +
                    spacer[i in self.linear] +
                    stars + '\n'
                    )

        # settings
        settings = "\nSettings:"
        # add_svdnoise named arg changed to noise in lsqfit 11.6.
        try:
            _noise = self.add_svdnoise
        except AttributeError:
            _noise = self.noise
        if not _noise or self.svdcut is None or self.svdcut < 0:
            settings += "\n  svdcut/n = {svdcut:.2g}/{svdn}".format(
                svdcut=self.svdcut if self.svdcut is not None else 0.0,
                svdn=self.svdn
                )
        else:
            settings += "\n  svdcut/n = {svdcut:.2g}/{svdn}*".format(
                    svdcut=self.svdcut, svdn=self.svdn
                    )
        criterion = self.stopping_criterion
        try:
            fmtstr = [
                "    tol = ({:.2g},{:.2g},{:.2g})",
                "    tol = ({:.2g}*,{:.2g},{:.2g})",
                "    tol = ({:.2g},{:.2g}*,{:.2g})",
                "    tol = ({:.2g},{:.2g},{:.2g}*)",
                ][criterion if criterion is not None else 0]
            settings += fmtstr.format(*self.tol)
        except:
            pass
        if criterion is not None and criterion == 0:
            settings +="    (itns/time = {itns}*/{time:.1f})".format(
                itns=self.nit, time=self.time
                )
        else:
            settings +="    (itns/time = {itns}/{time:.1f})".format(
                itns=self.nit, time=self.time
                )
        default_line = '\n  fitter = gsl_multifit    methods = lm/more/qr\n'
        newline = "\n  fitter = {}    {}\n".format(
            self.fitter, self.description
            )
        if newline != default_line:
            settings += newline
        else:
            settings += '\n'
        if maxline <= 0 or self.data is None:
            return table + settings
        # create table comparing fit results to data
        ny = self.y.size
        stride = 1 if maxline >= ny else (int(ny/maxline) + 1)
        if hasattr(self, 'fcn_p'):
            f = self.fcn_p
        elif self.x is False:
            f = self.fcn(self.p)
        else:
            f = self.fcn(self.x, self.p)
        if hasattr(f, 'keys'):
            f = gv.BufferDict(f)
        else:
            f = np.array(f)
        data = collect(self.y, f, style='v', stride=stride, extend=False)
        w1,w2,w3 = collect.width
        clabels = ("key","y[key]","f(p)[key]")
        if self.y.shape is not None and self.x is not False and self.x is not None:
            # use x[k] to label lines in table?
            try:
                x = np.array(self.x)
                xlist = []
                ct = 0
                for k in np.ndindex(x.shape):
                    if ct%stride != 0:
                        ct += 1
                        continue
                    xlist.append("%g" % x[k])
                assert len(xlist) == len(data)
            except:
                xlist = None
            if xlist is not None:
                for i,(d1,d2,d3) in enumerate(data):
                    data[i] = (xlist[i],d2,d3)
                clabels = ("x[k]","y[k]","f(x[k],p)")

        w1,w2,w3 = max(9,w1+4), max(9,w2+4), max(9,w3+4)
        table += "\nFit:\n"
        fst = "%%%ds%%%ds%%%ds\n" % (w1, w2, w3)
        table += fst % clabels
        table += (w1 + w2 + w3) * "-" + "\n"
        for di, stars in zip(data, collect.stars):
            table += fst[:-1] % tuple(di) + stars + '\n'

        return table + settings        
