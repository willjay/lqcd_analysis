"""
Visualization routines
"""
import gvar as gv
import pylab as plt
import seaborn as sns


def color_palette(*args, **kwargs):
    return sns.color_palette(*args, **kwargs)


def subplots(*args, **kwargs):
    """
    Wraps pylab.subplots(...)
    TODO: fix imports
    Backstory:
    pylab is supposed to be deprecated (?) due to its state-machine environment
    Related infor[https://matplotlib.org/faq/usage_faq.html]
    """
    return plt.subplots(*args, **kwargs)


def plot(ax, y, **kwargs):
    """Plot y on axis ax."""
    x = range(len(y))
    errorbar(ax, x, y, **kwargs)


def errorbar(ax, x, y, bands=False, **kwargs):
    """Wrapper to plot gvars using the matplotlib function errorbar."""
    xerr = gv.sdev(x)
    x = gv.mean(x)
    yerr = gv.sdev(y)
    y = gv.mean(y)
    if bands:
        ax.errorbar(x=x, y=y, **kwargs)
        facecolor = kwargs.get('color', ax.lines[-1].get_color())
        alpha = kwargs.get('alpha', 1.0)
        ax.fill_between(
            x,
            y - yerr,
            y + yerr,
            facecolor=facecolor,
            alpha=alpha)
    else:
        ax.errorbar(x=x, xerr=xerr, y=y, yerr=yerr, **kwargs)
    return ax


def axhline(ax, y, alpha=None, **kwargs):
    """Wrapper to plot gvars using matplotlib function axhline."""
    if alpha is None:
        alpha = 0.25
    mean = gv.mean(y)
    ax.axhline(mean, **kwargs)
    color = kwargs.get('color', 'k')
    axhspan(ax, y, alpha=alpha, color=color)
    return ax


def axhspan(ax, y, **kwargs):
    """Wrapper to plot gvars using matplotlib function axhspan."""
    mean = gv.mean(y)
    err = gv.sdev(y)
    ax.axhspan(mean - err, mean + err, **kwargs)


def axvline(ax, x, alpha=None, **kwargs):
    """Wrapper to plot gvars using matplotlib function axvline."""
    if alpha is None:
        alpha = 0.25
    mean = gv.mean(x)
    ax.axvline(mean, **kwargs)
    color = kwargs.get('color', 'k')
    axvspan(ax, x, alpha=alpha, color=color)


def axvspan(ax, x, **kwargs):
    """Wrapper to plot gvars using matplotlib function axvspan."""
    mean = gv.mean(x)
    err = gv.sdev(x)
    ax.axvspan(mean - err, mean + err, **kwargs)
