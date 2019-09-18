"""
Visualization routines
"""
import gvar as gv


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
        if 'color' in kwargs:
            facecolor = kwargs['color']
        else:
            facecolor = ax.lines[-1].get_color()
        alpha = kwargs['alpha'] if 'alpha' in kwargs else 1.0
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
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = 'k'
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
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = 'k'
    axvspan(ax, x, alpha=alpha, color=color)


def axvspan(ax, x, **kwargs):
    """Wrapper to plot gvars using matplotlib function axvspan."""
    mean = gv.mean(x)
    err = gv.sdev(x)
    ax.axvspan(mean - err, mean + err, **kwargs)
