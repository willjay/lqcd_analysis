"""
Formulae related to technical details of staggered fermions.
"""
import numpy as np
import scipy


def sh(am1):
    """
    Computes the quark bare mass am0 given the quark rest mass am1.

    For the HISQ action, a quark's bare mass am0 is related to its corresponding
    rest mass am1 according to a transcendental function, am0 = sh(am1).
    The notation "sh" is intended remind readers that its functional form
    involves (nested) sinh functions. See equations (B3) and (B5) of
    "B- and D-meson leptonic decay constants from four-flavor lattice QCD"
    Phys.Rev.D 98 (2018) 7, 074512. https://arxiv.org/pdf/1712.09262.pdf.
    Args:
        am1: float, the quark rest mass in lattice units
    Returns:
        am0: float, the quark bare mass in lattice units
    """
    return np.sinh(am1)*(1 - (1./6.) * naik_n(am1) * np.sinh(am1)**2)


def naik_n(am1):
    """
    Computes the coefficient N of the Naik term in the HISQ action.

    The HISQ action is a sum of single-link and three-link (or "long link")
    terms. The three-link term is known as the Naik improvement term.
    Schematically,
    S = psibar [ gamma(mu) (a Delta_mu - N/6 * a^3 Delta_mu^3) + a m0] psi.
    In this expression:
    * m0 is the bare quark mass.
    * N = (1 + epsilon) is the Naik coefficient
    This function computes the coefficient N. The correction "epsilon" is needed
    to improve the dispersion relation for heavy quarks, am0 !<< 1. The formula
    for N = (1 + epsilon) equation (A4) of
    "B- and D-meson leptonic decay constants from four-flavor lattice QCD"
    Phys.Rev.D 98 (2018) 7, 074512. https://arxiv.org/pdf/1712.09262.pdf.
    Args:
        am1: float, the value of the quark rest mass in lattice units
    Returns:
        float, the value of the Naik coefficient N = (1 + epsilon)
    """
    x = 2.0*am1 / np.sinh(2.0*am1)
    return (4.0 - 2.0*np.sqrt(1.0 + 3.0*x)) / np.sinh(am1)**2


def m_rest(m_bare):
    """
    Computes the rest mass am1 given the bare mass am0 for a quark, solving the
    transcendental equation am0 = sh(am1) using Brent's Method.
    Args:
        m_bare: float, the quark bare mass
    Returns:
        float, the quark rest mass m_rest = am1
    """
    def func(am1):
        return sh(am1) - m_bare
    return scipy.optimize.brentq(func, 0.75*m_bare, 1.25*m_bare)
        
