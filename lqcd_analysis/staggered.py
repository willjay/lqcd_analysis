"""
Formulae related to technical details of staggered fermions.
"""
import sys

import numpy as np
import scipy
from scipy import optimize


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

def naik_epsilon(am0):
    """
    Naik epsilon term from the bare input mass.
    Args:
        am0: float, the bare input mass
    Returns:
        float, the Naik epsilon term appearing in the HISQ action.
    """
    am1 = m_rest(am0)
    return naik_n(am1) -1

def ch(am1):
    """
    Computes Eq. B4 of 1712.09262, used to remove tree-level mass-dependent
    discretization effects of matrix elements involving heavy quarks with
    the HISQ action.
    Args:
        am1: float, rest mass
    Returns:
        float, cosh(am1)*(1-(1/2)N sinh^2(am1))
    """
    return np.cosh(am1)*(1 - (1./2)*naik_n(am1)*np.sinh(am1)**2)

def chfac(am0):
    """
    Convenience function implements Eqs. B12, B13 of 1712.09262.
    Args:
        am0: float, bare input mass
    Returns:
        float: ch(am1(am0)^(1/2)
    """
    return np.sqrt(ch(m_rest(am0)))

def test_fcns(am0):
    """
    Print results from various functions defined here.
    Args:
        am0: float, bare input mass
    Returns:
        None
    """
    print('am0:', am0)
    am1 = m_rest(am0)
    print('m_rest:', am1)
    print('naik_eps:', naik_epsilon(am0))
    print('chfac:' chfac(am0))

if __name__ == "__main__":
    try:
        am0 = float(sys.argv[1])
    except (IndexError, ValueError):
        print("Enter 'am0' as an argument.")
        sys.exit(1)
    test_fcns(am0)
