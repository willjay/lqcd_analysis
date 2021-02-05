"""
The module contains functions for estimating masses of hadrons using information
from experimental measurements as tabulated, e.g., by the Particle Data Group
(PDG).
"""
import logging
import re
import numpy as np

LOGGER = logging.getLogger(__name__)

MPI = 134.997 # MeV
MK = 497.611 # MeV
MD = 1864.830 # MeV
MDS = 1968.340 # MeV
MBS = 5366.890 # MeV
MB = 5279.320 # MeV
HBARC = 197.3269804 # MeV * fm


def estiamte_mass(state, alias_light, alias_heavy, a_fm=None):
    """
    Wrapper function for scale_mass(...) that provides support for switching to
    lattice units.
    """
    mass_mev = scale_mass(state, alias_light, alias_heavy)
    if not a_fm:
        return mass_mev
    return mass_mev * a_fm / HBARC


def scale_mass(state, alias_light, alias_heavy):
    """
    Estimates the mass of meson by scaling experimentally measured values as
    tabulated, e.g., by the Particle Data Group (PDG).
    Args:
        state: str, the name of a state like 'pi', 'k', 'd', 'ds', 'b', or 'bs'
        alias_light, alias_heavy: str, specification
            of the quarks in meson. Typical values include
            '1.0 m_light', '0.1 m_strange', '1.0 m_charm'
    Returns:
        float: an estimate for the mass in MeV

    Examples:
    ---------
    >>> scale_mass('pi', '1.0 m_light', '1.0 m_light'))
    140.0
    >>> scale_mass('pi', '0.037 m_strange', '0.037 m_strange'))
    139.92998249124454
    >>> scale_mass('k', '1.0 m_light', '1.0 m_strange'))
    498.0
    >>> scale_mass('k', '0.037 m_strange', '1.0 m_strange'))
    497.99110706345516
    >>> scale_mass('k', '0.2 m_strange', '1.0 m_strange'))
    535.7014627036752
    >>> scale_mass('d', '1.0 m_light', '1.0 m_charm'))
    1864.83
    >>> scale_mass('b', '1.0 m_light', '4.2 m_charm'))
    5279.32

    Notes:

    Pions and kaons are well-described as Goldstone bosons. Leading-order ChiPT
    suggests their masses should follow the Gell-Mann-Oakes-Renner relation.
    The square of the hadron mass should obey M^2 = const x mq, where mq is the
    average quark masses (mq1 + mq2) + 2.
    Thus: M_lattice = sqrt(mq_lattice / mq_pdg) x M_pdg.

    B and D mesons are heavy hadrons containing a "heavy" quark. In heavy quark
    effective theory, the mass of the hadron is proportional to the mass of the
    quark, suggesting that one could use
    M_lattice = (mh_lattice / mh_pdg) x M_pdg.
    However, for "intermediate" heavy quark masses appearing in many lattice
    simulations, this formula is not wholy satisfactory. For instance, for D
    mesons with heavier-than-physical charm quarks, the linear formula can
    grossly overestimate the true mass. Instead, it is convenient to use a
    power-law formula to interpolate between the light-quark and heavy-quark
    behaviors:
    M_lattice = (mh_lattice / mh_pdg)**alpha x M_pdg
    Heavy-quark limit: alpha --> 1
    Light-quark limit: alpha --> 0.5 (GMOR)
    Intermediate masses: alpha in (0.5, 1.0).
    A quick numerical study suggested that the following value for alpha will
    reproduce the correct mass to within perhaps 10 or 20%:
    D, Ds: alpha=0.65
    B, Bs: alpha=0.75
    Even this rough accuracy can be useful as an initial guess for a fitter.
    """
    # Match on tags like "1.0 m_light"
    mass_regex = re.compile(r"^(\d+[.]?\d+) m_(light|strange|charm)$")
    light = re.search(mass_regex, alias_light)
    heavy = re.search(mass_regex, alias_heavy)
    if not light:
        raise ValueError("Unrecognized tag %s" % alias_light)
    if not heavy:
        raise ValueError("Unrecognized tag %s" % alias_heavy)

    light_factor = float(light.group(1))
    light_quark = light.group(2)
    heavy_factor = float(heavy.group(1))
    heavy_quark = heavy.group(2)

    if state.lower() == 'pi':
        if alias_light != alias_heavy:
            raise ValueError(
                "Pions need matching quarks. Found %s, %s" %
                (alias_light, alias_heavy))

        if light_quark == 'light':
            return np.sqrt(light_factor) * MPI

        if light_quark == 'strange': 
            # Note: m_light = (1/27) x m_strange at the physical point
            return np.sqrt(27.0 * light_factor) * MPI

        raise ValueError(
            "Unrecognized quarks for a pion. Found %s, %s" %
            (alias_light, alias_heavy))

    if state.lower() == 'k':
        if (light_quark, heavy_quark) == ('light', 'strange'):
            # Note: m_light = (1/27) x m_strange at the physical point
            return np.sqrt(27./28.) * np.sqrt(light_factor/27. + heavy_factor) * mk

        if (light_quark, heavy_quark) == ('strange', 'strange'):
            # Note: m_light = (1/27) x m_strange at the physical point
            return np.sqrt(27./28.) * np.sqrt(light_factor + heavy_factor) * mk

        raise ValueError(
            "Unrecognized quarks for a kaon. Found %s, %s" %
            (alias_light, alias_heavy))

    if heavy_quark != 'charm':
        raise ValueError(
            "Unrecognized heavy quark for B or D mesons. Found %s"
            % heavy_quark)

    if state.lower() == 'd':
        return MD * (heavy_factor/1.0)**0.65

    if state.lower() == 'ds':
        return MDS * (heavy_factor/1.0)**0.65

    if state.lower() == 'b':
        # Note: m_bottom = (4.2) x m_charm at the physical point
        return MB * (heavy_factor/4.2)**0.75

    if state.lower() == 'bs':
        # Note: m_bottom = (4.2) x m_charm at the physical point
        return MBS * (heavy_factor/4.2)**0.75

    raise ValueError("Unrecognized inputs %s, %s %s" %
                     (state, alias_light, alias_heavy))
