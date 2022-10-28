"""
Tools for computnig finite-volume shifts to chiral logarithms.
"""
import os
import sys
import pandas as pd
import numpy as np
import itertools
from scipy.special import kn
import gvar as gv


def main():
    """
    Creates a table of mpi*L and finite volume shifts.
    """
    L = int(sys.argv[1])
    print(f"Computing finite volume corrections with L={L}")
    mpiL_min = 0.1
    mpiL_max = 10
    mpiL = np.linspace(mpiL_min, mpiL_max, num=100)
    print(f"Computing corrections for mpi*L in ({mpiL_min}, {mpiL_max})")
    coords = create_coords(L)
    shift = [coords['n'].apply(lambda n: summand_exact(n, x)).sum() for x in mpiL]
    for mL, delta in zip(mpiL, shift):
        print(f"{mL:.2f} {delta:.8f}")


def create_coords(L):
    """
    Creates a DataFrame with the coordinates of 3d lattice with spatial extent
    L and computes their norm.
    """
    coords = []
    for xyz in itertools.product(range(L), repeat=3):
        coords.append({'xyz': np.array(xyz)})
    df = pd.DataFrame(coords)
    df['n'] = df['xyz'].apply(lambda arr: np.sqrt(np.sum(arr**2)))
    mask = (df['n'] > 0)
    df = df[mask]
    return df


def summand_exact(n, mL):
    """
    Computes the summand in the exact sum in (A12) of
    D. Arndt and C.J.D. Lin
    Phys.Rev.D 70 (2004) 014503
    [https://arxiv.org/pdf/hep-lat/0403012.pdf]
    """
    nmL = n*mL
    return kn(1, nmL)/nmL


def table_interpolator(table_path=None):
    """
    Creates an interpolating function f(mpi*L) from a table of saved values
    for finite-volume shifts.
    """
    if table_path is None:
        table_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            'finite_volume.dat')
    # print("Reading table from", table_path)
    df = pd.read_csv(table_path, sep='\s+')
    return gv.cspline.CSpline(df['mpiL'].values, df['shift'].values, warn=True)


if __name__ == '__main__':
    main()