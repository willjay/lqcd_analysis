#!/usr/bin/env python
""" Setup script for the lqcd_analysis. """

# from setuptools import setup, find_packages, Extension
from os import path
from io import open
import setuptools
from numpy.distutils.core import setup, Extension
# io.open is needed for projects that support Python 2.7
# It ensures open() defaults to text mode with universal newlines,
# and accepts an argument to specify the text encoding

# specify external modules here, e.g., external C++ or Fortran code
# EXT1 = Extension(name='xsec', sources=['nuchic/xsec.pyf',
#                                        'nuchic/currents_opt_v1.f90',
#                                        'nuchic/xsec.f90',
#                                        'nuchic/mathtool.f90',
#                                        'nuchic/nform.f90'])

HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='lqcd_analysis',
    version='1.0',
    description='Analysis Tools for Lattice QCD',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/willjay/lqcd_analysis',
    author='William Jay',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.7, <4',
    install_requires=[
        'numpy',
        'scipy',
        'gvar',
        'corrfitter',
        'lsqfit',
        'seaborn',
    ],
    # Provide executable script to run the main code
    #entry_points={'console_scripts': ['nuchic = nuchic.main:nu_chic', ], },
    entry_points={},
    #ext_modules=[EXT1], # external modlues
    package_data={'': ['data/*']},
    extras_require={
        'test': ['pytest', 'coverage', 'pytest-cov'],
    },
)










