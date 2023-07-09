import numpy
from setuptools import setup
from Cython.Build import cythonize

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
    ext_modules=cythonize(
        [
            'humanoid_rhs_cython.pyx',
            'collision_cython.pyx',
            'hip_positions_cython.pyx',
            'hip_velocities_cython.pyx',
            'joint_locations_cython.pyx',
        ],
        annotate=True
    ),
    include_dirs=[numpy.get_include()],
)

# MacOS & Windows
# python setup.py build_ext --inplace

# Ubuntu
# python3 setup.py build_ext --inplace