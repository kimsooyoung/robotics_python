import numpy
from setuptools import setup
from Cython.Build import cythonize

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
    ext_modules=cythonize(
        [
            'cython_test.pyx',
            'nlink_rhs_cython.pyx',
        ],
        annotate=True
    ),
    include_dirs=[numpy.get_include()],
)


# python setup.py build_ext --inplace