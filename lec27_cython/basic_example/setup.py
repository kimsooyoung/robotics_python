from setuptools import setup
from Cython.Build import cythonize

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
    ext_modules=cythonize(
        'integrate_cython.pyx',
        annotate=True
    ),
)

# python setup.py build_ext --inplace