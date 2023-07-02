import numpy
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        [
            'convolve1_wo_optim.pyx',
            'convolve1.pyx',
            'compute_memview.pyx',
        ],                
        annotate=True),      # enables generation of the html annotation file
    include_dirs=[numpy.get_include()],
)

# python setup.py build_ext --inplace

# convolve_py : 4.981875833999999
# convolve1_wo_optim : 4.389919067
# convolve1 : 0.015742378000000556 (wo value checking)
# convolve1 : 0.009143810000001196