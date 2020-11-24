#cython: language_level=3
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(
        "tools.pyx", 
        compiler_directives={'language_level' : "3"},
    ),
    include_dirs=[numpy.get_include()],
)