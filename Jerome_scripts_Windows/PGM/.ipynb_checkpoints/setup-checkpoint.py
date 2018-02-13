try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

#from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(
name = 'cy_utilities',
ext_modules = cythonize("cy_utilities.pyx"),
include_dirs = [numpy.get_include()]
)
