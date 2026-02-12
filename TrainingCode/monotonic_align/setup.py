from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension(
    name="core",
    sources=["core.pyx"],
    include_dirs=[numpy.get_include()]
)

setup(
    name='monotonic_align',
    ext_modules=cythonize(ext)
)
