"""
Build script for the Cython monotonic alignment core.

Usage (from the project root directory):
    python monotonic_align/setup.py build_ext --inplace

This places the compiled .so alongside __init__.py so that
  `from monotonic_align.core import maximum_path_c`
works without path hacks.
"""

import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Resolve paths relative to this file (monotonic_align/)
_dir = os.path.dirname(os.path.abspath(__file__))

ext = Extension(
    name="monotonic_align.core",
    sources=[os.path.join(_dir, "core.pyx")],
    include_dirs=[numpy.get_include()],
)

setup(
    name="monotonic_align",
    ext_modules=cythonize(ext, language_level="3"),
)
