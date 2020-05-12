import os
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
            "covidsimulation.simulation",
            ["covidsimulation/simulation.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3", "-std=c++11"],
            extra_link_args=["-O2", "-march=native"],
            language="c++"
        ),
]

setup(
    name="covidsimulation",
    version="0.1.2",
    packages=find_packages(exclude=["test*"]),
    ext_modules = cythonize(extensions)
)
