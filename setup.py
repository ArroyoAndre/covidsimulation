import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

cwd = os.getcwd()

os.chdir(os.path.join(cwd, 'covidsimulation'))

try:
    extensions = [
        Extension(
                "simulation", 
                ["simulation.pyx"], 
                include_dirs=[numpy.get_include()], 
            ),
    ]

    setup(
        ext_modules = cythonize(extensions)
        )

finally:
    os.chdir(cwd)