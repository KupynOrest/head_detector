# flake8: noqa
"""
python setup.py build_ext -i
to compile
"""

from distutils.core import setup, Extension

import numpy
from Cython.Distutils import build_ext

setup(
    name="Sim3DR_Cython",  # not the package name
    cmdclass={"build_ext": build_ext},
    ext_modules=[
        Extension(
            "Sim3DR_Cython",
            sources=["./lib/rasterize.pyx", "./lib/rasterize_kernel.cpp"],
            language="c++",
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-std=c++11"],
        )
    ],
)
