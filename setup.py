import os
import re
import ast
from setuptools import setup, find_packages
from distutils.core import Extension
import numpy
from Cython.Distutils import build_ext


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    packages, links = [], []
    for line in lineiter:
        if line and (line.startswith("#") or line.startswith("-")):
            links.append(line)
        else:
            packages.append(line)
    return packages, links


def get_version():
    curr_dir = os.path.abspath(os.path.dirname(__file__))
    main_file = os.path.join(curr_dir, "head_detector/__init__.py")
    _version_re = re.compile(r"__version__\s+=\s+(?P<version>.*)")
    with open(main_file, "r", encoding="utf8") as f:
        match = _version_re.search(f.read())
        version = match.group("version") if match is not None else '"unknown"'
    return str(ast.literal_eval(version))


if __name__ == "__main__":
    packages, links = parse_requirements("requirements.txt")
    setup(
        name="head_detector",
        version=get_version(),
        author="Orest Kupyn",
        description="VGGHeads: A Large-Scale Synthetic Dataset for 3D Human Heads",
        package_dir={"": "."},
        url="https://github.com/KupynOrest/head_detector",
        packages=find_packages(include=['head_detector']),
        include_package_data=True,
        install_requires=packages,
        dependency_links=links,
        package_data={
            '': ['*.pkl'],
            'head_detector': ['assets/*', 'assets/flame_indices/*'],
        },
        cmdclass={"build_ext": build_ext},
        ext_modules=[
            Extension(
                "Sim3DR_Cython",
                sources=[
                    "./head_detector/Sim3DR/lib/rasterize_kernel.cpp",
                    "./head_detector/Sim3DR/lib/rasterize.pyx",
                ],
                language="c++",
                include_dirs=[numpy.get_include()],
                extra_compile_args=["-std=c++11"],
            )
        ],
    )
