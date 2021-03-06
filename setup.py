import os
import re
import sys

from setuptools import setup, find_packages


PY_VER = sys.version_info

if not PY_VER >= (3, 6):
    raise RuntimeError("dsmlt doesn't support Python earlier than 3.6")


def read(f):
    return open(os.path.join(os.path.dirname(__file__), f)).read().strip()


def read_requirements():
    with open("requirements.txt") as f:
        return f.read().strip().split("\n")


extras_require = {}


def read_version():
    regexp = re.compile(r'^__version__\W*=\W*"([\d.abrc]+)"')
    init_py = os.path.join(os.path.dirname(__file__), "dsmlt", "__init__.py")
    with open(init_py) as f:
        for line in f:
            match = regexp.match(line)
            if match is not None:
                return match.group(1)
        else:
            msg = "Cannot find version in dsmlt/__init__.py"
            raise RuntimeError(msg)


classifiers = [
    "License :: OSI Approved :: MIT License",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Operating System :: POSIX",
    "Development Status :: 3 - Dev",
]


setup(
    name="dsmlt",
    version=read_version(),
    description=(
        "dsmlt is a set of data science and machine learning " "tools"
    ),
    long_description="\n\n".join((read("README.rst"), read("CHANGES.txt"))),
    classifiers=classifiers,
    platforms=["POSIX"],
    author="Pavlo Stadnikov",
    author_email="paul.stadnikov@gmail.com",
    url="https://github.com/pawlyk/dsml-tools",
    download_url="https://github.com/pawlyk/dsml-tools",
    license="MIT",
    packages=find_packages(),
    install_requires=read_requirements(),
    extras_require=extras_require,
    include_package_data=True,
)
