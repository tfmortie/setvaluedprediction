#!/usr/bin/env python
import os
from setuptools import setup

ROOT = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
    README = f.read()
with open(os.path.join("svp", "version.txt")) as f:
    VERSION = f.read().strip()

setup(
    name="setvaluedprediction",
    version=VERSION,
    license="MIT license",
    description="Set-valued predictors in Python",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Thomas Mortier",
    author_email="thomas.mortier92@gmail.com",
    url="https://github.com/tfmortie/setvaluedprediction",
    packages=["svp"],
    install_requires=[
        "Ninja",
        "joblib",
        "numpy",
        "pandas",
        "scikit-learn",
        "setuptools",
        "torch>=1.10.0",
    ],
    include_package_data=True,
)
