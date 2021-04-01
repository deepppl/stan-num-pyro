from setuptools import setup
import os
import sys

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="stan-numpyro",
    version="0.0.1",
    author="Guillaume Baudart, Louis Mandel",
    description="Run Stan models with NumPyro",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deepppl/stan-num-pyro",
    packages=["stannumpyro"],
    install_requires=["numpyro==0.6.0", "pandas", "tensorflow_probability"],
)
