from setuptools import setup
import os
import sys

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="stan-pyro",
    version="0.0.1",
    author="Guillaume Baudart, Louis Mandel",
    description="Run Stan models with Numpyro",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deepppl/runtimes/numpyro",
    packages=["stanpyro", "stanpyro_cuda"],
    install_requires=["pyro-ppl==1.5.1", "pandas"],
)
