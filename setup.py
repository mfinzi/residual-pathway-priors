from setuptools import setup,find_packages
import sys, os

setup(name="Residual Pathway Priors",
      description="",
      version='0.0',
      author='-',
      author_email='maf820@nyu.edu',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['h5py','objax','pytest','sklearn',
      'olive-oil-ml @ git+https://github.com/mfinzi/olive-oil-ml',
      'emlp @ git+https://github.com/mfinzi/equivariant-MLP','optax','tqdm>=4.38'],
      packages=['rpp'],
      long_description=open('README.md').read(),
)
