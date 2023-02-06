from distutils.core import setup
from setuptools import dist

dist.Distribution().fetch_build_eggs(['Cython', 'numpy<=1.19'])

import numpy
from Cython.Build import cythonize

required = [
    "cython",
    "numpy",
    "torch",
    "editdistance",
    "scikit-learn",
    "tqdm",
    "pymoo"
]

setup(name='HVAE',
      version='0.1',
      description='Hierarchical Variational Autoencoder',
      author='smeznar',
      packages=['src'],
      setup_requires=["numpy", "Cython"],
      ext_modules=cythonize("src/cyfunc.pyx"),
      include_dirs=[numpy.get_include()],
      install_requires=required)
