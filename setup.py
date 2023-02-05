from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("src/cyfunc.pyx"),
    include_dirs=[numpy.get_include()]
)

# from distutils.core import setup
# import os
# from setuptools import dist
#
# # dist.Distribution().fetch_build_eggs(['Cython', 'numpy<=1.19'])
#
# import numpy
# from Cython.Build import cythonize
#
# required = [
#     "cython",
#     "numpy",
#     "torch",
#     "editdistance",
#     "scikit-learn",
#     "tqdm",
#     "pymoo"
# ]
#
# setup(name='HVAE',
#       version='0.5',
#       description='Hierarchical Variational Autoencoder.',
#       author='smeznar',
#       packages=['src'],
#       ext_modules=cythonize([os.path.join('src', 'fasteval', 'cyfunc.pyx')]),
#       install_requires=required)
