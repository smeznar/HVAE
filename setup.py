import os
from setuptools import setup, find_packages


with open("README.md", "r") as fl:
    long_description = fl.read()


def parse_requirements(file):
    required_packages = []
    with open(os.path.join(os.path.dirname(__file__), file)) as req_file:
        for line in req_file:
            required_packages.append(line.strip())
    return required_packages


setup(name='EDHiE',
      version='1.0.0',
      url='https://github.com/smeznar/HVAE',
      author='Sebastian Mežnar',
      author_email='smeznar@gmail.com',
      license='GNU General Public License v3.0',
      keywords=["symbolic regression", "equation discovery", "EDHiE", "HVAE", "machine learning"],
      description='EDHiE',
      long_description=long_description,
      long_description_content_type='text/markdown',
      py_modules=['EDHiE'],
      packages=find_packages(),
      classifiers=['Intended Audience :: Information Technology',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 3',
                   "Topic :: Scientific/Engineering",
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   "Topic :: Scientific/Engineering :: Mathematics"],

      install_requires=parse_requirements('requirements.txt'))