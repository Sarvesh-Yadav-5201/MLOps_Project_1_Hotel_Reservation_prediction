##...........................PROJECT MANAJEMENT............................##
from setuptools import setup, find_packages

with open ('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name= 'mlops-hotel-1',
    version= '0.2',
    author= 'sarvesh',
    packages= find_packages(),
    install_requires= requirements,
)
