#!/usr/bin/env python3

from distutils.core import setup

setup(
    name='NRoot',
    version='0.0.1',
    description='Experiment for looking at data dependencies',
    long_description=open('README.md', 'r').read(),
    author='David Bau',
    author_email='david.bau@gmail.com',
    url='https://github.com/davidbau/nroot',
    packages=[],
    install_requires=[
        'numpy',
        'pillow',
        'python-templet',
        'progressbar2',
        'matplotlib',
        'scipy',
        'termcolor',
        'Theano >= 0.7.0',
        'Blocks',
    ],
    scripts=[],
    license='MIT',
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Education"
    ]
)
