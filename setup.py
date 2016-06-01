#!/usr/bin/env python3

from distutils.core import setup

setup(
    name='netintent',
    version='0.0.1',
    description='Experiment for looking at neural network internal intentions',
    long_description=open('README.md', 'r').read(),
    author='David Bau',
    author_email='davidbau@mit.edu',
    url='https://github.com/davidbau/netintent',
    packages=[],
    install_requires=[
        'numpy',
        'pillow',
        'python-templet',
        'progressbar2',
        'matplotlib',
        'scipy',
        'termcolor',
        'bokeh == 0.10.0',
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
