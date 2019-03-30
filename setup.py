#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy==1.14.5',
                'scipy==1.1.0']

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Rafael Lopes Conde dos Reis",
    author_email='rafael.lcreis@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="MV Port is a Python package to perform Mean-Variance Analysis. It provides a Portfolio class with a variety of methods to help on your portfolio optimization tasks.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='mvport',
    name='mvport',
    packages=find_packages(include=['mvport']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/condereis/mean-variance-portfolio',
    version='1.3.1',
    zip_safe=False,
)
