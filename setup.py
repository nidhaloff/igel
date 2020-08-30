#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
from igel import __version__, __email__, __author__

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'pandas==1.1.1',
    'PyYAML==5.3.1',
    'scikit-learn==0.23.2'
]

setup_requirements = ['pytest-runner',]
test_requirements = ['pytest>=3']

print("requirements: ", requirements)

setup(
    author=__author__,
    author_email=__email__,
    python_requires='>=3.6',
    classifiers=[
        # 'Development Status :: 2 - Pre-Alpha',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="a machine learning tool that allows to train, test and use models without writing code",
    entry_points={
        'console_scripts': [
            'igel=igel.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='igel',
    name='igel',
    packages=find_packages(include=['igel', 'igel.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/nidhaloff/igel',
    version=__version__,
    zip_safe=False,
)
