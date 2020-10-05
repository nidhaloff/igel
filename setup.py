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

setup_requirements = ['pytest-runner']
test_requirements = ['pytest>=3']
# docs_require = ['sphinx-copybutton']

setup(
    author=__author__,
    author_email=__email__,
    maintainer_email=__email__,
    maintainer=__author__,
    python_requires='>=3.6',
    classifiers=[
        # 'Development Status :: 4 - Beta',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering',
        'Topic :: Education',
        'Topic :: Software Development',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="a machine learning tool that allows you to train, test and use models without writing code",
    entry_points={
        'console_scripts': [
            'igel=igel.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords=['igel',
              'machine learning',
              'artificial intelligence',
              'supervised learning',
              'unsupervised learning',
              'neural network',
              'linear regression',
              'logsitic regression',
              'random forest',
              'decision tree',
              'clustering',
              'support vector machine',
              'SVM',
              'ML',
              'sklearn',
              'scikit-learn',
              'regression',
              'classification'],
    name='igel',
    packages=find_packages(include=['igel', 'igel.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/nidhaloff/igel',
    download_url="https://pypi.org/project/igel/",
    version=__version__,
    zip_safe=False,
)
