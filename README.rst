====
igel
====

.. image:: assets/cover.png
    :width: 100%
    :scale: 50%
    :align: center
    :alt: igel-icon

|

|


.. image:: https://img.shields.io/pypi/v/igel.svg
        :target: https://pypi.python.org/pypi/igel

.. image:: https://img.shields.io/travis/nidhaloff/igel.svg
        :target: https://travis-ci.com/nidhaloff/igel

.. image:: https://pepy.tech/badge/igel
        :target: https://pepy.tech/project/igel

.. image:: https://pepy.tech/badge/igel/month
        :target: https://pepy.tech/project/igel/month

.. image:: https://readthedocs.org/projects/igel/badge/?version=latest
        :target: https://igel.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/pypi/wheel/igel
        :alt: PyPI - Wheel
        :target: https://pypi.python.org/pypi/igel

.. image:: https://img.shields.io/pypi/status/igel
        :alt: PyPI - Status
        :target: https://pypi.python.org/pypi/igel

|
|

A machine learning tool that allows to train/fit, test and use models **without writing code**


* Free software: MIT license
* Documentation: https://igel.readthedocs.io.


.. note::

    The project is still under heavy development. Feel free to open an issue if you encountered any.
    I appreciate any feedback in order to improve the package ;)

Motivation & Goal
------------------

The goal of the project is to provide machine learning for **everyone**, both technical and non technical
users.

I needed a tool sometimes, which I can use to fast create a machine learning prototype. Whether to build
some proof of concept or create a fast draft model to prove a point. I find myself often stuck at writing
boilerplate code and/or thinking too much of how to start this.

Therefore, I decided to create **igel**. Hopefully, it will make it easier for technical and non technical
users to build machine learning models.

Intro
--------

igel is built on top of scikit-learn. It provides a simple way to use machine learning without writing
a **single line of code**

All you need is a yaml file, where you need to describe what you are trying to do. That's it!

Installation
-------------

- The easiest way is to install igel using `pip <https://packaging.python.org/guides/tool-recommendations/>`_

.. code-block:: console

    $ pip install igel

- Check the docs for other ways to install igel from source

Quick Start
------------

- First step is to provide a yaml file:

.. code-block:: yaml

        # model definition
        model:
            # in the type field, you can write the type of problem you want to solve. Whether regression or classification
            # Then, provide the algorithm you want to use on the data. Here I'm using the random forest algorithm
            type: regression
            algorithm: random forest

        # target you want to predict
        # Here, as an example, I'm using a dataset, where I want to predict the GPA values.
        # Depending on your data, you need to provide the target(s) you want to predict here
        target:
            - GPA

In the example above, we declare that we have a regression
problem and we want to use the random forest model
to solve it. Furthermore, the target we want to
predict is GPA (since I'm using this simple `dataset <https://www.kaggle.com/luddarell/101-simple-linear-regressioncsv>`_ )
`
- Run this command in Terminal, where you provide the **path to your dataset** and the **path to the yaml file**

.. code-block:: console

    $ igel fit --data_path 'path_to_your_csv_dataset.csv' --yaml_file 'path_to_your_yaml_file.yaml'


That's it. Your "trained" model can be now found in the model_results folder
(automatically created for you in your current working directory).
Furthermore, a description can be found in the description.json file inside the model_results folder.

Examples
----------
Check the examples folder, where you can use the csv data to run a simple example from terminal

TODO
-----
- add option as arguments to the models
- add multiple file support
- provide an api to evaluate models

Contributors
------------

None yet. Why not be the first?
Contributions are always welcome. Please check the contribution guidelines first.
