====
igel
====

.. image:: https://github.com/nidhaloff/igel/blob/master/assets/logo.png
    :width: 50
    :align: center
    :alt: igel-icon

.. image:: https://img.shields.io/pypi/v/igel.svg
        :target: https://pypi.python.org/pypi/igel

.. image:: https://img.shields.io/travis/nidhaloff/igel.svg
        :target: https://travis-ci.com/nidhaloff/igel

.. image:: https://readthedocs.org/projects/igel/badge/?version=latest
        :target: https://igel.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




A machine learning tool that allows to train/fit, test and use models without writing code


* Free software: MIT license
* Documentation: https://igel.readthedocs.io.


.. note::

    The project is under heavy development. Feel free to open an issue if you encountered any

Intro
--------

igel is built on top of scikit-learn. It provides a simple way to use machine learning without writing
a **single line of code**

All you need is a yaml file, where you need to describe what you are trying to do. That's it!

Quick Start
------------

- First step is to provide a yaml file:

.. code-block:: yaml

        # model definition
        model:
            type: regression
            algorithm: forest

        # target you want to predict
        target:
            - GPA

In the example above, we declare that we have a regression
problem and we want to use the random forest model
to solve it. Furthermore, the target we want to
predict is GPA (since I'm using this simple dataset: https://www.kaggle.com/luddarell/101-simple-linear-regressioncsv)

- Run this command in Terminal, where you provide the **path to your dataset** and the **path to the yaml file**

.. code-block:: console

    $ igel fit --data_path 'path_to_your_csv_dataset' --model_definition_file 'path_to_your_yaml_file'


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
