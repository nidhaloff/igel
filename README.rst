====
igel
====

.. image:: assets/logo1.png
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
            type: classification
            algorithm: random forest

        # target you want to predict
        # Here, as an example, I'm using the famous indians-diabetes dataset, where I want to predict whether someone have diabetes or not.
        # Depending on your data, you need to provide the target(s) you want to predict here
        target:
            - sick

In the example above, I'm using random forest to classify whether someone have
diabetes or not depending on some features in the dataset
I used this `indian-diabetes dataset <https://www.kaggle.com/uciml/pima-indians-diabetes-database>`_ )

`
- Run this command in Terminal, where you provide the **path to your dataset** and the **path to the yaml file**

.. code-block:: console

    $ igel fit --data_path 'path_to_your_csv_dataset.csv' --yaml_file 'path_to_your_yaml_file.yaml'

    # or shorter

    $ igel fit -dp 'path_to_your_csv_dataset.csv' -yml 'path_to_your_yaml_file.yaml'


you can run this command to get instruction on how to use the model:

.. code-block:: console

    $ igel --help

    # or just

    $ igel -h


That's it. Your "trained" model can be now found in the model_results folder
(automatically created for you in your current working directory).
Furthermore, a description can be found in the description.json file inside the model_results folder.

E2E Example
-----------

A complete end to end solution is provided in this section to prove the capabilities of **igel**.
As explained previously, you need to create a yaml configuration file. Here is an end to end example for
predicting whether someone have diabetes or not using the **decision tree** algorithm. The dataset can be found in the examples folder.

-  **Fit/Train a model**:

.. code-block:: yaml

        model:
            type: classification
            algorithm: decision tree

        target:
            - sick

.. code-block:: console

    $ igel fit -dp path_to_the_dataset -yml path_to_the_yaml_file

That's it, igel will now fit the model for you and save it in a model_results folder in your current directory.


- **Evaluate the model**:

Evaluate the pre-fitted model. Igel will load the pre-fitted model from the model_results directory and evaluate it for you.
You just need to run the evaluate command and provide the path to your evaluation data.

.. code-block:: console

    $ igel evaluate -dp path_to_the_evaluation_dataset

That's it! Igel will evaluate the model and store statistics/results in an **evaluation.json** file inside the model_results folder

- **Predict**:

Use the pre-fitted model to predict on new data. This is done automatically by igel, you just need to provide the
path to your data that you want to use prediction on.

.. code-block:: console

    $ igel predict -dp path_to_the_new_dataset

That's it! Igel will use the pre-fitted model to make predictions and save it in a **predictions.csv** file inside the model_results folder


Examples
----------

Check the examples folder, where you will find the indian-diabetes data and a yaml file example

TODO
-----
- add option as arguments to the models
- add multiple file support
- provide an api to evaluate models

Contributions
--------------

Contributions are always welcome.
Make sure you read `the guidelines <https://igel.readthedocs.io/en/latest/contributing.html>`_ first
