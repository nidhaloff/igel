====
igel
====

.. image:: assets/logo.jpg
    :width: 100%
    :scale: 100%
    :align: center
    :alt: igel-icon


|

|

.. image:: https://img.shields.io/pypi/v/igel?color=green
        :alt: PyPI
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

.. image:: https://img.shields.io/twitter/url?url=https%3A%2F%2Ftwitter.com%2FNidhalBaccouri
        :alt: Twitter URL
        :target: https://twitter.com/NidhalBaccouri

|
|

A machine learning tool that allows you to train/fit, test and use models **without writing code**


* Free software: MIT license
* Documentation: https://igel.readthedocs.io.

.. contents:: Table of Contents
    :depth: 3

|
|

Motivation & Goal
------------------

The goal of the project is to provide machine learning for **everyone**, both technical and non technical
users.

I needed a tool sometimes, which I can use to fast create a machine learning prototype. Whether to build
some proof of concept or create a fast draft model to prove a point. I find myself often stuck at writing
boilerplate code and/or thinking too much of how to start this.

Therefore, I decided to create **igel**. Hopefully, it will make it easier for technical and non technical
users to build machine learning models.

Features
---------
- Supports all state of the art machine learning models (even preview models)
- Supports different data preprocessing methods
- Provides flexibility and data control while writing configurations
- Supports cross validation
- Supports yaml and json format
- Supports different sklearn metrics for regression, classification and clustering
- Supports multi-output/multi-target regression and classification
- Supports multi-processing for parallel model construction

Intro
--------

igel is built on top of scikit-learn. It provides a simple way to use machine learning without writing
a **single line of code**

All you need is a **yaml** (or **json**) file, where you need to describe what you are trying to do. That's it!

Igel supports all sklearn's machine learning functionality, whether regression, classification or clustering.

Installation
-------------

- The easiest way is to install igel using `pip <https://packaging.python.org/guides/tool-recommendations/>`_

.. code-block:: console

    $ pip install -U igel

- Check the docs for other ways to install igel from source

Models
-------

Igel's supported models:

.. code-block:: console

        +--------------------+----------------------------+-------------------------+
        |      regression    |        classification      |        clustering       |
        +--------------------+----------------------------+-------------------------+
        |   LinearRegression |         LogisticRegression |                  KMeans |
        |              Lasso |                      Ridge |     AffinityPropagation |
        |          LassoLars |               DecisionTree |                   Birch |
        | BayesianRegression |                  ExtraTree | AgglomerativeClustering |
        |    HuberRegression |               RandomForest |    FeatureAgglomeration |
        |              Ridge |                 ExtraTrees |                  DBSCAN |
        |  PoissonRegression |                        SVM |         MiniBatchKMeans |
        |      ARDRegression |                  LinearSVM |    SpectralBiclustering |
        |  TweedieRegression |                      NuSVM |    SpectralCoclustering |
        | TheilSenRegression |            NearestNeighbor |      SpectralClustering |
        |    GammaRegression |              NeuralNetwork |               MeanShift |
        |   RANSACRegression | PassiveAgressiveClassifier |                  OPTICS |
        |       DecisionTree |                 Perceptron |                    ---- |
        |          ExtraTree |               BernoulliRBM |                    ---- |
        |       RandomForest |           BoltzmannMachine |                    ---- |
        |         ExtraTrees |       CalibratedClassifier |                    ---- |
        |                SVM |                   Adaboost |                    ---- |
        |          LinearSVM |                    Bagging |                    ---- |
        |              NuSVM |           GradientBoosting |                    ---- |
        |    NearestNeighbor |        BernoulliNaiveBayes |                    ---- |
        |      NeuralNetwork |      CategoricalNaiveBayes |                    ---- |
        |         ElasticNet |       ComplementNaiveBayes |                    ---- |
        |       BernoulliRBM |         GaussianNaiveBayes |                    ---- |
        |   BoltzmannMachine |      MultinomialNaiveBayes |                    ---- |
        |           Adaboost |                       ---- |                    ---- |
        |            Bagging |                       ---- |                    ---- |
        |   GradientBoosting |                       ---- |                    ---- |
        +--------------------+----------------------------+-------------------------+

Quick Start
------------

you can run this command to get instruction on how to use the model:

.. code-block:: console

    $ igel --help

    # or just

    $ igel -h
    """
    Take some time and read the output of help command. You ll save time later if you understand how to use igel.
    """

- Demo:

.. image:: assets/igel-help.gif

---------------------------------------------------------------------------------------------------------

First step is to provide a yaml file (you can also use json if you want). You can do this manually by creating a .yaml file and editing it yourself.
However, if you are lazy (and you probably are, like me :D), you can use the igel init command to get started fast:




.. code-block:: console

    """
    igel init <args>
    possible optional args are: (notice that these args are optional, so you can also just run igel init if you want)
    -type: regression, classification or clustering
    -model: model you want to use
    -target: target you want to predict


    Example:
    If I want to use neural networks to classify whether someone is sick or not using the indian-diabetes dataset,
    then I would use this command to initliaze a yaml file:
    $ igel init -type "classification" -model "NeuralNetwork" -target "sick"
    """
    $ igel init

After runnig the command, an igel.yaml file will be created for you in the current working directory. You can
check it out and modify it if you want to, otherwise you can also create everything from scratch.

- Demo:

.. image:: assets/igel-init.gif

-----------------------------------------------------------------------------------------------------------

.. code-block:: yaml

        # model definition
        model:
            # in the type field, you can write the type of problem you want to solve. Whether regression, classification or clustering
            # Then, provide the algorithm you want to use on the data. Here I'm using the random forest algorithm
            type: classification
            algorithm: RandomForest     # make sure you write the name of the algorithm in pascal case
            arguments:
                n_estimators: 100   # here, I set the number of estimators (or trees) to 100
                max_depth: 30       # set the max_depth of the tree

        # target you want to predict
        # Here, as an example, I'm using the famous indians-diabetes dataset, where I want to predict whether someone have diabetes or not.
        # Depending on your data, you need to provide the target(s) you want to predict here
        target:
            - sick

In the example above, I'm using random forest to classify whether someone have
diabetes or not depending on some features in the dataset
I used this `indian-diabetes dataset <https://www.kaggle.com/uciml/pima-indians-diabetes-database>`_)


- The expected way to use igel is from terminal (igel CLI):

Run this command in terminal to fit/train a model, where you provide the **path to your dataset** and the **path to the yaml file**

.. code-block:: console

    $ igel fit --data_path 'path_to_your_csv_dataset.csv' --yaml_file 'path_to_your_yaml_file.yaml'

    # or shorter

    $ igel fit -dp 'path_to_your_csv_dataset.csv' -yml 'path_to_your_yaml_file.yaml'

    """
    That's it. Your "trained" model can be now found in the model_results folder
    (automatically created for you in your current working directory).
    Furthermore, a description can be found in the description.json file inside the model_results folder.
    """

- Demo:

.. image:: assets/igel-fit.gif

--------------------------------------------------------------------------------------------------------

You can then evaluate the trained/pre-fitted model:

.. code-block:: console

    $ igel evaluate -dp 'path_to_your_evaluation_dataset.csv'
    """
    This will automatically generate an evaluation.json file in the current directory, where all evaluation results are stored
    """

- Demo:

.. image:: assets/igel-eval.gif

------------------------------------------------------------------------------------------------------


Finally, you can use the trained/pre-fitted model to make predictions if you are happy with the evaluation results:

.. code-block:: console

    $ igel predict -dp 'path_to_your_test_dataset.csv'
    """
    This will generate a predictions.csv file in your current directory, where all predictions are stored in a csv file
    """

- Demo:

.. image:: assets/igel-pred.gif

.. image:: assets/igel-predict.gif

----------------------------------------------------------------------------------------------------------


You can combine the train, evaluate and predict phases using one single command called experiment:

.. code-block:: console

    $ igel experiment -DP "path_to_train_data path_to_eval_data path_to_test_data" -yml "path_to_yaml_file"

    """
    This will run fit using train_data, evaluate using eval_data and further generate predictions using the test_data
    """

- Demo:

.. image:: assets/igel-experiment.gif


- Alternatively, you can also write code if you want to:

..  code-block:: python

    from igel import Igel

    # provide the arguments in a dictionary
    params = {
            'cmd': 'fit',    # provide the command you want to use. whether fit, evaluate or predict
            'data_path': 'path_to_your_dataset',
            'yaml_path': 'path_to_your_yaml_file'
    }

    Igel(**params)
    """
    check the examples folder for more
    """

Interactive Mode
------------------

Interactive mode is new in >= v0.2.6

This mode basically offers you the freedom to write arguments on your way.
You are not restricted to write the arguments directly when using the command.

This means practically that you can use the commands (fit, evaluate, predict, experiment etc..)
without specifying any additional arguments. For example:

..  code-block:: python

    igel fit

if you just write this and click enter, you will be prompted to provide the additional mandatory arguments.
Any version <= 0.2.5 will throw an error in this case, which why you need to make sure that you have
a >= 0.2.6 version.

- Demo (init command):

.. image:: assets/igel-init-interactive.gif

- Demo (fit command):

.. image:: assets/igel-fit-interactive.gif

As you can see, you don't need to memorize the arguments, you can just let igel ask you to enter them.
Igel will provide you with a nice message explaining which argument you need to enter.

The value between brackets represents the default value. This means if you provide no value and hit return,
then the value between brackets will be taken as the default value.

Overview
----------
The main goal of igel is to provide you with a way to train/fit, evaluate and use models without writing code.
Instead, all you need is to provide/describe what you want to do in a simple yaml file.

Basically, you provide description or rather configurations in the yaml file as key value pairs.
Here is an overview of all supported configurations (for now):

.. code-block:: yaml

    # dataset operations
    dataset:
        type: csv
        read_data_options: # options you want to supply for reading your data
            sep:  # Delimiter to use.
            delimiter:  # Alias for sep.
            header:     # Row number(s) to use as the column names, and the start of the data.
            names:  # List of column names to use
            index_col: # Column(s) to use as the row labels of the DataFrame,
            usecols:    # Return a subset of the columns
            squeeze:    # If the parsed data only contains one column then return a Series.
            prefix:     # Prefix to add to column numbers when no header, e.g. ‘X’ for X0, X1, …
            mangle_dupe_cols:   # Duplicate columns will be specified as ‘X’, ‘X.1’, …’X.N’, rather than ‘X’…’X’. Passing in False will cause data to be overwritten if there are duplicate names in the columns.
            dtype:  # Data type for data or columns
            engine:     # Parser engine to use. The C engine is faster while the python engine is currently more feature-complete.
            converters: # Dict of functions for converting values in certain columns. Keys can either be integers or column labels.
            true_values: # Values to consider as True.
            false_values: # Values to consider as False.
            skipinitialspace: # Skip spaces after delimiter.
            skiprows: # Line numbers to skip (0-indexed) or number of lines to skip (int) at the start of the file.
            skipfooter: # Number of lines at bottom of file to skip
            nrows: # Number of rows of file to read. Useful for reading pieces of large files.
            na_values: # Additional strings to recognize as NA/NaN.
            keep_default_na: # Whether or not to include the default NaN values when parsing the data.
            na_filter: # Detect missing value markers (empty strings and the value of na_values). In data without any NAs, passing na_filter=False can improve the performance of reading a large file.
            verbose: # Indicate number of NA values placed in non-numeric columns.
            skip_blank_lines: # If True, skip over blank lines rather than interpreting as NaN values.
            parse_dates: # try parsing the dates
            infer_datetime_format: # If True and parse_dates is enabled, pandas will attempt to infer the format of the datetime strings in the columns, and if it can be inferred, switch to a faster method of parsing them.
            keep_date_col: # If True and parse_dates specifies combining multiple columns then keep the original columns.
            dayfirst: # DD/MM format dates, international and European format.
            cache_dates: # If True, use a cache of unique, converted dates to apply the datetime conversion.
            thousands: # the thousands operator
            decimal: # Character to recognize as decimal point (e.g. use ‘,’ for European data).
            lineterminator: # Character to break file into lines.
            escapechar: # One-character string used to escape other characters.
            comment: # Indicates remainder of line should not be parsed. If found at the beginning of a line, the line will be ignored altogether. This parameter must be a single character.
            encoding: # Encoding to use for UTF when reading/writing (ex. ‘utf-8’).
            dialect: # If provided, this parameter will override values (default or not) for the following parameters: delimiter, doublequote, escapechar, skipinitialspace, quotechar, and quoting
            delim_whitespace: # Specifies whether or not whitespace (e.g. ' ' or '    ') will be used as the sep
            low_memory: # Internally process the file in chunks, resulting in lower memory use while parsing, but possibly mixed type inference.
            memory_map: # If a filepath is provided for filepath_or_buffer, map the file object directly onto memory and access the data directly from there. Using this option can improve performance because there is no longer any I/O overhead.


        split:  # split options
            test_size: 0.2  # 0.2 means 20% for the test data, so 80% are automatically for training
            shuffle: True   # whether to shuffle the data before/while splitting
            stratify: None  # If not None, data is split in a stratified fashion, using this as the class labels.

        preprocess: # preprocessing options
            missing_values: mean    # other possible values: [drop, median, most_frequent, constant] check the docs for more
            encoding:
                type: oneHotEncoding  # other possible values: [labelEncoding]
            scale:  # scaling options
                method: standard    # standardization will scale values to have a 0 mean and 1 standard deviation  | you can also try minmax
                target: inputs  # scale inputs. | other possible values: [outputs, all] # if you choose all then all values in the dataset will be scaled


    # model definition
    model:
        type: classification    # type of the problem you want to solve. | possible values: [regression, classification, clustering]
        algorithm: NeuralNetwork    # which algorithm you want to use. | type igel algorithms in the Terminal to know more
        arguments: default          # model arguments: you can check the available arguments for each model by running igel help in your terminal
        use_cv_estimator: false     # if this is true, the CV class of the specific model will be used if it is supported
        cross_validate:
            cv: # number of kfold (default 5)
            n_jobs:   # The number of CPUs to use to do the computation (default None)
            verbose: # The verbosity level. (default 0)

    # target you want to predict
    target:
        - put the target you want to predict here
        - you can assign many target if you are making a multioutput prediction


E2E Example
-----------

A complete end to end solution is provided in this section to prove the capabilities of **igel**.
As explained previously, you need to create a yaml configuration file. Here is an end to end example for
predicting whether someone have diabetes or not using the **decision tree** algorithm. The dataset can be found in the examples folder.

-  **Fit/Train a model**:

.. code-block:: yaml

        model:
            type: classification
            algorithm: DecisionTree

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

Advanced Usage
---------------

You can also carry out some preprocessing methods or other operations by providing them in the yaml file.
Here is an example, where the data is split to 80% for training and 20% for validation/testing.
Also, the data are shuffled while splitting.

Furthermore, the data are preprocessed by replacing missing values with the mean ( you can also use median, mode etc..).
check `this link <https://www.kaggle.com/uciml/pima-indians-diabetes-database>`_ for more information


.. code-block:: yaml

        # dataset operations
        dataset:
            split:
                test_size: 0.2
                shuffle: True
                stratify: default

            preprocess: # preprocessing options
                missing_values: mean    # other possible values: [drop, median, most_frequent, constant] check the docs for more
                encoding:
                    type: oneHotEncoding  # other possible values: [labelEncoding]
                scale:  # scaling options
                    method: standard    # standardization will scale values to have a 0 mean and 1 standard deviation  | you can also try minmax
                    target: inputs  # scale inputs. | other possible values: [outputs, all] # if you choose all then all values in the dataset will be scaled

        # model definition
        model:
            type: classification
            algorithm: RandomForest
            arguments:
                # notice that this is the available args for the random forest model. check different available args for all supported models by running igel help
                n_estimators: 100
                max_depth: 20

        # target you want to predict
        target:
            - sick

Then, you can fit the model by running the igel command as shown in the other examples

.. code-block:: console

    $ igel fit -dp path_to_the_dataset -yml path_to_the_yaml_file

For evaluation

.. code-block:: console

    $ igel evaluate -dp path_to_the_evaluation_dataset

For production

.. code-block:: console

    $ igel predict -dp path_to_the_new_dataset

Examples
----------

In the examples folder in the repository, you will find a data folder,where the famous indian-diabetes, iris dataset
and the linnerud (from sklearn) datasets are stored.
Furthermore, there are end to end examples inside each folder, where there are scripts and yaml files that
will help you get started.


The indian-diabetes-example folder contains two examples to help you get started:

- The first example is using a **neural network**, where the configurations are stored in the neural-network.yaml file
- The second example is using a **random forest**, where the configurations are stored in the random-forest.yaml file

The iris-example folder contains a **logistic regression** example, where some preprocessing (one hot encoding)
is conducted on the target column to show you more the capabilities of igel.

Furthermore, the multioutput-example contains a **multioutput regression** example.
Finally, the cv-example contains an example using the Ridge classifier using cross validation.

I suggest you play around with the examples and igel cli. However,
you can also directly execute the fit.py, evaluate.py and predict.py if you want to.

Links
------

- Article: https://medium.com/@nidhalbacc/machine-learning-without-writing-code-984b238dd890

Contributions
--------------

You think this project is useful and you want to bring new ideas, new features, bug fixes, extend the docs?

Contributions are always welcome.
Make sure you read `the guidelines <https://igel.readthedocs.io/en/latest/contributing.html>`_ first

License
--------

MIT license

Copyright (c) 2020-present, Nidhal Baccouri
