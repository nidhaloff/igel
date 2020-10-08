=====
Usage
=====


you can run this command to get instruction on how to use the model:

.. code-block:: console

    $ igel --help

    # or just

    $ igel -h
    """
    Take some time and read the output of help command. You ll save time later if you understand how to use igel.
    """


First step is to provide a yaml file. You can do this manually by creating a .yaml file and editing it yourself.
However, if you are lazy (and you probably are, like me :D), you can use the igel init command to get started fast:

.. code-block:: console

    """
    igel init <args>
    possible optional args are: (notice that these args are optional, so you can also just run igel init if you want)
    -type: regression or classification
    -model: model you want to use
    -target: target you want to predict


    Example:
    If I want to use neural networks to classify whether someone is sick or not using the indian-diabetes dataset,
    then I would use this command to initliaze a yaml file:
    $ igel init -type "classification" -model "NeuralNetwork" -target "sick"
    """
    $ igel init

After running the command, an igel.yaml file will be created for you in the current working directory. You can
check it out and modify it if you want to, otherwise you can also create everything from scratch.


.. code-block:: yaml

        # model definition
        model:
            # in the type field, you can write the type of problem you want to solve. Whether regression or classification
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

You can then evaluate the trained/pre-fitted model:

.. code-block:: console

    $ igel evaluate -dp 'path_to_your_evaluation_dataset.csv'
    """
    This will automatically generate an evaluation.json file in the current directory, where all evaluation results are stored
    """

Finally, you can use the trained/pre-fitted model to make predictions if you are happy with the evaluation results:

.. code-block:: console

    $ igel predict -dp 'path_to_your_test_dataset.csv'
    """
    This will generate a predictions.csv file in your current directory, where all predictions are stored in a csv file
    """

You can combine the train, evaluate and predict phases using one single command called experiment:

.. code-block:: console

    $ igel experiment -DP "path_to_train_data path_to_eval_data path_to_test_data" -yml "path_to_yaml_file"

    """
    This will run fit using train_data, evaluate using eval_data and further generate predictions using the test_data
    """

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
        read_data_options: default
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
        type: classification    # type of the problem you want to solve. | possible values: [regression, classification]
        algorithm: NeuralNetwork    # which algorithm you want to use. | type igel algorithms in the Terminal to know more
        arguments: default          # model arguments: you can check the available arguments for each model by running igel help in your terminal

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

You can also carry out some preprocessing methods or other operations by providing the it in the yaml file.
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

Finally, the multioutput-example contains a **multioutput regression** example.

I suggest you play around with the examples and igel cli. However,
you can also directly execute the fit.py, evaluate.py and predict.py if you want to.
