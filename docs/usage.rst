=====
Usage
=====

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


E2E
----


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

