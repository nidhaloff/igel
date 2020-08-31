=====
Usage
=====


- First step is to provide a yaml file:

.. code-block:: yaml

        # model definition
        model:
            type: regression
            algorithm: random forest

        # target you want to predict
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

