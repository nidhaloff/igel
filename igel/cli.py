"""Console script for igel."""
import sys
import argparse
from igel import Igel, models_dict, metrics_dict, __version__
import pandas as pd


class CLI(object):
    """CLI describes a command line interface for interacting with igel, there
    are several different functions that can be performed.

    """

    available_args = {
        # fit, evaluate and predict args:
        "dp": "data_path",
        "yml": "yaml_path",
        "DP": "data_paths",

        # models arguments
        "name": "model_name",
        "model": "model_name",
        "type": "model_type",
        "tg": "target"
    }

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Igel CLI Runner',
            usage=f'''
 ___           _    ____ _     ___
|_ _|__ _  ___| |  / ___| |   |_ _|
 | |/ _` |/ _ \ | | |   | |    | |
 | | (_| |  __/ | | |___| |___ | |
|___\__, |\___|_|  \____|_____|___|
    |___/


igel <command> [<args>]
- Available sub-commands at the moment are:
   init              initialize a yaml file with default parameters
   fit               fits a model
   evaluate          evaluate the performance of a pre-fitted model
   predict           predicts using a pre-fitted model
   experiment        this command will run fit, evaluate and predict together
   help              get help about how to use igel
   models            get a list of supported machine learning algorithms/models
   metrics           get a list of all supported metrics

- Available arguments:

    # for usage with the fit, evaluate or predict command:
    --data_path         Path to your dataset
    --yaml_file         Path to your yaml file

    # for usage with the experiment command:
    --data_paths        Paths to data you want to use for fitting,
                        evaluating and predict respectively.

    --yaml_file         Path to the yaml file that will be used
                        when fitting the model.

    # for getting help with the models command:
    --model_type        type of the model you want to get help on
                        -> whether regression, classification or clustering.

    --model_name        name of the model you want to get help on.
    ------------------------------------------

    or for the short version
    # for usage with the fit, evaluate or predict command:
    -dp         Path to your dataset
    -yml        Path to your yaml file

    # for usage with the experiment command:
    -DP         Paths to data you want to use for fitting,
                evaluating and predict respectively.


    -yml        Path to the yaml file that will be used when fitting the model.

    # for getting help with the models command:
    -type       type of the model you want to get help on
                -> whether regression, classification or clustering.

    -name       name of the model you want to get help on.
    ------------------------------------------

- Quick Start:

igel -h
# print this help guide

igel init -type regression -model RandomForest
# automatically create a yaml file in the working directory
# with some default parameters to get you started fast

igel models
# type this to get a list of supported models

igel models -type regression -name RandomForest
# this will give you a help on how to use the
# RandomForestRegressor and will provide
# you a link to get more help in the sklearn website

igel metrics
# get a list of all supported metrics

igel fit -dp "path_to_data" -yml "path_to_yaml_file"
# fit a model

igel evaluate -dp "path_to_data"
# evaluate the trained/pre-fitted model

igel predict -dp "path_to_data"
# make predictions using the trained/pre-fitted model

igel experiment -DP "path_to_train_data \\
                    path_to_evaluation_data \\
                    path_to_data_you_want_to_predict_on"
                -yml "path_to_yaml_file"
# this will run fit using the trian data first,
# then will run evaluate using the evaluation data
# and finally, will generate predictions on the last
# data you passed to predict on.
                    ''')

        self.parser.add_argument('command', help='Subcommand to run')
        self.cmd = self.parse_command()
        self.args = sys.argv[2:]
        self.dict_args = self.convert_args_to_dict()
        getattr(self, self.cmd.command)()

    def validate_args(self, dict_args: dict) -> dict:
        """
        validate arguments entered by the user and transform short args to the representation needed by igel
        @param dict_args: dict of arguments
        @return: new validated and transformed args

        """
        d_args = {}
        for k, v in dict_args.items():
            if k not in self.available_args.keys() and k not in self.available_args.values():
                print(f'Unrecognized argument -> {k}')
                self.parser.print_help()
                exit(1)

            elif k in self.available_args.values():
                d_args[k] = v

            else:
                d_args[self.available_args[k]] = v

        return d_args

    def convert_args_to_dict(self) -> dict:
        """
        convert args list to a dictionary
        @return: args as dictionary
        """

        dict_args = {self.args[i].replace('-', ''): self.args[i + 1] for i in range(0, len(self.args) - 1, 2)}
        dict_args = self.validate_args(dict_args)
        dict_args['cmd'] = self.cmd.command
        return dict_args

    def parse_command(self):
        """
        parse command, which represents the function that will be called by igel
        @return: command entered by the user
        """
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        cmd = self.parser.parse_args(sys.argv[1:2])
        if not hasattr(self, cmd.command):
            print('Unrecognized command')
            self.parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        return cmd

    def help(self, *args, **kwargs):
        self.parser.print_help()

    def init(self, *args, **kwargs):
        """
        initialize a dummy/default yaml file as a starting point. The user can provide args directly in the terminal
        usage:
            igel init <args>

        if not args are provided, the user will be prompted to enter basic information.
        """
        d = dict(self.dict_args)
        d.pop('cmd')
        if not d:
            print(f""
                  f"{'*' * 10} You entered interactive mode! {'*' * 10} \n"
                  f"This is happening because you didn't enter all mandatory arguments in order to use the cli\n"
                  f"Therefore, you will need to provide few information before proceeding.\n")
            model_type = input(f"enter type of the problem you want to solve: [regression]       ") or "regression"
            d['model_type'] = model_type
            model_name = input(f"enter algorithm you want to use: [NeuralNetwork]        ") or "NeuralNetwork"
            d['model_name'] = model_name
            target = input(f"enter the target you want to predict  "
                           "(this is usually a column name in your csv dataset):        ")
            d['target'] = target

        Igel.create_init_mock_file(**d)

    def _accept_user_input(self, yaml_needed: bool = False,
                           default_data_path: str = './train_data.csv',
                           default_yaml_path: str = './igel.yaml'):
        """
        accept user input if the user did not provide all mandatory args in the terminal.
        """
        print(f""
              f"{'*'*10} You entered interactive mode! {'*'*10} \n"
              f"This is happening because you didn't enter all mandatory arguments in order to use the cli\n"
              f"Therefore, you will need to provide few information before proceeding.\n")
        data_path = input(f"enter path to your data: [{default_data_path}]        ") or default_data_path
        self.dict_args['data_path'] = data_path
        if yaml_needed:
            yaml_path = input(f"enter path to your yaml file: [{default_yaml_path}]        ") or default_yaml_path
            self.dict_args['yaml_path'] = yaml_path

    def fit(self, *args, **kwargs):
        print("""
         _____          _       _
        |_   _| __ __ _(_)_ __ (_)_ __   __ _
          | || '__/ _` | | '_ \| | '_ \ / _` |
          | || | | (_| | | | | | | | | | (_| |
          |_||_|  \__,_|_|_| |_|_|_| |_|\__, |
                                        |___/
        """)
        d = dict(self.dict_args)
        d.pop('cmd')
        if not d:
            self._accept_user_input(yaml_needed=True)

        Igel(**self.dict_args)

    def predict(self, *args, **kwargs):
        print("""
         ____               _ _      _   _
        |  _ \ _ __ ___  __| (_) ___| |_(_) ___  _ __
        | |_) | '__/ _ \/ _` | |/ __| __| |/ _ \| '_ \
        |  __/| | |  __/ (_| | | (__| |_| | (_) | | | |
        |_|   |_|  \___|\__,_|_|\___|\__|_|\___/|_| |_|


        """)
        d = dict(self.dict_args)
        d.pop('cmd')
        if not d:
            self._accept_user_input()
        Igel(**self.dict_args)

    def evaluate(self, *args, **kwargs):
        print("""
         _____            _             _   _
        | ____|_   ____ _| |_   _  __ _| |_(_) ___  _ __
        |  _| \ \ / / _` | | | | |/ _` | __| |/ _ \| '_ \
        | |___ \ V / (_| | | |_| | (_| | |_| | (_) | | | |
        |_____| \_/ \__,_|_|\__,_|\__,_|\__|_|\___/|_| |_|

        """)
        d = dict(self.dict_args)
        d.pop('cmd')
        if not d:
            self._accept_user_input()

        Igel(**self.dict_args)

    def _print_models_overview(self):
        print(f"\nIgel's supported models overview: \n")
        reg_algs = list(models_dict.get('regression').keys())
        clf_algs = list(models_dict.get('classification').keys())
        cluster_algs = list(models_dict.get('clustering').keys())
        df_algs = pd.DataFrame.from_dict({
            "regression": reg_algs,
            "classification": clf_algs,
            "clustering": cluster_algs
        }, orient="index").transpose().fillna('----')

        df = self._tableize(df_algs)
        print(df)

    def _show_model_infos(self, model_name: str, model_type: str):
        if not model_name:
            print(f"Please enter a supported model")
            self._print_models_overview()
        else:
            if not model_type:
                print(f"Please enter a type argument to get help on the chosen model\n"
                      f"type can be whether regression, classification or clustering \n")
                self._print_models_overview()
                return
            if model_type not in ('regression', 'classification', 'clustering'):
                raise Exception(f"{model_type} is not supported! \n"
                                f"model_type need to be regression, classification or clustering")

            models = models_dict.get(model_type)
            model_data = models.get(model_name)
            model, link, *cv_class = model_data.values()
            print(f"model type: {model_type} \n"
                  f"model name: {model_name} \n"
                  f"sklearn model class: {model.__name__} \n"

                  f"{'-' * 60}\n"
                  f"You can click the link below to know more about the optional arguments\n"
                  f"that you can use with your chosen model ({model_name}).\n"
                  f"You can provide these optional arguments in the yaml file if you want to use them.\n"
                  f"link:\n{link} \n")

    def models(self):
        """
        show an overview of all models supported by igel
        """
        if not self.dict_args or len(self.dict_args.keys()) <= 1:
            self._print_models_overview()
            print("-"*100)
            model_name = input("Enter the model name, you want to get infos about (e.g NeuralNetwork):    ")
            model_type = input("Enter the type (choose from regression, classification or clustering):   ")
            if model_name and model_type:
                self._show_model_infos(model_name, model_type)
        else:
            model_name = self.dict_args.get('model_name', None)
            model_type = self.dict_args.get('model_type', None)
            self._show_model_infos(model_name, model_type)

    def metrics(self):
        """
        show an overview of all metrics supported by igel
        """
        print(f"\nIgel's supported metrics overview: \n")
        reg_metrics = [func.__name__ for func in metrics_dict.get('regression')]
        clf_metrics = [func.__name__ for func in metrics_dict.get('classification')]

        df_metrics = pd.DataFrame.from_dict({
            "regression": reg_metrics,
            "classification": clf_metrics
        }, orient="index").transpose().fillna('----')

        df_metrics = self._tableize(df_metrics)
        print(df_metrics)

    def experiment(self):
        """
        run a whole experiment: this is a pipeline that includes fit, evaluate and predict.
        """
        print("""
         _____                      _                      _
        | ____|_  ___ __   ___ _ __(_)_ __ ___   ___ _ __ | |_
        |  _| \ \/ / '_ \ / _ \ '__| | '_ ` _ \ / _ \ '_ \| __|
        | |___ >  <| |_) |  __/ |  | | | | | | |  __/ | | | |_
        |_____/_/\_\ .__/ \___|_|  |_|_| |_| |_|\___|_| |_|\__|
                   |_|

        """)
        d = dict(self.dict_args)
        d.pop('cmd')
        if not d:
            default_train_data_path = './train_data.csv'
            default_eval_data_path = './eval_data.csv'
            default_test_data_path = './test_data.csv'
            default_yaml_path = './igel.yaml'
            print(f""
                  f"{'*' * 10} You entered interactive mode! {'*' * 10} \n"
                  f"This is happening because you didn't enter all mandatory arguments in order to use the cli\n"
                  f"Therefore, you will need to provide few information before proceeding.\n")
            train_data_path = input(
                f"enter path to your data: [{default_train_data_path}]        ") or default_train_data_path
            eval_data_path = input(
                f"enter path to your data: [{default_eval_data_path}]        ") or default_eval_data_path
            test_data_path = input(
                f"enter path to your data: [{default_test_data_path}]        ") or default_test_data_path
            yaml_path = input(
                f"enter path to your yaml file: [{default_yaml_path}]        ") or default_yaml_path

            # prepare the dict arguments:
            train_args = {"cmd": "fit",
                          "yaml_path": yaml_path,
                          "data_path": train_data_path}
            eval_args = {"cmd": "evaluate",
                         "data_path": eval_data_path}
            pred_args = {"cmd": "predict",
                         "data_path": test_data_path}

        else:
            data_paths = self.dict_args['data_paths']
            yaml_path = self.dict_args['yaml_path']
            train_data_path, eval_data_path, pred_data_path = data_paths.strip().split(' ')
            # print(f"{train_data_path} | {eval_data_path} | {test_data_path}")
            train_args = {"cmd": "fit",
                          "yaml_path": yaml_path,
                          "data_path": train_data_path}
            eval_args = {"cmd": "evaluate",
                         "data_path": eval_data_path}
            pred_args = {"cmd": "predict",
                         "data_path": pred_data_path}

        Igel(**train_args)
        Igel(**eval_args)
        Igel(**pred_args)

    def _tableize(self, df):
        """
        pretty-print a dataframe as table
        """
        if not isinstance(df, pd.DataFrame):
            return
        df_columns = df.columns.tolist()
        max_len_in_lst = lambda lst: len(sorted(lst, reverse=True, key=len)[0])
        align_center = lambda st, sz: "{0}{1}{0}".format(" "*(1+(sz-len(st))//2), st)[:sz] if len(st) < sz else st
        align_right = lambda st, sz: "{0}{1} ".format(" "*(sz-len(st)-1), st) if len(st) < sz else st
        max_col_len = max_len_in_lst(df_columns)
        max_val_len_for_col = dict([(col, max_len_in_lst(df.iloc[:,idx].astype('str'))) for idx, col in enumerate(df_columns)])
        col_sizes = dict([(col, 2 + max(max_val_len_for_col.get(col, 0), max_col_len)) for col in df_columns])
        build_hline = lambda row: '+'.join(['-' * col_sizes[col] for col in row]).join(['+', '+'])
        build_data = lambda row, align: "|".join([align(str(val), col_sizes[df_columns[idx]]) for idx, val in enumerate(row)]).join(['|', '|'])
        hline = build_hline(df_columns)
        out = [hline, build_data(df_columns, align_center), hline]
        for _, row in df.iterrows():
            out.append(build_data(row.tolist(), align_right))
        out.append(hline)
        return "\n".join(out)

    def version(self):
        print(f"igel version: {__version__}")

    def info(self):
        print(f"""
            package name:           igel
            version:                {__version__}
            author:                 Nidhal Baccouri
            maintainer:             Nidhal Baccouri
            contact:                nidhalbacc@gmail.com
            license:                MIT
            description:            use machine learning without writing code
            dependencies:           pandas, sklearn, pyyaml
            requires python:        >= 3.6
            First release:          27.08.2020
            official repo:          https://github.com/nidhaloff/igel
            written in:             100% python
            status:                 stable
            operating system:       independent
        """)


def main():
    CLI()


if __name__ == "__main__":
    main()
