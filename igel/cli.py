"""Console script for igel."""
import sys
import argparse
from igel import IgelModel, models_dict, metrics_dict
import inspect


class CLI(object):
    """CLI describes a command line interface for interacting with igel, there
    are several different functions that can be performed.

    """

    available_args = {
        # fit, evaluate and predict args:
        "dp": "data_path",
        "yml": "yaml_path",

        # models arguments
        "name": "model_name",
        "type": "model_type"
    }

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Igel CLI Runner',
            usage='''

                    igel <command> [<args>]
                    - Available sub-commands at the moment are:
                       fit                 fits a model
                       evaluate            evaluate the performance of a pre-fitted model
                       predict             Predicts using a pre-fitted model
                       help                get help about how to use igel
                       algorithms          get a list of supported machine learning algorithms

                    - Available arguments:
                        --data_path         Path to your dataset
                        --yaml_file         Path to your yaml file
                        ------------------------------------------
                        or for a short version
                        -dp                 Path to your dataset
                        -yml                Path to your yaml file
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

    def fit(self, *args, **kwargs):
        IgelModel(self.cmd.command, **self.dict_args).fit()

    def predict(self, *args, **kwargs):
        IgelModel(self.cmd.command, **self.dict_args).predict()

    def evaluate(self, *args, **kwargs):
        IgelModel(self.cmd.command, **self.dict_args).evaluate()

    def print_models_overview(self):
        print(f"\n\n"
              f"{'*' * 60}  Supported machine learning algorithms  {'*' * 60} \n\n"
              f"1 - Regression algorithms: \n"
              f"{'-' * 50} \n"
              f"{list(models_dict.get('regression').keys())} \n\n"
              f"{'=' * 120} \n"
              f"2 - Classification algorithms: \n"
              f"{'-' * 50} \n"
              f"{list(models_dict.get('classification').keys())} \n"
              f" \n")

    def models(self):
        if not self.dict_args:
            self.print_models_overview()
        else:
            print("models args: ", self.dict_args)
            model_name = self.dict_args.get('model_name', None)
            model_type = self.dict_args.get('model_type', None)

            if not model_name or not model_type:
                print(f"Please enter a supported model")
                self.print_models_overview()
            else:
                if model_type not in ('regression', 'classification'):
                    raise Exception(f"{model_type} is not supported! \n"
                                    f"model_type need to be regression or classification")

                models: dict = models_dict.get(model_type)
                model = models.get(model_name.replace('_', ' '))
                print(f"model type: {model_type} | model_name: {model_name} | sklearn class: {model.__name__} \n"
                      f"inspect: {inspect.getfullargspec(model.__init__)} \n"
                      f"model: {model.__init__.__code__.co_varnames}")

    def metrics(self):
        print(f"\n\n"
              f"{'*' * 60}  Supported metrics  {'*' * 60} \n\n"
              f"1 - Regression metrics: \n"
              f"{'-' * 50} \n"
              f"{[ func.__name__ for func in metrics_dict.get('regression')]} \n\n"
              f"{'=' * 120} \n"
              f"2 - Classification metrics: \n"
              f"{'-' * 50} \n"
              f"{[ func.__name__ for func in metrics_dict.get('classification')]} \n"
              f" \n")


def main():
    CLI()


if __name__ == "__main__":
    main()
