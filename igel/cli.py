"""Console script for igel."""
import sys
import argparse
from igel import IgelModel


class CLI(object):
    """CLI describes a command line interface for interacting with igel, there
    are several different functions that can be performed. These functions are:

    - fit - fits a model on the input file specified to it
    - predict - Given a list of $hat{y}$ values, compute $d(\\hat{y}, y) under a
      specified metric

    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='igel cli runner',
            usage='''igel <command> [<args>]
                    Available sub-commands:
                       fit                 fits a model
                       predict             Predicts using a pre-fitted model
''')
        self.parser.add_argument('command', help='Subcommand to run')
        self.cmd = self.parse_command()
        self.args = sys.argv[2:]
        self.dict_args = self.convert_args_to_dict()
        getattr(self, self.cmd.command)()

    def convert_args_to_dict(self):
        dict_args = {self.args[i].replace('-', ''): self.args[i + 1] for i in range(0, len(self.args) - 1, 2)}
        return dict_args

    def parse_command(self):
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        cmd = self.parser.parse_args(sys.argv[1:2])
        if not hasattr(self, cmd.command):
            print('Unrecognized command')
            self.parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        return cmd

    def fit(self):
        IgelModel(self.cmd.command, **self.dict_args).fit()

    def predict(self):
        IgelModel(self.cmd.command, **self.dict_args).predict()


def main():
    CLI()


if __name__ == "__main__":
    main()
