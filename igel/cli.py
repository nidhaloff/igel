"""Console script for igel."""
import sys
import argparse
from igel import Igel


class CLI(object):
    """CLI describes a command line interface for interacting with igel, there
    are several different functions that can be performed. These functions are:

    - fit - fits a model on the input file specified to it
    - predict - Given a list of $hat{y}$ values, compute $d(\\hat{y}, y) under a
      specified metric

    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='igel cli runner',
            usage='''igel <command> [<args>]
Available sub-commands:
   fit                 fits a model
   predict             Predicts using a pre-fitted model
''')
        parser.add_argument('command', help='Subcommand to run')
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        print("command here is : ", args.command)
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def fit(self):
        Igel()


def main():
    CLI()


if __name__ == "__main__":
    main()
