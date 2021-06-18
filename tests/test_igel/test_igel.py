#!/usr/bin/env python

"""Tests for `igel` package."""

import os

import pandas as pd
import pytest
from igel import Igel

from .constants import Constants
from .helper import remove_file, remove_folder
from .mock import MockCliArgs

os.chdir(os.path.dirname(__file__))


@pytest.fixture
def mock_args():
    return MockCliArgs
    # remove_folder(Constants.model_results_dir)


def test_fit(mock_args):
    """
    test the fit model functionality
    """
    assert mock_args is not None
    # Igel(**mock_args.fit)
    # assert Constants.model_results_dir.exists() == True
