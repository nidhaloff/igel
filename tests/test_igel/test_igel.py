#!/usr/bin/env python

"""Tests for `igel` package."""

import os

import pytest
from igel import Igel

from .constants import Constants
from .helper import remove_folder
from .mock import MockCliArgs

os.chdir(os.path.dirname(__file__))


@pytest.fixture
def mock_args():
    yield MockCliArgs
    remove_folder(Constants.model_results_dir)
    assert Constants.model_results_dir.exists() == False


def test_fit(mock_args):
    """
    test the fit model functionality
    """
    assert mock_args is not None
    Igel(**mock_args.fit)
    assert Constants.model_results_dir.exists() == True
    assert Constants.description_file.exists() == True
    assert Constants.evaluation_file.exists() == False

def test_export(mock_args):
    """
    test the export model functionality
    """
    assert mock_args is not None  
    Igel(**mock_args.fit)  
    Igel(**mock_args.export)
    assert Constants.onnx_model_file.exists() == True
