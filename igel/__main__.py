import random

import igel
import typer
from igel import Igel, metrics_dict, models_dict

app = typer.Typer()


@app.command()
def init(
    model_type: str = typer.Option(
        ...,
        help="type of the machine learning problem you want to solve",
    ),
    model_name: str = typer.Option(
        ...,
        help="name of the model you want to use. Run the igel models command to get a list of all supported models",
    ),
    target: str = typer.Option(
        ..., help="target you want to predict in your dataset"
    ),
):
    Igel.create_init_mock_file(
        model_type=model_type, model_name=model_name, target=target
    )


@app.command()
def version():
    """show igel's installed version"""
    typer.secho(f"igel version: {igel.__version__}")


@app.command()
def info():
    """show metadata about igel"""
    typer.echo(
        f"""
        package name:           igel
        version:                {igel.__version__}
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
    """
    )
