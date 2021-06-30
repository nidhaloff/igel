import igel
import typer
from igel import Igel, metrics_dict, models_dict

app = typer.Typer()


def get_model_types() -> list:
    return ["regression", "classification"]


def get_model_names() -> list:
    return models_dict.keys()


@app.command()
def init(
    model_type: str = typer.Argument(
        default=get_model_types,
        help="type of the machine learning problem you want to solve : regression, classification or clustering",
    ),
    model_name: str = typer.Argument(
        default=get_model_names, help="name of the model you want to use"
    ),
    target: str = typer.Argument(
        help="target you want to predict in your dataset"
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
