import igel
import typer

app = typer.Typer()


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
