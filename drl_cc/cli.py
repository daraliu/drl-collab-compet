import pathlib

import click
from unityagents import UnityEnvironment

from drl_cc import tennis


HERE = pathlib.Path(__file__).absolute().parent
DEFAULT_PATH_TO_TENNIS_ENV = HERE.parent.joinpath("Tennis_Linux/Tennis.x86_64")


@click.group()
def cli():
    pass


@cli.command(
    "demo-tennis",
    help="Run a demo of Tennis agents - trained or random (if no model provided)")
@click.argument(
    "DIR_MODEL",
    required=False,
    type=click.Path(dir_okay=False, file_okay=True, exists=True))
@click.option(
    "--unity-tennis-env", "-e",
    type=click.Path(dir_okay=False, file_okay=True, exists=True),
    default=DEFAULT_PATH_TO_TENNIS_ENV,
    help=f"Path to Unity Tennis Environment executable, default: {DEFAULT_PATH_TO_TENNIS_ENV}")
def demo(dir_model, unity_tennis_env):
    env = UnityEnvironment(file_name=str(unity_tennis_env))
    if dir_model is None:
        click.echo("Using Random agent")
    else:
        click.echo(f"Loading trained agent model from {dir_model.absolute()}")
    score = tennis.demo(env, dir_model)
    click.echo(
        f"Episode completed with {'random' if dir_model is None else 'trained'} agent. "
        f"Score: {score:2f}")
    env.close()


