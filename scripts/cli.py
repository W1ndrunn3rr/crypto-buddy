import click
from scripts.predict import predict
from scripts.train import train
from scripts.get_data import get_data


@click.group()
def cli():
    """Cryptocurrency analysis toolkit"""
    pass


@cli.command()
@click.argument("currency", default="bitcoin")
@click.argument("forecast_days", default="7")
def cli_predict(currency, forecast_days):
    """Run predictions for given cryptocurrency"""
    click.echo(f"Running prediction for: {currency}")
    predict(currency, forecast_days)


@cli.command()
@click.argument("currency", default="bitcoin")
def cli_train(currency):
    """Train model for given cryptocurrency"""
    click.echo(f"Training model for: {currency}")
    train(currency)


@cli.command()
@click.argument("currency", default="bitcoin")
def cli_get_data(currency):
    """Fetch data for given cryptocurrency"""
    click.echo(f"Fetching data for: {currency}")
    get_data(currency)


if __name__ == "__main__":
    cli()
