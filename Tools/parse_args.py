import click

from TimeSeriesPrediction.model import Commons


class Parse_Args:
    def __init__(self):
        pass

    filename = click.argument(
        "filename",
        nargs=1,
        type=click.Path(exists=False),
    )
    modeltype = click.argument(
        "mtype",
        nargs=1,
        type=click.Choice(Commons.model_mapping.keys()),
    )

    save_xlsx = click.option(
        "-x",
        "--save",
        type=str,
        default=None,
        help="Debug: Saves test table to specified file as xlsx file, used for debugging and testing.",
    )
    debug = click.option(
        "-d",
        "--debug",
        is_flag=True,
        help="Prints more info: prints debug table and more metrics.",
    )

    overwrite = click.option(
        "-o",
        "--overwrite",
        is_flag=True,
        help="Overwrites (if exists) else trains pre-existing model.",
    )

    # May create a helper in the future, run_all.py has inbuilt caching
    # cache = click.option(
    #     "-c",
    #     "--cache",
    #     type=str,
    #     default="cache",
    #     help="Creates a cache file for each stock in provided directory, add the rows that don't already exist, while adding missing data. Can be used with the -s or --stocks option as a file path.",
    # )

    # load = click.option(
    #     "-l",
    #     "--load",
    #     type=str,
    #     default="cache",
    #     help="Creates a cache file for each stock in provided directory, add the rows that don't already exist, while adding missing data. Can be used with the -s or --stocks option as a file path.",
    # )

    @staticmethod
    def stocks(default=None, multiple=True):
        if default is None:
            default = []
        req = default is None
        return click.option(
            "-s",
            "--stocks",
            multiple=multiple,
            type=str,
            default=default,
            required=req,
            help="Select stocks for use."
            " Stock options: stock (e.g. AAPL), stock.market (e.g. AAPL.NASDAQ),"
            " stock.market:period (e.g. AAPL:1mo), stock.market:start.stop (e.g. AAPL:01-01-2020.12-31-2023),"
            " stock.market:start (e.g. AAPL:01-01-2020), or file path to CSV."
            " Period options: 1d (1 day), 1mo (1 month), 1y (1 year), ytd (year-to-date), max (maximum available data).",
        )

    seed = click.option(
        "-e",
        "--seed",
        type=int,
        default=None,
        help="""\b
    Random seed if not specified. Set fixed seed for supported models, setting seed will make the model deterministic but the input data from yFinance isn't deterministic.\n""",
    )  # Warning: May not work for all models, if you create your own model, you customize it to set seed
    # Due to a problem with Yahoo this is not guaranteed to work https://github.com/ranaroussi/yfinance/issues/626

    split = click.option(
        "-t",
        "--split",
        type=float,
        default=0.8,
        help="Splits training and test data. Higher value means more training data (Input a float between 0 and 1).",
    )

    @staticmethod
    def parser(help_text: str):
        return click.command(help=help_text)
