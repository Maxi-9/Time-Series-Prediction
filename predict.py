from TimeSeriesPrediction.model import Commons
from Tools.parse_args import Parse_Args


@Parse_Args.parser("Predict values using trained ML model.")
@Parse_Args.filename
@Parse_Args.seed
@Parse_Args.stocks(multiple=False)
def main(filename, stocks, seed):
    if ".csv" in stocks:
        raise ValueError("Please select a single stock not csv file")

    model = Commons.load_from_file(filename)

    model.set_seed(seed)
    df = model.features.get_stocks_parse(stocks)

    # Predict closing value
    pred_date, pred_close = model.predict(df)

    print(f"Predicted closing value({pred_date}) for {stocks}: {pred_close:.2f}")


if __name__ == "__main__":
    main()
