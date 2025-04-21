import traceback

from TimeSeriesPrediction.data import Data
from TimeSeriesPrediction.model import Commons
from Tools.parse_args import Parse_Args


# Train Args:  python3 Train.py Models/test.spm Linear -t 1 -o -s StockList/train.csv


@Parse_Args.parser("Train ML model.")
@Parse_Args.filename
@Parse_Args.split
@Parse_Args.overwrite
@Parse_Args.modeltype
@Parse_Args.debug
@Parse_Args.seed
@Parse_Args.save_xlsx
@Parse_Args.stocks()
def main(filename, debug, split, overwrite, mtype, stocks, save, seed):
    model = None

    if not overwrite:
        model = Commons.load_from_file(filename, if_exists=True)

    if model is None:
        model = Commons.model_mapping[mtype]()

    stocks = model.features.flatten_stocklist(stocks)
    model.set_seed(seed)
    test_sets = []
    for stockName in stocks:
        try:
            print(f"Training: {stockName}")

            df = model.features.get_stocks_parse(stockName)

            # Split data,
            train_df, test_df = Data.train_test_split(df, split)

            model.train(train_df)

            model.training_stock.append(stockName)
            if split != 1:
                test_sets.append((stockName, test_df.copy()))
        except Exception as e:
            print("Failed on: ", stockName, e)
            traceback.print_tb(e.__traceback__)

    model.save_model(filename)
    print(f"Successfully saved model at: {filename}")

    if split != 1 and len(test_sets) > 0:
        print("\n\nFinished Training, now testing")
        for stock, test_df in test_sets:
            print(f"Testing: {stock}")
            test_df["pred_value"] = model.batch_predict(test_df)

            print(
                model.calculate_metrics(
                    test_df,
                    reduced=(not debug),
                    print_table=debug,
                    save_table=f"{stock}-{save}",
                )
            )


if __name__ == "__main__":
    main()
