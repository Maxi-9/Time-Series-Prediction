#!/usr/bin/env python3
import csv
import json
import os
import sys
from datetime import datetime, time

import backtrader as bt

# Import your libraries for feature engineering and models.
from TimeSeriesPrediction.features import Features
from TimeSeriesPrediction.model import Commons
from Tools.parse_args import Parse_Args


def append_csv_log(csv_path, header, row):
    """Append a row to the CSV log file; create header if not already present."""
    try:
        file_exists = os.path.exists(csv_path)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"Error logging to CSV file {csv_path}: {e}")


class ModelBasedStrategy(bt.Strategy):
    params = (
        ("model", None),  # Loaded prediction model (expects a predict(df) method)
        (
            "profile",
            None,
        ),  # Dictionary specifying trading method settings for this stock
        ("log_csv", ""),  # Path to CSV log file for this model
        ("buy_time", time(7, 40)),  # Configurable buy time (e.g. 9:40 AM)
        ("sell_time", time(13, 50)),  # Configurable sell time (e.g. 3:50 PM)
        ("seed", None),  # Optional seed for reproducibility
    )

    def __init__(self):
        self.order = None
        self.open_order_placed = False
        self.close_order_placed = False
        # Cache the first open value for reference if needed.
        self.initial_data = self.data.open[0]

    def next(self):
        # Retrieve the current date/time from the data feed.
        current_datetime = self.data.datetime.datetime(0)
        current_time = current_datetime.time()

        # --- BUY EXECUTION AT OR AFTER buy_time ---
        if (not self.open_order_placed) and current_time >= self.p.buy_time:
            try:
                # Prepare data for prediction.
                # For demonstration, we build a minimal pandas DataFrame using the current open price.
                # In your full implementation, you may want to collect a proper slice with all needed columns.
                pred_on = self.data.open[0]
                import pandas as pd

                df = pd.DataFrame(
                    {
                        "Open": [pred_on],
                        # Add additional columns required by your model here…
                    }
                )

                # Run the prediction using your model.
                date_pred, predicted = self.p.model.predict(df)

                # Check that the predicted date matches today.
                today_str = current_datetime.strftime("%Y-%m-%d")
                if date_pred.strftime("%Y-%m-%d") != today_str:
                    raise ValueError(
                        f"Predicted date {date_pred.strftime('%Y-%m-%d')} does not match today's date {today_str}"
                    )

                # Log prediction result to CSV.
                csv_header = ["Date", "Stock", "Pred_On", "Predicted"]
                log_row = {
                    "Date": current_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                    "Stock": self.data._name,
                    "Pred_On": pred_on,
                    "Predicted": predicted,
                }
                append_csv_log(self.p.log_csv, csv_header, log_row)
                print(f"[{self.data._name}] Prediction: {pred_on} -> {predicted}")

                # --- Place Buy Order According to Trading Profile ---
                buy_method = self.p.profile.get("buy_method", "market")
                if buy_method == "market":
                    self.order = self.buy()  # Market order.
                elif buy_method == "limit":
                    # Use limit order (price taken from profile or default to current price).
                    limit_price = self.p.profile.get("limit_price", pred_on)
                    self.order = self.buy(price=limit_price, exectype=bt.Order.Limit)
                elif buy_method == "stop":
                    # Use stop order.
                    stop_price = self.p.profile.get("stop_price", pred_on)
                    self.order = self.buy(price=stop_price, exectype=bt.Order.Stop)
                elif buy_method == "stop-limit":
                    # Use stop-limit order; requires both stop and limit prices.
                    stop_price = self.p.profile.get("stop_price", pred_on)
                    limit_price = self.p.profile.get("limit_price", pred_on)
                    self.order = self.buy(
                        price=limit_price, exectype=bt.Order.StopLimit
                    )
                else:
                    # If the profile is unrecognized, default to market.
                    self.order = self.buy()

                self.open_order_placed = True

            except Exception as e:
                print(f"Error processing open order for {self.data._name}: {e}")

        # --- SELL EXECUTION AT OR AFTER sell_time ---
        if (
            self.open_order_placed
            and (not self.close_order_placed)
            and current_time >= self.p.sell_time
        ):
            try:
                # For this example, we use a market order to exit.
                self.order = self.sell()
                self.close_order_placed = True
            except Exception as e:
                print(f"Error processing sell order for {self.data._name}: {e}")

    def notify_order(self, order):
        # Callback to check order status and print execution info.
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"[{self.data._name}] Buy executed at {order.executed.price}")
            elif order.issell():
                print(f"[{self.data._name}] Sell executed at {order.executed.price}")


@Parse_Args.parser("Trading Bot with Model-based orders and backtrader integration.")
@Parse_Args.filename
@Parse_Args.seed
def main(filename, seed):
    today_date = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load daily configuration from the provided config file
    with open(filename, "r") as config_file:
        config = json.load(config_file)

    models_configs = config.get("models", [])
    stocks_raw = config.get("stocks", None)
    if not models_configs:
        print("No models found in the configuration.")
        sys.exit(1)

    # Process the stock list from config (flattened if needed)
    all_stocks = Features.flatten_stocklist([stocks_raw])

    # Step 1: For each model, read its log CSV (if it exists) and collect stocks already processed today
    model_predicted = {}  # key: log CSV path, value: set of stocks predicted today
    for model_cfg in models_configs:
        csv_log_path = model_cfg.get("log_csv")
        predicted = set()
        if os.path.exists(csv_log_path):
            try:
                with open(csv_log_path, "r") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if row.get("Date") == today_date:
                            predicted.add(row.get("Stock"))
            except Exception as e:
                print(f"Error reading log file {csv_log_path}: {e}")
        model_predicted[csv_log_path] = predicted

    # Step 2: Determine missing stocks per model and create the union of missing stocks
    model_missing = {}  # key: log CSV path, value: set of missing stocks for that model
    union_missing = set()
    for model_cfg in models_configs:
        csv_log_path = model_cfg.get("log_csv")
        missing = set(all_stocks) - model_predicted.get(csv_log_path, set())
        model_missing[csv_log_path] = missing
        union_missing |= missing

    if not union_missing:
        print("All stocks have already been processed for all models for today.")
        sys.exit(0)

    # Step 3: Gather raw stock data only for stocks in the union of missing stocks.
    raw_dat = {}
    for stock in union_missing:
        try:
            # Parse stock name and associated parameters
            parsed = Features.parse_name(stock)
            if not parsed:
                print(f"Failed to parse stock {stock}")
                continue
            raw_stock_name, start_date, end_date, period = parsed

            dat = Features.get_raw_stock(raw_stock_name, period, start_date, end_date)
            if dat is None or dat.empty:
                print(f"No data found for stock: {raw_stock_name}")
                continue

            # Verify that the latest date in the data matches today’s date
            data_date = dat.index[-1].strftime("%Y-%m-%d")
            if data_date != today_date:
                print(
                    f"Last date {data_date} does not match today's date {today_date} for {raw_stock_name}"
                )
                continue

            raw_dat[stock] = dat
        except Exception as e:
            print(f"Error processing stock {stock}: {e}")

    if not raw_dat:
        print("No test stocks found with up-to-date data.")
        sys.exit(1)

    # CSV header for log file remains the same
    csv_header = ["Date", "Stock", "Pred_On", "Predicted"]

    # Step 4: Process each model only for its missing stocks.
    for model_cfg in models_configs:
        csv_log_path = model_cfg.get("log_csv")
        missing_stocks = model_missing.get(csv_log_path, set())
        if not missing_stocks:
            print(
                f"All stocks already processed for model {model_cfg.get('model_type')}. Skipping."
            )
            continue

        print(
            f"Processing model type: {model_cfg.get('model_type')} from {model_cfg.get('model_path')}"
        )
        model = Commons.load_from_file(model_cfg.get("model_path"))
        if model is None:
            print(f"Model file not found: {model_cfg.get('model_path')}")
            continue

        # Set global seed for reproducibility if provided
        if seed:
            model.set_seed(seed)

        # Load and prepare stock data only for missing stocks for this model
        model_stock_dat = {}
        for stock in missing_stocks:
            if stock not in raw_dat:
                print(f"Stock {stock} not available in raw data. Skipping.")
                continue
            try:
                # Use the model's own features method to extract relevant data
                df = model.features.get_stocks(
                    raw_dat[stock], raw_stocks=raw_dat[stock].copy(deep=True)
                )
                if df is None or df.empty:
                    print(
                        f"No data found for stock: {stock} in model {model_cfg.get('model_type')}"
                    )
                    continue
                model_stock_dat[stock] = df
            except Exception as e:
                print(f"Error loading data for stock {stock}: {e}")

        # Process predictions for each missing stock for this model
        for stock, df in model_stock_dat.items():
            try:
                # Get the value on which prediction is based (e.g., the 'Open' column value)
                pred_on = None
                cols = model.features.predict_on.cols(prev_cols=True)
                if cols and cols[0] in df.columns:
                    pred_on = df[cols[0]].iloc[-1]

                # Run prediction
                date, predicted = model.predict(df)
                if date.strftime("%Y-%m-%d") != today_date:
                    raise ValueError(
                        f"Predicted date {date.strftime('%Y-%m-%d')} does not match today's date {today_date}"
                    )

                row = {
                    "Date": now_time,
                    "Stock": stock,
                    "Pred_On": pred_on,
                    "Predicted": predicted,
                }
                append_csv_log(csv_log_path, csv_header, row)
                print(
                    f"{stock}: {pred_on} -> {predicted} | {date.strftime('%Y-%m-%d')}"
                )
            except Exception as err:
                print(f"Error processing stock {stock}: {err}")


if __name__ == "__main__":
    main()
