#!/usr/bin/env python3
import csv
import json
import os
from datetime import datetime, date

import yfinance as yf

from TimeSeriesPrediction.features import Features
from TimeSeriesPrediction.model import Commons
from Tools.parse_args import Parse_Args


def append_csv_log(csv_path, header, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def build_log_path(base_log, model_type, profile_type):
    base_dir = os.path.dirname(base_log)
    filename = f"{model_type}_{profile_type}.csv"
    return os.path.join(base_dir, filename)


@Parse_Args.parser("Train ML model.")
@Parse_Args.filename
def main(filename):

    with open(filename) as f:
        cfg = json.load(f)

    today = date.today()
    stocks = Features.flatten_stocklist([cfg["stocks"]])

    for m in cfg["models"]:
        model = Commons.load_from_file(m["model_path"])
        if model is None:
            print(f"⚠️  Could not load model {m['model_path']}, skipping.")
            continue
        profiles = m.get("profiles", [])
        log_dir = m.get("log_dir", "log/")

        for prof in profiles:
            log_csv = build_log_path(
                log_dir, model.model_type, prof.get("profile_type", "default")
            )
            header = [
                "DateTime",
                "Stock",
                "Pred_On",
                "Current_Price",
                "Predicted",
                "Purchased",
                "Close_Price",
            ]

            for stock in stocks:
                parsed = Features.parse_name(stock)
                if not parsed:
                    continue
                raw_name, start, end, period = parsed
                raw_df = Features.get_raw_stock(raw_name, period, start, end)
                if raw_df is None or raw_df.empty:
                    continue

                last_date = raw_df.index[-1].date()
                if last_date != today:
                    continue

                # feature‐engineering & predict
                feat_df = model.features.get_stocks(raw_df, raw_stocks=raw_df.copy())
                date_pred, predicted = model.predict(feat_df)

                if date_pred.date() != today:
                    print(f" ⚠️  prediction date mismatch for {raw_name}, skipping.")
                    continue

                pred_on = raw_df["Open"][-1]

                # get current market price via yfinance
                ticker = yf.Ticker(raw_name)
                intraday = ticker.history(period="1d")
                if intraday.empty:
                    current_price = ""
                else:
                    current_price = intraday["Close"][-1]

                row = {
                    "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Stock": raw_name,
                    "Pred_On": pred_on,
                    "Current_Price": current_price,
                    "Predicted": predicted,
                    "Purchased": True,
                    "Close_Price": "",
                }
                append_csv_log(log_csv, header, row)
                print(f"[{raw_name}] logged to {log_csv}")


if __name__ == "__main__":
    main()
