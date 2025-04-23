#!/usr/bin/env python3
import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, date, timedelta

import pandas as pd
import yfinance as yf

from TimeSeriesPrediction.features import Features
from TimeSeriesPrediction.model import Commons
from Tools.parse_args import Parse_Args

CACHE_PATH = "log/close_cache.json"


def append_csv_log(csv_path, header, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def build_log_path(base_log, model_type):
    base_dir = os.path.dirname(base_log)
    filename = f"{model_type}_log.csv"
    return os.path.join(base_dir, filename)


def log_close_fill(raw_cache: dict[str, pd.DataFrame], log_dir: str = "log/"):
    today = date.today()
    for fname in os.listdir(log_dir):
        if not fname.endswith(".csv"):
            continue

        path = os.path.join(log_dir, fname)
        df = pd.read_csv(path, parse_dates=["Date"])

        missing = df["Close_Price"].isna() | (
            df["Close_Price"].astype(str).str.strip() == ""
        )

        if not missing.any():
            continue

        for idx in df[missing].index:
            row = df.loc[idx]
            row_date = row["Date"].date()
            if row_date == today:
                continue
            ticker = row["Stock"]
            # Use raw_cache for historical close data
            close_val = None
            if ticker in raw_cache:
                ticker_df = raw_cache[ticker]
                # Find the close value for the correct date
                match = ticker_df[ticker_df.index.date == row_date]
                if not match.empty and "Close" in match.columns:
                    close_val = match["Close"].iloc[0]
            if close_val is None:
                print(
                    f"  ⚠️  Could not fetch close for {ticker} on {row_date} from cache."
                )
                continue
            df.at[idx, "Close_Price"] = close_val
            print(
                f"[{fname}] Filled Close_Price for {ticker} on {row_date}: {close_val}"
            )

        df.to_csv(path, index=False)


def process_one(task):
    raw_name, raw_df, model_path, model_type, log_dir, now = task
    model = Commons.load_from_file(model_path)
    if model is None:
        print(f"⚠️  Could not load model at {model_path}")
        return None

    today = date.today()
    try:
        feat_df = model.features.get_stocks(None, raw_stocks=raw_df.copy())
        date_pred, predicted = model.predict(feat_df)
        if date_pred.date() != today:
            return None

        pred_on = raw_df["Open"].iloc[-1]
        intraday = yf.Ticker(raw_name).history(period="1d")
        current_price = intraday["Close"].iloc[-1] if not intraday.empty else ""
        should_buy = model.features.predict_on.shouldBuy(
            predicted_value=predicted, current_values=feat_df.iloc[-1]
        )

        row = {
            "Date": now.strftime("%Y-%m-%d"),
            "Time": now.strftime("%H:%M:%S"),
            "Stock": raw_name,
            "Pred_On": pred_on,
            "Current_Price": current_price,
            "Predicted": predicted,
            "Purchased": should_buy,
            "Close_Price": "",
        }
        return build_log_path(log_dir, model_type), row

    except Exception as e:
        print(f"Error processing {raw_name} with model {model_type}: {e}")
        return None


@Parse_Args.parser("Train ML model.")
@Parse_Args.filename
def main(filename):
    with open(filename) as f:
        cfg = json.load(f)

    today = date.today()
    now = datetime.now()

    # Build fetch list for raw data
    stocks = Features.flatten_stocklist([cfg["stocks"]])
    fetch_list = []
    for stock in stocks:
        parsed = Features.parse_name(stock)
        if not parsed:
            continue
        raw_name, start, end, period = parsed
        if start is None:
            start = today - timedelta(days=730)

        # end = today + timedelta(days=1)
        fetch_list.append((raw_name, period, start, end))

    # Batch‐download all raw historical frames
    raw_cache = Features.get_batch_raw_stocks(fetch_list)
    stock_count = len(raw_cache)
    raw_cache = {
        tick: df
        for tick, df in raw_cache.items()
        if not df.empty and df.index[-1].date() == today
    }

    if len(raw_cache) == 0:
        if stock_count != 0:
            print(
                f"All stocks were dropped, {stock_count - len(raw_cache)} were dropped"
            )
        print("No data to process, exiting...")
        return

    print(
        f"Processing {len(raw_cache)} stocks, {stock_count - len(raw_cache)} were dropped..."
    )

    # Build the list of work tasks
    tasks = []
    for m in cfg["models"]:
        model_path = m["model_path"]
        model = Commons.load_from_file(model_path, if_exists=True)
        if model is None:
            print(f"⚠️  Could not load model {model_path}, skipping.")
            continue
        model_type = model.model_type
        log_dir = m.get("log_dir", "log/")
        for raw_name, raw_df in raw_cache.items():
            tasks.append((raw_name, raw_df, model_path, model_type, log_dir, now))

    print(
        f"Processing {len(tasks)} tasks, for {len(raw_cache)} stocks over {len(cfg['models'])} models."
    )

    # Spin up workers and process all stocks × models in parallel
    rows_by_csv = defaultdict(list)
    with ProcessPoolExecutor() as executor:
        for result in executor.map(process_one, tasks):
            if result is None:
                continue
            log_csv, row = result
            print(f"Processed {log_csv}")
            rows_by_csv[log_csv].append(row)

    print(f"Finished Processing at {datetime.now()}")

    # Write each CSV once, in a batch
    for log_csv, rows in rows_by_csv.items():
        os.makedirs(os.path.dirname(log_csv), exist_ok=True)
        df_rows = pd.DataFrame(rows)
        mode = "a" if os.path.exists(log_csv) else "w"
        df_rows.to_csv(log_csv, mode=mode, header=(mode == "w"), index=False)
        print(f"{'Wrote' if mode == 'w' else 'Appended'} {len(rows)} rows to {log_csv}")

    print("Finished Logging, now checking for missing data...")

    # Finally, fill in any missing historical Close_Price
    log_close_fill(raw_cache)


if __name__ == "__main__":
    main()
