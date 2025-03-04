import importlib
import importlib.util
import os
import sys
from datetime import datetime

import pandas as pd
import yfinance as yf
from overrides import overrides
from pandas import DataFrame


class Predictable:
    def __init__(
        self,
        sOpen=True,
        sClose=True,
        sHigh=True,
        sLow=True,
        sVolume=True,
        sDividends=True,
        sStock_Splits=True,
    ):
        self.Open = sOpen
        self.Close = sClose
        self.High = sHigh
        self.Low = sLow
        self.Volume = sVolume
        self.Dividends = sDividends
        self.Stock_Splits = sStock_Splits


class BaseFeature:
    def __init__(
        self,
        columns=None,
        is_sensitive=True,
        uses_data=True,
        base_sensitive=None,
        normalize=True,
        is_number: bool = True,
    ):
        """
        Inheritable class to allow for complex analysis on data, or import external data as features. While trying to prevent any data leaking
        :param columns: Represents which columns this feature represents (or creates)
        :param is_sensitive: Shifts the feature forward by 1 day (needs to be true, if it uses_data)
        :param uses_data: Calculates the feature based on data (passes windowed data instead of index)
        :param base_sensitive: Provide a Predictable object to be able to predict on this feature(and provide _calculate)
        """

        assert (
            is_sensitive or base_sensitive is None
        )  # Require sensitive if can_predict
        assert is_sensitive or not uses_data  # Require sensitive if uses_data
        assert columns is not None
        assert base_sensitive is None or len(columns) == 1

        self.columns = columns

        self.normalize = normalize
        self.is_sensitive = is_sensitive
        self.uses_data = uses_data
        self.base_sensitive = base_sensitive
        self.is_number = is_number

    def __eq__(self, other):
        return self.__class__ == other.__class__

    def __hash__(self):
        return hash(self.__class__)

    def cols(self, prev_cols=False):
        if self.is_sensitive and prev_cols:
            return ["prev_" + col for col in self.columns]
        return self.columns

    def true_col(self):
        if self.is_sensitive:
            return "true_" + self.columns[0]
        raise ValueError("This feature is not sensitive so it doesn't have a true col.")

    def calculate(self, df: pd.DataFrame, window=None) -> pd.DataFrame:
        """
        Calculates the feature with either windowed data or just the index.

        :param df: The main stock data
        :param window: The window size for calculations
        :return: New DataFrame with calculated feature
        """
        if self.uses_data:
            if window is None or window >= df.shape[0]:
                window = 40

            results = []
            for i in range(window, df.shape[0]):
                start = max(0, i - window)
                end = i
                window_data = df.iloc[start:end]
                result = self._calculate(window_data)
                if result is pd.DataFrame:
                    result = result.squeeze()
                results.append(result)

            results_df = pd.DataFrame(
                results, index=df.iloc[window:].index, columns=self.columns
            )

            return results_df
        else:
            calc_result = self._calculate(df.index)
            results_df = pd.DataFrame(calc_result, index=df.index, columns=self.columns)
            return results_df

    def _calculate(self, df: pd.DataFrame) -> dict[str, float]:
        raise NotImplementedError(
            "_calculate method must be implemented in subclasses."
        )

    def calcBuyDays(self, filt_data, predicted_values):
        if self.base_sensitive is None:
            raise ValueError("This feature cannot be used for prediction.")

        buy_signals = []
        for i in range(len(filt_data)):
            current_value = filt_data.iloc[i]
            predicted_value = predicted_values[i]
            buy_signal = self._calc_buy_signal(current_value, predicted_value)
            buy_signals.append(buy_signal)

        return buy_signals

    def _calc_buy_signal(self, current_value, predicted_value):
        # Implement your buy/sell signal logic here
        raise NotImplementedError(
            "_calc_buy_signal method must be implemented in subclasses."
        )

    def price_diff(self, df):
        # Implement your price difference logic here
        raise NotImplementedError(
            "price_diff method must be implemented in subclasses."
        )

    def prediction_col(self):
        return "pred_value"


# Base Stock Features
class Open(BaseFeature):
    def __init__(self):
        super().__init__(
            columns=["Open"],
            is_sensitive=True,
            uses_data=True,
            base_sensitive=Predictable(),
        )

    @overrides
    def _calculate(self, df: pd.DataFrame) -> dict[str, float]:
        pass  # Not implementing because base feature

    @overrides
    def _calc_buy_signal(self, current_value, predicted_value):
        if predicted_value > current_value["prev_open"]:
            return True
        else:
            return False

    @overrides
    def price_diff(self, df):
        # Assuming that each feature only contains a single column
        open_column = Features.Open.true_col()
        true_close_column = Features.Close.true_col()
        # Calculate the percentage of change between true close and true open
        change = (df[true_close_column] - df[open_column]) / df[open_column]
        return change + 1


class Close(BaseFeature):
    def __init__(self):
        super().__init__(
            columns=["Close"],
            is_sensitive=True,
            uses_data=True,
            base_sensitive=Predictable(sOpen=False),
        )

    @overrides
    def _calculate(self, df: pd.DataFrame) -> dict[str, float]:
        pass  # Not implementing because base feature

    @overrides
    def _calc_buy_signal(self, current_value, predicted_value):
        if predicted_value > current_value[list(Features.Open.cols())[0]]:
            return True
        else:
            return False

    @overrides
    def price_diff(self, df):
        # Assuming that each feature only contains a single column
        open_column = list(Features.Open.cols())[0]
        true_close_column = Features.Close.true_col()
        # Calculate the percentage of change between open - previous close
        change = (df[true_close_column] - df[open_column]) / df[open_column]
        return change + 1


class Increased(BaseFeature):
    def __init__(self):
        # Use "Increased" as the feature name, marking it as a sensitive feature
        super().__init__(
            columns=["Increased"],
            is_sensitive=True,
            uses_data=True,
            base_sensitive=Predictable(sOpen=True, sClose=True),
            normalize=False,
            is_number=True,
        )

    def _calculate(self, df: pd.DataFrame) -> dict[str, float]:
        # Compare open to close within the same day
        open_column = Features.Open.cols()[0]  # Get the Open column
        close_column = Features.Close.cols()[0]  # Get the Close column

        if open_column not in df or close_column not in df:
            raise ValueError(
                f"Missing required Open/Close columns: {open_column} and {close_column}"
            )

        df = df.copy()  # Prevent SettingWithCopyWarning
        # Calculate the increase flag for the last row only
        increased = int(df[close_column].iloc[-1] > df[open_column].iloc[-1])
        return {"Increased": increased}

    def _calc_buy_signal(self, current_value, predicted_value):
        # Buy if the predicted probability is above 0.5
        return 1 if predicted_value > 0.5 else 0

    @overrides
    def price_diff(self, df):
        open_column = Features.Open.cols()[0]  # Get the Open column
        close_column = Features.Close.cols()[0]  # Get the Close column

        # Calculate the percentage change from Open to Close
        change = (df[close_column] - df[open_column]) / df[open_column]
        return change + 1


class High(BaseFeature):
    def __init__(self):
        super().__init__(
            columns=["High"],
            is_sensitive=True,
            uses_data=True,
        )

    @overrides
    def _calculate(self, df: pd.DataFrame) -> dict[str, float]:
        pass  # Not implementing because base feature


class Low(BaseFeature):
    def __init__(self):
        super().__init__(
            columns=["Low"],
            is_sensitive=True,
            uses_data=True,
        )

    @overrides
    def _calculate(self, df: pd.DataFrame) -> dict[str, float]:
        pass  # Not implementing because base feature


class Volume(BaseFeature):
    def __init__(self):
        super().__init__(
            columns=["Volume"],
            is_sensitive=True,
            uses_data=True,
        )

    @overrides
    def _calculate(self, df: pd.DataFrame) -> dict[str, float]:
        pass  # Not implementing because base feature


class Dividends(BaseFeature):
    def __init__(self):
        super().__init__(
            columns=["Dividends"],
            is_sensitive=True,
            uses_data=True,
        )

    @overrides
    def _calculate(self, df: pd.DataFrame) -> dict[str, float]:
        pass  # Not implementing because base feature


class Stock_Splits(BaseFeature):
    def __init__(self):
        super().__init__(
            columns=["Stock_Splits"],
            is_sensitive=True,
            uses_data=True,
        )

    @overrides
    def _calculate(self, df: pd.DataFrame) -> dict[str, float]:
        pass  # Not implementing because base feature


class FeatureMeta(type):
    def __getattr__(cls, attr):
        try:
            if attr in cls.base_features:
                return cls.base_features[attr]
            else:
                return cls.feature_list[attr]

        except KeyError:
            raise AttributeError(
                f"Column '{attr}' not found in feature data. Possible Features: {cls.list_added_cols()}"
            )


class Features(metaclass=FeatureMeta):
    feature_list: dict[str, BaseFeature] = {"Increased": Increased()}
    base_features: dict[str, BaseFeature] = {
        "Open": Open(),
        "High": High(),
        "Low": Low(),
        "Close": Close(),
        "Volume": Volume(),
        "Dividends": Dividends(),
        "Stock_Splits": Stock_Splits(),
    }

    @classmethod
    def add(cls, name: str, feature: BaseFeature):
        """
        Adds a feature to the internal index
        :param name: Name of the feature, used when doing Features[name]
        :param feature: The class of the feature representing its logic
        :return:
        """
        cls.feature_list[name] = feature

    @staticmethod
    def flatten_stocklist(stocks: list[str]) -> list[str]:
        flattened_list = []

        for stock in stocks:
            if ".csv" in stock:
                with open(stock, "r") as f:
                    for line in f:
                        for item in line.split(","):
                            flattened_list.append(item.strip())
            else:
                flattened_list.append(stock)
        return flattened_list

    @staticmethod
    def propagate_attrs(orig_df, new_df):
        new_df.attrs = orig_df.attrs.copy()
        return new_df

    @staticmethod
    def get_raw_stock(
        name: str,
        period: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> pd.DataFrame:
        ticker = yf.Ticker(name)
        historical_data = ticker.history(start=start_date, end=end_date, period=period)

        if historical_data.empty:
            raise Exception(f"Stock {name} not found")

        historical_data.index = historical_data.index.tz_localize(None).normalize()
        historical_data.attrs["last_date"] = historical_data.index[-1]

        return historical_data

    @staticmethod
    def parse_name(name: str):
        """
        Parses the input string for stock, start_date, end_date, and period.
        Returns a tuple: (stock, start_date, end_date, period).
        """
        stock = name
        start_date = None
        end_date = None
        period = None
        if ":" in name:
            stock, info = name.split(":", maxsplit=1)
            if "-" in info:
                if "," in info:
                    start_date_str, end_date_str = info.split(",", maxsplit=1)
                    start_date = datetime.strptime(start_date_str, "%m-%d-%Y")
                    end_date = datetime.strptime(end_date_str, "%m-%d-%Y")
                else:
                    start_date = datetime.strptime(info, "%m-%d-%Y")
            else:
                period = info
        return stock, start_date, end_date, period

    def get_stocks_parse(self, name: str) -> pd.DataFrame:
        stock, start_date, end_date, period = self.parse_name(name)
        original_df = self.get_stocks(
            stock, period=period, start_date=start_date, end_date=end_date
        )

        return original_df

    @classmethod
    def list_added_cols(cls):
        for feature in cls.base_features.values():
            for col in feature.cols():
                yield col

        for feature in cls.feature_list.values():
            for col in feature.cols():
                yield col

    def feat_list(self):
        for feature in self.train_on:
            yield feature

        yield self.predict_on

    def list_cols(self, with_true=False, prev_cols=False):
        for feature in self.train_on:
            for col in feature.cols():
                if feature.is_sensitive and prev_cols:
                    yield "prev_" + col
                else:
                    yield col

        for col in self.predict_on.cols():
            if self.predict_on.is_sensitive and prev_cols:
                yield "prev_" + col
            else:
                yield col

            if with_true:
                yield "true_" + col

    def train_cols(self, prev_cols=True):
        for feature in self.train_on:
            for col in feature.cols():
                if prev_cols and feature.is_sensitive:
                    yield "prev_" + col
                else:
                    yield col

    def predict_cols(self, with_true=True, prev_cols=False):
        for col in self.predict_on.cols():
            if prev_cols:
                yield "prev_" + col
                if with_true:
                    yield "true_" + col
            else:
                yield col

    def true_col(self):
        return self.predict_on.true_col()

    def prediction_col(self):
        return self.predict_on.prediction_col()

    @staticmethod
    def drop_na(df: pd.DataFrame):
        # print single full row full width
        df.dropna(inplace=True)

    def get_stocks(
        self,
        name: str,
        period: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        raw_stocks: DataFrame = None,
    ) -> pd.DataFrame:
        """
        Gets the stock data from yfinance and calculates each feature
        :param raw_stocks: Allows you to convert raw stock data into parsed model specific format
        :param name: Name of stock
        :param period: Period of how much data to take
        :param start_date: starting date of what data to take
        :param end_date: Ending date of what data to take
        :return: Dataframe with each main feature requested and each feature requested
        """
        if raw_stocks is not None:
            df = raw_stocks
        else:
            df = self.get_raw_stock(name, period, start_date, end_date)

        # Debug
        # print("10 rows of DataFrame before dropping NaNs:")
        # pd.set_option("display.max_columns", None)
        # print(df.tail(10))

        feat_data: dict[BaseFeature, pd.DataFrame] = {}
        for feature in self.feat_list():
            if feature not in self.base_features.values():
                feat_df = feature.calculate(df.copy())
            else:
                # print(f"DF cols: {df.columns} vs requested cols: {feature.cols()}")
                feat_df = df[feature.cols()].copy()

            feat_data[feature] = feat_df

            if feature is self.predict_on:
                for col in feature.columns:
                    prev_col = f"prev_{col}"
                    feat_df.loc[:, prev_col] = feat_df[col].shift(1)
                    true_col = f"true_{col}"
                    feat_df.rename(columns={col: true_col}, inplace=True)
            elif feature.is_sensitive:
                for col in feature.columns:
                    prev_col = f"prev_{col}"
                    feat_df.loc[:, prev_col] = feat_df[col].shift(1)
                    feat_df.drop(columns=[col], inplace=True)

        # Concatenate feature data
        df = pd.concat(feat_data.values(), axis=1)

        # Propagate attrs to the new DataFrame
        # df = self.propagate_attrs(
        #     self.get_raw_stock(name, period, start_date, end_date), df
        # )

        df = df[list(self.list_cols(with_true=True, prev_cols=True))]

        orig_len = len(df)
        self.drop_na(df)

        # print(f"Dropped {orig_len - len(df)} rows out of {orig_len} rows")

        return df

    def price_diff(self, df):
        # Give each row to predict_on.price_diff
        return self.predict_on.price_diff(df)

    def get_buy_df(self, data, pred_val_col: str = None):
        if pred_val_col is None:
            pred_val_col = self.predict_on.true_col()

        predicted_values = data[pred_val_col]

        filt_data = data[self.list_cols(prev_cols=True)]

        # remove pred_val_col
        data.drop([pred_val_col], axis=1)

        # Calculate buy days based on the actual features and predicted values
        buy_signals_list = self.predict_on.calcBuyDays(filt_data, predicted_values)

        # Convert the list of buy signals to a pandas Series
        buy_signals_series = pd.Series(buy_signals_list, index=filt_data.index)

        return buy_signals_series

    def __init__(self, features: list[BaseFeature], predict_on: BaseFeature):
        assert (
            predict_on.base_sensitive is not None
        )  # Make sure that you can actually predict on the given feature
        assert features is not []  # Make sure that you have at least one feature

        # Sets sensitivity for each base feature
        Features.Open.is_sensitive = predict_on.base_sensitive.Open
        Features.Close.is_sensitive = predict_on.base_sensitive.Close
        Features.High.is_sensitive = predict_on.base_sensitive.High
        Features.Low.is_sensitive = predict_on.base_sensitive.Low
        Features.Volume.is_sensitive = predict_on.base_sensitive.Volume
        Features.Dividends.is_sensitive = predict_on.base_sensitive.Dividends
        Features.Stock_Splits.is_sensitive = predict_on.base_sensitive.Stock_Splits

        self.predict_on: BaseFeature = predict_on
        self.train_on: list[BaseFeature] = features


def import_children(directory="Features"):
    models_dir = os.path.join(os.path.dirname(__file__), directory)

    for filename in os.listdir(models_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]
            file_path = os.path.join(models_dir, filename)
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # Add the module to sys.modules under its name
            sys.modules[module_name] = module


import_children()
