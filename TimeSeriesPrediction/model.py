from __future__ import annotations

import importlib.util
import os
import sys
import time
import warnings
from abc import ABC, abstractmethod
from contextlib import redirect_stdout
from datetime import datetime
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from joblib import dump

from TimeSeriesPrediction.data import Data
from TimeSeriesPrediction.features import Features
from TimeSeriesPrediction.metrics import Metrics
from TimeSeriesPrediction.normalizer import Normalizer

warnings.filterwarnings("ignore", category=FutureWarning)


# Exceptions:
class ModelNotTrainedError(Exception):
    """Exception raised when trying to use an untrained model."""

    def __init(self):
        super().__init__("Model not trained! Must be trained first.")


class BaseClassError(TypeError):
    """Exception raised when trying to instantiate a class that should be inherited."""

    def __init__(self):
        super().__init__(
            "Commons class should not be instantiated directly, please use one of its inheritors."
        )


class Commons(ABC):
    """
    Base Class for all models and should be inherited.
    Not to be initiated directly!
    """

    model_mapping = {}

    def __init__(self, model, model_type: str, features: Features, lookback: int = 30):
        # Sets the version of scheme(stored values)
        self.seed = None
        self.model_version: float = 1.0
        self.model_type: str = model_type

        # Model training settings
        self.features = features
        self.lookback: int = lookback
        self.training_stock: [str] = []
        self.is_trained = False

        # Create model with settings
        self.model = model

        # Normalize
        self.normalizer = None

    def set_seed(self, seed: int | None = None):
        self.seed = seed

    def use_seed(self, seed: int | None = None):
        if seed is None:
            seed = int(time.time() * 1000) % 2**32
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull):
                # your code here—for example:
                seed = int(time.time() * 1000) % 2**32
                os.environ["PYTHONHASHSEED"] = str(seed)
                pl.seed_everything(seed, workers=True)
                np.random.seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    def save_model(self, file: str, compress_lvl=6):
        """
        Saves model to specific file
        :param compress_lvl: Sets the level of compression for the file.
        :param file: Save location in raw string form
        :return: None
        """
        dump(self, file, compress=("gzip", compress_lvl))

    @abstractmethod
    def _train(self, df: pd.DataFrame):
        """
        :param df: DataFrame with all features
        :return: None
        """
        raise NotImplementedError()

    @abstractmethod
    def _batch_predict(self, df: pd.DataFrame) -> np.array:
        """
        :param df: Stock market data with all features
        :return: returns just the prediction column (pred_value column)
        """
        raise NotImplementedError()

    @abstractmethod
    def _predict(self, df: pd.DataFrame) -> float:
        """
        :param df: takes input
        :return: returns just the prediction
        """
        raise NotImplementedError()

    def train(self, df: pd.DataFrame):
        """
        Trains model on df. Warning some models have special constrains, such as Linear model can only be trained on 1 set of data.
        :param df: DataFrame with all features
        :return: None
        """
        self.use_seed(self.seed)
        self._train(self._normalize(df, allow_calibration=True))

    # Overwritten for prediction of outputs, Adds column of "pred_value" as close predictions from model

    def batch_predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Takes the entire dataset and predicts values from it. Useful for calculate metrics of model.
        :param df: Stock market data with all features
        :return: returns just the prediction column (pred_value column)
        """
        self.use_seed(self.seed)
        if not self.is_trained:
            raise ModelNotTrainedError
        return self._inv_normalize(
            np.concatenate(
                (
                    np.full(
                        (self.lookback - 1, 1), np.nan
                    ),  # 2D array with shape (lookback - 1, 1)
                    self._batch_predict(self._normalize(df)).reshape(
                        -1, 1
                    ),  # Reshape to 2D
                )
            ),
            based_on=self.features.true_col(),
        )

    def predict(self, df: pd.DataFrame) -> Tuple[datetime, float]:
        """
        Predicts on given values, if model type can only handle 1 input row, it will use the last row as the input.

        :param df: takes input
        :return: returns just the prediction
        """
        self.use_seed(self.seed)
        if "last_date" in df.attrs:
            date = df.attrs["last_date"]
        else:
            date = df.index[-1]
        return (
            date,
            self._inv_normalize_value(
                self._predict(self._normalize(df)), str(self.features.true_col())
            )[0],
        )

    def _normalize(
        self, df: pd.DataFrame, convert: [str] = None, allow_calibration=False
    ):
        if self.normalizer is None:
            if convert is None:
                convert = self.features.list_cols(prev_cols=True, with_true=True)

            if allow_calibration is True:
                self.normalizer = Normalizer(df, convert)
            else:
                raise ValueError(f"Normalizer not set, please train first")
            df = self.normalizer.scale(df)
        else:
            df = self.normalizer.scale(df)
        return df

    def _inv_normalize(self, df: np.ndarray, based_on: str):
        if self.normalizer is not None:
            return self.normalizer.inv_normalize_np(df, based_on)
        else:
            raise ValueError(f"Normalizer not set, please train first")

    def _inv_normalize_value(self, value: float, based_on: str):
        if self.normalizer is not None:
            return self.normalizer.inv_normalize_value(value, based_on)
        else:
            raise ValueError(f"Normalizer not set, please train first")

    def calculate_metrics(
        self, df: pd.DataFrame, reduced=True, print_table=False, save_table=None
    ) -> Metrics:
        """
        Takes input df with all the columns (predicted values, and predicted on values) and calculates metrics on it.

        :param save_table:
        :param print_table: Prints debug table
        :param reduced: reduce the amount of data calculated
        :param df: DataFrame with all columns (features + predicted values)
        :return: Metrics object
        """
        Data.drop_na(df)
        y_true = df[list(self.features.predict_cols(prev_cols=True))[0]]
        y_pred = df[self.features.prediction_col()]
        buy_true = self.features.get_buy_df(df)

        buy_pred = self.features.get_buy_df(
            df, pred_val_col=self.features.prediction_col()
        )

        price_dif = self.features.price_diff(df)

        # Print combined df with original df and buy_true, buy_pred and price_diff
        if print_table:
            combined_df = pd.concat(
                [
                    df,
                    buy_true.to_frame("buy_true").rename(
                        columns={buy_true.name: "buy_true"}
                    ),
                    buy_pred.to_frame("buy_pred").rename(
                        columns={buy_pred.name: "buy_pred"}
                    ),
                    price_dif.to_frame("price_diff"),
                ],
                axis=1,
            )
            print(combined_df)

        if save_table is not None and "." in save_table:
            with pd.ExcelWriter(Data.sanitize_name(save_table)) as writer:
                combined_df = pd.concat(
                    [
                        df,
                        buy_true.to_frame("buy_true").rename(
                            columns={buy_true.name: "buy_true"}
                        ),
                        buy_pred.to_frame("buy_pred").rename(
                            columns={buy_pred.name: "buy_pred"}
                        ),
                        price_dif.to_frame("price_diff"),
                    ],
                    axis=1,
                )
                combined_df.index = combined_df.index.strftime("%Y-%m-%d")
                combined_df.to_excel(writer, sheet_name="Stock Data")

        return Metrics.create(
            y_true=y_true,
            y_pred=y_pred,
            buy_true=buy_true,
            buy_pred=buy_pred,
            price_dif=price_dif,
            is_number=self.features.predict_on.is_number and not reduced,
        )

    @staticmethod
    def load_from_file(file: str, if_exists=False):
        """
        Load model from file.
        :param file: Raw file location
        :param if_exists: If file doesn't exist, don't throw error if True
        :return: returns the model(of any variant) from specified file
        """

        try:
            return joblib.load(file)
        except FileNotFoundError:
            if if_exists:
                return None
            else:
                raise FileNotFoundError


def import_children(directory="Types"):
    models_dir = os.path.join(os.path.dirname(__file__), directory)
    sys.path.insert(0, models_dir)  # Add the directory to sys.path
    for file in os.listdir(models_dir):
        if file.endswith("Model.py"):
            model_name = file[:-3]  # Remove the .py extension
            try:
                importlib.import_module(model_name)
            except (ImportError, AttributeError) as e:
                print(f"Error importing {model_name}: {e}")
    sys.path.remove(models_dir)  # Remove the directory from sys.path after importing


# Assuming the child classes are in a directory named 'children'
import_children()
