import numpy as np
import pandas as pd
from overrides import overrides
from sklearn.linear_model import SGDRegressor

from TimeSeriesPrediction.data import Data
from TimeSeriesPrediction.features import Features
from TimeSeriesPrediction.model import Commons


class RegressionModel(Commons):
    # Creates new model
    def __init__(self):
        feat = [
            Features.Open,
            Features.BB,
            Features.RSI,
            Features.Date,
            Features.MA,
            Features.MACD,
        ]
        super().__init__(SGDRegressor(), "Linear", Features(feat, Features.Close))

    # Trains Model on given data
    @overrides
    def _train(self, df: pd.DataFrame):
        # Create a new DataFrame with the necessary columns

        x, y = Data.train_split(
            df, self.features.train_cols(), self.features.predict_on
        )
        # Train the model on the dataset
        self.model.fit(x, y)
        self.is_trained = True

    @overrides
    def _batch_predict(self, df: pd.DataFrame) -> np.array:
        x_test, y_test = Data.train_split(
            df, self.features.train_cols(), self.features.predict_on
        )

        predictions = []
        for i in range(len(x_test) - self.lookback + 1):
            x_window = x_test[i : i + self.lookback]

            # Make a prediction for the next time step
            prediction = self.model.predict(x_window)
            predictions.append(prediction[0])

        return np.array(predictions)

    @overrides
    def _predict(self, df: pd.DataFrame) -> float:
        x = df[self.features.train_cols()].values[
            -self.lookback :
        ]  # Select the last row

        # Predict the target values using the model
        prediction = self.model.predict(x)

        # Get result
        return prediction


# Important, as it adds the model to the CLI
Commons.model_mapping["Linear"] = RegressionModel
