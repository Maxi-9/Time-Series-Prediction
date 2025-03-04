import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Normalizer:
    # Needs: Data Frame and the columns that you want to be normalized
    # Use: Scalars only hold 1 min/max value, this holds it for all given columns

    def __init__(self, df: pd.DataFrame, scale_cols: [str]):
        self.scalars = {}
        df = df.copy(deep=True)
        for col in scale_cols:
            scaler = MinMaxScaler()
            df.loc[:, col] = scaler.fit_transform(df[[col]])
            self.scalars[col] = scaler

    def scale(self, df: pd.DataFrame) -> pd.DataFrame:
        # Scales using the scalars
        df = df.copy(deep=True)
        for col, scaler in self.scalars.items():
            if col in df.columns:
                df[col] = scaler.transform(df[[col]])
        return df

    def inv_normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy(deep=True)
        for col, scaler in self.scalars.items():
            if col in data.columns:
                data[col] = scaler.inverse_transform(data[[col]])
        return data

    def inv_normalize_np(self, data: np.ndarray, based_on: str) -> np.ndarray:
        scaler = self.scalars.get(based_on)

        if scaler is not None:
            return scaler.inverse_transform(data)
        else:
            raise ValueError(f"No scaler found for column: {based_on}")

    def inv_normalize_col(
        self, data: pd.DataFrame, convert: str, based_on: str
    ) -> pd.DataFrame:
        data = data.copy(deep=True)
        scaler = self.scalars.get(based_on)
        if scaler is not None:
            data[convert] = scaler.inverse_transform(data[[convert]])
            return data
        else:
            raise ValueError(f"No scaler found for column: {based_on}")

    def inv_normalize_value(self, value: float, based_on: str):
        scaler = self.scalars.get(based_on)

        if scaler is not None:
            # Reshape your input to match the original data shape
            value = np.array(value).reshape(-1, 1)
            inv_normalized = scaler.inverse_transform(value)[0]
            return inv_normalized
        else:
            raise ValueError(f"No scaler found for column: {based_on}")
