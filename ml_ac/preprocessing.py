from io import BytesIO
from typing import Tuple
import pandas as pd


X_COLS = ["JOB_TYPE", "DEPTH_M", "DIAMETER_MM", "MATERIAL", "LENGTH_M"]
Y_COL = "JOB_COST"


class Preprocess:
    def __init__(
        self,
        outlier_deviation_max: int = 3,
    ) -> None:
        """
        Runs preprocessing steps of loading data, removing null data and removing
        outliers.

        Parameters
        ----------
        outlier_deviation_max : int
            Sets the maximum standard deviations from mean for data to
            be a non outlier.
        """
        self.thresh = outlier_deviation_max

    def load_data(self, data_stream: BytesIO) -> Tuple[pd.DataFrame, pd.DataFrame]:

        self.raw_data = pd.read_csv(data_stream)
        data = self.raw_data[~self.raw_data[Y_COL].isnull()]

        y = data[[Y_COL]]
        X = data[X_COLS]
        return X, y

    def remove_outliers(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        for col in X[X.columns[X.dtypes != "object"]]:

            min_val = X[col].mean() - self.thresh * X[col].std()
            max_val = X[col].mean() + self.thresh * X[col].std()
            X = X[(X[col] <= max_val) & (X[col] >= min_val)]

        y = y.loc[X.index]

        return X, y

    def removed_data(self, X: pd.DataFrame) -> pd.DataFrame:
        removed_rows = self.raw_data.iloc[~self.raw_data.index.isin(X.index)]
        return removed_rows
