import numpy as np
import pandas as pd

from config import VARIABLE_SPECS, VariableSpec

RAW_DATA_PATH = "data/raw_data/fred_macro_monthly.csv"


TRANSFORMS = {
    "none": lambda x: x,
    "diff": lambda x: x.diff(),
    "log": lambda x: np.log(x),
    "log_diff": lambda x: np.log(x).diff(),
    "log_diff_12": lambda x: np.log(x).diff(12),
}

INVERSE = {
    "none": lambda x, anchor: x,
    "diff": lambda x, anchor: x.cumsum() + anchor,
    "log": lambda x, anchor: np.exp(x),
    "log_diff": lambda x, anchor: np.exp(x.cumsum())*anchor,
    "log_diff_12": lambda x, anchor: np.exp(x) * anchor 
}


def transform_data(raw_path: str = RAW_DATA_PATH, raw_df:pd.DataFrame = None) -> pd.DataFrame:
    "implement stationarity transformations and export transformed CSV"
    if not raw_df:
        raw_df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    
    output = {}

    for var in VARIABLE_SPECS:
        if var.name not in raw_df.columns:
            continue

        function = TRANSFORMS[var.transformation] if var.transformation else TRANSFORMS["none"]
        output[var.name] = function(raw_df[var.name])

    output_df = pd.DataFrame(output)
    output_df.index.name = "date"
    output_df.to_csv("data/processed/transformed_fred_data.csv")

    return pd.DataFrame(output)


def inverse_transform(transformed_df: pd.DataFrame, raw_df: pd.DataFrame = RAW_DATA_PATH) -> pd.DataFrame:
    "reverse transformations to get back to original scale"
    output = {}
    for var in VARIABLE_SPECS:
        if var.name not in transformed_df.columns or var.name not in raw_df.columns:
            continue

        if var.transformation is None:
            output[var.name] = transformed_df[var.name]
            continue
        elif var.transformation == "log_diff_12":
            # only works as long as we don't foecast more than 12 months in the future
            anchor = raw_df[var.name].iloc[-12:].values
        else:
            anchor = raw_df[var.name].iloc[-1]

        function = INVERSE[var.transformation]
        output[var.name] = function(transformed_df[var.name], anchor)

    output_df = pd.DataFrame(output)
    output_df.index.name = "date"
    output_df.to_csv("data/processed/inverse_transformed_data.csv")

    return pd.DataFrame(output)


if __name__ == "__main__":
    transform_data()
