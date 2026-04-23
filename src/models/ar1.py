import pandas as pd
import numpy as np
import statsmodels.api as sm
from statistics import NormalDist

from ..utils import build_forecast_index
from ..logging_utils import log_debug


class AR1Model:
    # simple AR(1) model with optional intercept, fitted by OLS regression
    # y_t = beta + phi * y_{t-1} + epsilon_t, epsilon_t ~ N(0, sigma2)
    phi: float
    beta: float
    sigma2: float
    last_value: float
    last_index: pd.Timestamp

    # initialize model.  None to fit from data
    def __init__(self):
        self.phi = None
        self.beta = None
        self.sigma2 = None
        self.last_value = None
        self.last_index = None

    # fit from data series, using OLS regression of y_t on y_{t-1} with intercept
    def fit(self, series: pd.Series):
        # OLS regression of y_t on y_{t-1} with intercept
        data = series.values
        y = data[1:]
        X = data[:-1].reshape(-1,1)
        X = np.column_stack([np.ones(len(X)), X])  # add intercept
        model = sm.OLS(y, X).fit()
        self.beta, self.phi = model.params
        self.sigma2 = model.mse_resid
        self.last_value = series.iloc[-1]
        self.last_index = series.index[-1]
        
        log_debug(f"AR1 fitted: N={len(data)}, beta={self.beta:.6f}, phi={self.phi:.6f}, sigma2={self.sigma2:.6f}")
        return self
    
    # forecast future values with confidence intervals. Returns a DataFrame with columns "forecast", "lower", "upper"
    def forecast(self, horizon: int, alpha: float = 0.05) -> pd.DataFrame:
        if self.phi is None or self.beta is None:
            raise ValueError("Model must be fitted before prediction.")
        
        last_value = self.last_value
        forecasts, lower, upper = [], [], []

        for h in range(1, horizon+1):
            next_value = self.beta + self.phi * last_value
            forecasts.append(next_value)
            last_value = next_value

            # confidence bands
            z = NormalDist().inv_cdf(1 - alpha / 2)
            if abs(self.phi) < 1:
                var_h = self.sigma2 * (1 - self.phi**(2*h)) / (1 - self.phi**2)
            else:
                var_h = self.sigma2 * h  # non-stationary case, variance grows linearly

            std_h = np.sqrt(var_h)
            lower.append(next_value - z * std_h)
            upper.append(next_value + z * std_h)

        idx = build_forecast_index(self.last_index, list(range(1, horizon + 1)))
        return pd.DataFrame({"forecast": forecasts, "lower": lower, "upper": upper}, index=idx)
