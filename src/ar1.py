import pandas as pd
import numpy as np
import statsmodels.api as sm

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
        y = series.iloc[1:]
        X = series.iloc[:-1]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        self.beta, self.phi = model.params
        self.sigma2 = model.mse_resid
        self.last_value = series.iloc[-1]
        self.last_index = series.index[-1]  
        return self
    
    # forecast future values with confidence intervals. Returns a DataFrame with columns "forecast", "lower", "upper"
    def forecast(self, horizon: int, z:float = 1.96) -> pd.Series:
        if self.phi is None or self.beta is None:
            raise ValueError("Model must be fitted before prediction.")
        
        last_value = self.last_value
        forecasts, lower, upper = [], [], []

        for h in range(1, horizon+1):
            next_value = self.beta + self.phi * last_value
            forecasts.append(next_value)
            last_value = next_value

            # confidence bands
            if abs(self.phi) < 1:
                var_h = self.sigma2 * (1 - self.phi**(2*h)) / (1 - self.phi**2)
            else:
                var_h = self.sigma2 * h  # non-stationary case, variance grows linearly

            std_h = np.sqrt(var_h)
            lower.append(next_value - z * std_h)
            upper.append(next_value + z * std_h)

        # create index for future periods  
        idx = pd.RangeIndex(start=self.last_index + 1, 
                            stop=self.last_index + 1 + horizon)
        
        return pd.DataFrame({"forecast": forecasts, "lower": lower, "upper": upper}, index=idx)