import pandas as pd
import numpy as np
from statistics import NormalDist

class VAR:
    lags: int
    coefs: np.ndarray
    sigma_u: np.ndarray
    var_names: list[str]
    data: pd.DataFrame

    def __init__(self, lags: int = 1):
        self.lags = lags
        self.coefs = None
        self.sigma_u = None
        self.var_names = None
        self.data = None

    def fit(self, data: pd.DataFrame):
        # get var names
        self.data = data
        self.var_names = data.columns.tolist()

        # Build regressor matrix with lags
        X = pd.concat([data.shift(lag) for lag in range(1, self.lags + 1)], axis=1).dropna()
        X = np.column_stack((np.ones(X.shape[0]), X.values))  # add constant

        # Build target matrix
        Y = data.iloc[self.lags:].values

        # Do OLS regression using matrices to solve all equations at once
        self.coefs = np.linalg.inv(X.T @ X) @ X.T @ Y
        residuals = Y - X @ self.coefs
        self.sigma_u = (residuals.T @ residuals) / (len(Y) - X.shape[1])

        return self
    
    def forecast(self, horizon: int, alpha: float = 0.05) -> pd.DataFrame:
        if self.coefs is None or self.sigma_u is None:
            raise ValueError("Model must be fitted before prediction.")
        
        n = len(self.var_names)
        forecasts = np.zeros((horizon, n))

        # select only the last lags for each var and flip it so that we have most recent lags first
        lag_vars = self.data.iloc[-self.lags:].values[::-1]
        # reshape into a vector keeping the order of var1_lag1, var2_lag1, ..., varN_lag1, var1_lag2, ...
        lag_vars = lag_vars.reshape(1, self.lags * n)

        for h in range(horizon):
            # add constant
            x = np.column_stack((np.ones(1), lag_vars))
            # predict next value
            next_values = x @ self.coefs
            forecasts[h] = next_values
            # create new x by dropping oldest lag and adding new forecast as lag1
            lag_vars = np.hstack((next_values, lag_vars[:, :-n]))  

        # h-step forecast MSE using companion matrix recursion
        mse_matrices = self._forecast_mse(horizon)

        # build index of future periods
        last_index = self.data.index[-1]
        idx = pd.RangeIndex(start=last_index + 1, stop=last_index + 1 + horizon)

        # build dataframe with forecasts and cofidence intervals
        z = NormalDist().inv_cdf(1 - alpha / 2)
        output = {}
        for i, col in enumerate(self.var_names):
            std_h = np.array([np.sqrt(mse_matrices[h][i, i]) for h in range(horizon)])
            output[col] = pd.DataFrame({
                "forecast": forecasts[:, i],
                "lower": forecasts[:, i] - z * std_h,
                "upper": forecasts[:, i] + z * std_h
            }, index=idx)
        
        return output



    def _companion_matrix(self) -> np.ndarray:
        n = len(self.var_names)
        p = self.lags
        F = np.zeros((n * p, n * p))

        # coefs is (1 + n*p, n), first row is intercept
        # A_l is the (n x n) block for lag l
        for lag in range(1, p + 1):
            start = 1 + (lag - 1) * n
            A_l = self.coefs[start:start + n, :].T   # (n, n)
            F[:n, (lag - 1) * n: lag * n] = A_l

        if p > 1:
            F[n:, :-n] = np.eye(n * (p - 1))

        return F


    def _forecast_mse(self, horizon: int) -> list[np.ndarray]:
        n = len(self.var_names)
        F = self._companion_matrix()

        Q = np.zeros_like(F)
        Q[:n, :n] = self.sigma_u

        mse = []
        F_power = np.eye(F.shape[0])
        cumulative = np.zeros_like(F)

        for _ in range(horizon):
            cumulative += F_power @ Q @ F_power.T
            mse.append(cumulative[:n, :n].copy())
            F_power = F_power @ F

        return mse
                


