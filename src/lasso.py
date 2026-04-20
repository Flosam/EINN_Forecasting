import pandas as pd
import numpy as np

class Lasso:
    lags: int
    horizon: int
    lmbda: float
    max_iter: int
    tol: float
    coefs: np.ndarray
    intercept: float
    data: pd.DataFrame
    target: str
    residuals: np.ndarray

    def __init__(self, lags: int = 1, horizon: int = 1, lmbda: float = 0.1, max_iter: int = 1000, tol: float = 1e-6):
        self.lags = lags
        self.horizon = horizon
        self.lmbda = lmbda
        self.max_iter = max_iter
        self.tol = tol
        self.coefs = None
        self.intercept = None
        self.data = None
        self.target = None
        self.residuals = None
        
    def fit(self, data: pd.DataFrame, target:str):
        # save data and target for forecasting later
        self.data = data.copy()
        self.target = target
        if len(self.data) - self.lags - self.horizon <= 0:
            raise ValueError("Data too short for requested lags and horizon.")

        # create lags of features
        X = pd.concat([data.shift(lag) for lag in range(self.lags + 1)], axis=1).dropna()
        # align X and y after creating lags and shifting y to predict next period
        X = X.iloc[:-self.horizon,:].values

        # select target variable shifted by -1 to predict next period
        y = self.data[target].shift(-self.horizon).dropna().values
        y = y[self.lags:]  # align with X after dropping lags

        n, p = X.shape

        # center X and y to handle intercept separately
        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        X_c = X - X_mean
        y_c = y - y_mean

        # compute column norms of X for Beta_hat computation
        col_norms = (X_c ** 2).sum(axis=0) / n

        # initialize coefficients
        betas = np.zeros(p)

        # coordinate descent algorithm
        for iter in range(self.max_iter):
            betas_old = betas.copy()

            for j in range(p):
                # compute partial residual excluding feature j
                r_j = y_c - X_c @ betas + X_c[:, j] * betas[j]
                # correlation of feature j with partial residual
                z_j = (X_c[:, j] @ r_j)/n
                # apply soft thresholding to get Lasso coefficient
                betas[j] = self._soft_thresholding(z_j, self.lmbda) / col_norms[j]

            # check convergence
            if np.max(np.abs(betas - betas_old)) < self.tol:
                print(f"Coverged at iteration: {iter}")
                break

        self.coefs = betas
        # recover intercept from centering
        self.intercept = y_mean - X_mean @ betas

        # compute insample predictions and residuals for bootstrapping
        insample_preds = self.intercept + X @ self.coefs
        self.residuals = y - insample_preds

        return self
    

    def forecast(self, alpha: float = 0.05, n_bootstrap: int = 1000) -> pd.DataFrame:
        if self.coefs is None or self.intercept is None:
            raise ValueError("Model must be fitted before prediction.")
        
        p = self.data.shape[1]
        # create X for forecasting future periods by adding lags
        X = self.data.iloc[-(self.lags + 1):].values[::-1].flatten()  # get last lags and flatten into vector

        # predict next value
        forecast = self.intercept + X @ self.coefs
        
        # bootstrap confidence bands
        bootstrap_forecasts = np.zeros(n_bootstrap)
        # bootstrap loop
        for b in range(n_bootstrap):
            # resample residuals with replacement
            resampled_residuals = np.random.choice(self.residuals, size=1, replace=True)
            # generate bootstrap forecast path
            bootstrap_forecasts[b] = self.intercept + X @ self.coefs + resampled_residuals[0]
             

        # compute confidence intervals from bootstrap distribution
        lower = np.percentile(bootstrap_forecasts,100*alpha/2)
        upper = np.percentile(bootstrap_forecasts, 100*(1-alpha/2))

        # build index of future periods
        last_index = self.data.index[-1]
        idx = pd.RangeIndex(start=last_index + self.horizon, stop=last_index + self.horizon + 1)

        return pd.DataFrame({"forecast": forecast, "lower": lower, "upper": upper}, index=idx)

    
    def _soft_thresholding(self, z: float, lmbda: float) -> float:
        if z > lmbda:
            return z - lmbda
        elif z < -lmbda:
            return z + lmbda
        else:
            return 0.0
