import pandas as pd
import numpy as np
from statistics import NormalDist

from ..utils import build_forecast_index
from ..logging_utils import log_debug

class VAR:
    lags: int
    coefs: np.ndarray
    sigma_u: np.ndarray
    var_names: list[str]
    data: pd.DataFrame
    use_bic: bool
    max_bic_lags: int

    def __init__(self, lags: int = 1, use_bic: bool = False, max_bic_lags: int = 4):
        self.lags = lags
        self.use_bic = use_bic
        self.max_bic_lags = max_bic_lags
        self.coefs = None
        self.sigma_u = None
        self.var_names = None
        self.data = None

    def _compute_bic(self, data: pd.DataFrame) -> tuple[int, np.ndarray]:
        """Compute BIC for lag orders 1..max_bic_lags and return optimal lag order and residual cov."""
        bic_scores = []
        residual_covs = []
        n = data.shape[1]

        for p in range(1, self.max_bic_lags + 1):
            X = pd.concat([data.shift(lag) for lag in range(1, p + 1)], axis=1).dropna()
            X = np.column_stack((np.ones(X.shape[0]), X.values))
            Y = data.iloc[p:].values

            T = len(Y)
            dof = T - X.shape[1]
            if dof < 1:
                log_debug(f"VAR BIC lag={p}: skipped (insufficient DOF)")
                continue
            
            try:
                coefs = np.linalg.inv(X.T @ X) @ X.T @ Y
            except np.linalg.LinAlgError:
                log_debug(f"VAR BIC lag={p}: skipped (singular matrix)")
                continue
            
            residuals = Y - X @ coefs
            sigma_u = (residuals.T @ residuals) / T
            
            # Check for valid covariance matrix
            if not np.all(np.isfinite(sigma_u)):
                log_debug(f"VAR BIC lag={p}: skipped (non-finite covariance)")
                continue

            # BIC = log(det(Sigma)) + (k/T)*log(T)
            # where k = n*(1 + n*p) is number of parameters
            try:
                log_det_sigma = np.log(np.linalg.det(sigma_u))
                if not np.isfinite(log_det_sigma):
                    log_debug(f"VAR BIC lag={p}: skipped (non-finite determinant)")
                    continue
            except (np.linalg.LinAlgError, ValueError):
                log_debug(f"VAR BIC lag={p}: skipped (determinant computation failed)")
                continue

            num_params = n * (1 + n * p)
            bic = log_det_sigma + (num_params / T) * np.log(T)

            bic_scores.append(bic)
            residual_covs.append(sigma_u)
            log_debug(f"VAR BIC lag={p}: score={bic:.6f}")

        if not bic_scores:
            raise ValueError("No valid lag orders found for BIC. Check data quality.")

        optimal_lag = np.argmin(bic_scores) + 1
        log_debug(f"VAR BIC auto-selected lag order: {optimal_lag}")
        return optimal_lag, residual_covs[np.argmin(bic_scores)]

    def fit(self, data: pd.DataFrame):
        # get var names
        self.data = data
        self.var_names = data.columns.tolist()

        # Auto-select lag order using BIC if requested
        if self.use_bic:
            self.lags, _ = self._compute_bic(data)
            log_debug(f"VAR fitted with auto-selected lags={self.lags}")
        else:
            log_debug(f"VAR fitted with manual lags={self.lags}")

        # Build regressor matrix with lags
        X = pd.concat([data.shift(lag) for lag in range(1, self.lags + 1)], axis=1).dropna()
        X = np.column_stack((np.ones(X.shape[0]), X.values))  # add constant

        # Build target matrix
        Y = data.iloc[self.lags:].values

        # Check for insufficient data
        dof = len(Y) - X.shape[1]
        if dof < 1:
            raise ValueError(f"Insufficient degrees of freedom: {dof}. Need more data or reduce lags.")

        # Do OLS regression using matrices to solve all equations at once
        try:
            self.coefs = np.linalg.inv(X.T @ X) @ X.T @ Y
        except np.linalg.LinAlgError:
            raise ValueError("Singular matrix in VAR regression. Check data for multicollinearity.")
        
        residuals = Y - X @ self.coefs
        self.sigma_u = (residuals.T @ residuals) / dof
        
        # Check for NaN/inf in covariance
        if not np.all(np.isfinite(self.sigma_u)):
            raise ValueError("Non-finite values in residual covariance matrix.")

        # Check companion matrix stability (eigenvalues should be < 1 for stationarity)
        F = self._companion_matrix()
        eigenvalues = np.linalg.eigvals(F)
        max_eig = np.max(np.abs(eigenvalues))
        log_debug(f"VAR companion matrix max eigenvalue: {max_eig:.6f}")
        if max_eig > 1.0:
            log_debug(f"WARNING: VAR process is non-stationary (max eig={max_eig:.4f} > 1). Forecasts may diverge.")

        return self
    
    def forecast(self, horizon: int, alpha: float = 0.05, target: str | None = None) -> pd.DataFrame:
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
            
            # Check for extreme divergence (sign of instability)
            if np.any(np.abs(next_values) > 1e6):
                log_debug(f"VAR forecast diverged at horizon {h}: {next_values}")
                raise ValueError(f"VAR forecast diverged at horizon {h}. Process is unstable.")

        # h-step forecast MSE using companion matrix recursion
        mse_matrices = self._forecast_mse(horizon)

        last_index = self.data.index[-1]
        idx = build_forecast_index(last_index, list(range(1, horizon + 1)))

        target_name = target or self.var_names[0]
        i = self.var_names.index(target_name)

        z = NormalDist().inv_cdf(1 - alpha / 2)
        std_h = np.array([np.sqrt(mse_matrices[h][i, i]) for h in range(horizon)])
        
        # Cap extreme standard deviations
        std_h = np.minimum(std_h, 100)
        
        return pd.DataFrame(
            {
                "forecast": forecasts[:, i],
                "lower": forecasts[:, i] - z * std_h,
                "upper": forecasts[:, i] + z * std_h,
            },
            index=idx,
        )



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

        # Check for non-finite companion matrix
        if not np.all(np.isfinite(F)):
            raise ValueError("Non-finite values in companion matrix.")

        Q = np.zeros_like(F)
        Q[:n, :n] = self.sigma_u

        mse = []
        F_power = np.eye(F.shape[0])
        cumulative = np.zeros_like(F)

        for h in range(horizon):
            # Check for numerical overflow/underflow
            if not np.all(np.isfinite(F_power)):
                log_debug(f"F_power diverged at horizon {h}. Stopping MSE computation.")
                break
            
            cumulative += F_power @ Q @ F_power.T
            
            # Extract MSE for target variable and handle any NaN
            mse_h = cumulative[:n, :n].copy()
            if not np.all(np.isfinite(mse_h)):
                log_debug(f"MSE matrix contains non-finite values at horizon {h}. Using previous values.")
                if mse:
                    mse_h = mse[-1].copy()
                else:
                    mse_h = self.sigma_u.copy()
            
            mse.append(mse_h)
            F_power = F_power @ F

        return mse
                


