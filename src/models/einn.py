import pandas as pd
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax 
from ..utils import build_forecast_index
from ..logging_utils import log_debug

class EINN(nnx.Module):

    def __init__(
        self,
        din: int,
        dmid: tuple[int, ...] = (32, 32),
        rngs: nnx.Rngs = nnx.Rngs(0),
        learning_rate: float = 0.001,
        epochs: int = 1000,
        lags: int = 1,
        pc_weight: float = 0.5,
        horizons: list[int] = [1, 3, 6, 12],
        horizon_weight: float = 0.7
    ):
        self.rngs = rngs
        self.lags = lags
        self.epochs = epochs
        self.pc_weight = pc_weight
        self.horizons = horizons
        self.residuals = nnx.data(None)
        self.horizon_weight = horizon_weight

        # define layers as attributes
        layers = []
        layers.append(nnx.LayerNorm(din, rngs=rngs))  # batch norm on input layer
        input_dim = din
        for i, hidden_dim in enumerate(dmid):
            layers.append(nnx.Linear(input_dim, hidden_dim, rngs=rngs))
            layers.append(nnx.relu)
            input_dim = hidden_dim
        layers.append(nnx.Linear(input_dim, 4, rngs=rngs))  # output layer
        self.layers = nnx.Sequential(*layers)

        # define philips curve parameters for loss function
        self.beta_raw = nnx.Param(jnp.array(0.0), name="beta_raw")
        self.kappa_raw = nnx.Param(jnp.array(0.0), name="kappa_raw")
        
        # define optimizer
        self.optimizer = nnx.Optimizer(self, optax.adam(learning_rate), wrt=nnx.Param)

    def __call__(self, x):
        return self.layers(x)
    
    def _train_step(self, X, y, nkpc_gap, nkpc_expectations):
        def loss_fn(model):
            preds = model(X)
            pred_error = jnp.mean((preds - y) ** 2)

            # Phillips curve loss across configured horizons
            beta, kappa = model._get_coeffs()
            nkpc_preds = beta * nkpc_expectations + kappa * nkpc_gap
            # weight horizons with decaying weight, normalized to sum to 1
            horizon_weights = jnp.array([self.horizon_weight ** (h-1) for h in self.horizons])
            horizon_weights = horizon_weights / jnp.sum(horizon_weights)
            # compute weighted average of nkpc_preds across horizons
            per_horizon_error = jnp.mean((preds - nkpc_preds) ** 2, axis=0)
            pc_error = jnp.dot(horizon_weights, per_horizon_error)

            return (1-self.pc_weight) * pred_error + self.pc_weight * pc_error

        loss, grads = nnx.value_and_grad(loss_fn)(self)
        self.optimizer.update(self, grads)
        return loss
    
    def _get_coeffs(self):
        beta = nnx.sigmoid(self.beta_raw.value)
        kappa = jnp.exp(self.kappa_raw.value)
        return beta, kappa
    
    def get_nkpc_params(self) -> dict[str, float]:
        """Get learned NKPC parameters: beta (expectations weight) and kappa (gap weight)."""
        beta, kappa = self._get_coeffs()
        return {
            "beta": float(beta),  # Forward expectations weight (Phillips curve slope)
            "kappa": float(kappa),  # Output gap weight (sensitivity to slack)
        }

    def fit(self, data: pd.DataFrame, 
            target: str = "cpi_all_items", 
            gap_variables: list[str] = ["unemployment_rate", "natural_rate_unemployment"],
            expectations_variable: str = "inflation_expectations_umich"):
        
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data.")
        max_h = max(self.horizons)
        if len(data) - self.lags - max_h <= 0:
            raise ValueError(f"Data too short for requested lags and {max_h}-step target.")

        # create lags of features
        X = pd.concat([data.shift(lag) for lag in range(self.lags + 1)], axis=1).dropna().values
        X = X[:-max_h] # align with y after creating multi-step target

        # create target array for multi-step forecasting
        y = pd.concat([data[target].shift(-h) for h in self.horizons], axis=1).dropna().values
        y = y[self.lags:]  # align with X after dropping lags

        # create nkpc variables for Phillips curve loss on matching horizons
        nkpc_gap = data[gap_variables[0]] - data[gap_variables[1]]
        nkpc_expectations = data[expectations_variable]
        nkpc_gap = pd.concat([nkpc_gap.shift(-h) for h in self.horizons], axis=1).dropna().values
        nkpc_expectations = pd.concat(
            [nkpc_expectations.shift(-h) for h in self.horizons], axis=1
        ).dropna().values
        nkpc_gap = nkpc_gap[self.lags:]
        nkpc_expectations = nkpc_expectations[self.lags:]

        X = jnp.asarray(X, dtype=jnp.float32)
        y = jnp.asarray(y, dtype=jnp.float32)
        nkpc_gap = jnp.asarray(nkpc_gap, dtype=jnp.float32)
        nkpc_expectations = jnp.asarray(nkpc_expectations, dtype=jnp.float32)

        # training loop
        for epoch in range(self.epochs):
            loss = self._train_step(X, y, nkpc_gap, nkpc_expectations)
            if epoch % 100 == 0:
                log_debug(f"EINN Epoch {epoch}, Loss: {loss:.4f}")
        
        beta, kappa = self._get_coeffs()
        log_debug(f"EINN training complete. Final loss: {loss:.4f}, beta={beta:.6f}, kappa={kappa:.6f}")
        
        # get residuals for bootstrap confidence intervals
        self.residuals = nnx.data(np.asarray(y - self(X)))

        return self
    

    def forecast(self, data: pd.DataFrame, n_bootstrap: int = 1000, alpha: float = 0.05) -> pd.DataFrame:
        if self.residuals is None:
            raise ValueError("Model must be fitted before prediction.")
        if n_bootstrap < 0:
            raise ValueError("n_bootstrap must be non-negative.")

        # create lags of features for forecasting
        X = pd.concat([data.shift(lag) for lag in range(self.lags + 1)], axis=1).dropna().values
        if len(X) == 0:
            raise ValueError("Data too short to build lagged forecasting features.")
        X = jnp.asarray(X[-1:, :], dtype=jnp.float32)
        # predict next values
        preds = np.asarray(self(X))[0]

        if n_bootstrap == 0:
            lower = preds
            upper = preds
        else:
            # Bootstrap confidence intervals
            # For each horizon, independently resample residuals to build confidence band
            bootstrap_forecasts = np.zeros((n_bootstrap, len(preds)), dtype=float)
            for b in range(n_bootstrap):
                # For each horizon, resample a residual independently
                for h_idx in range(len(preds)):
                    idx = np.random.randint(0, self.residuals.shape[0])
                    resampled_residual = self.residuals[idx, h_idx]
                    bootstrap_forecasts[b, h_idx] = preds[h_idx] + resampled_residual

            upper = np.percentile(bootstrap_forecasts, 100*(1-alpha/2), axis=0).flatten()
            lower = np.percentile(bootstrap_forecasts, 100*alpha/2, axis=0).flatten()

        last_index = data.index[-1]
        idx = build_forecast_index(last_index, self.horizons)

        return pd.DataFrame({"forecast": preds, "lower": lower, "upper": upper}, index=idx)
