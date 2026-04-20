import pandas as pd
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax 

class BaseNN(nnx.Module):

    def __init__(
        self,
        din: int,
        dmid: tuple[int, ...] = (32, 32),
        rngs: nnx.Rngs = nnx.Rngs(0),
        learning_rate: float = 0.001,
        epochs: int = 1000,
        lags: int = 1,
    ):
        self.rngs = rngs
        self.lags = lags
        self.epochs = epochs
        self.residuals = nnx.data(None)
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
        # define optimizer
        self.optimizer = nnx.Optimizer(self, optax.adam(learning_rate), wrt=nnx.Param)

    def __call__(self, x):
        return self.layers(x)
    
    def _train_step(self, X, y):
        def loss_fn(model):
            preds = model(X)
            return jnp.mean((preds - y) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(self)
        self.optimizer.update(self, grads)
        return loss
    
    def fit(self, data: pd.DataFrame, target: str):
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data.")
        if len(data) - self.lags - 12 <= 0:
            raise ValueError("Data too short for requested lags and 12-step target.")

        # create lags of features
        X = pd.concat([data.shift(lag) for lag in range(self.lags + 1)], axis=1).dropna().values
        X = X[:-12] # align with y after creating multi-step target
        # create target array for multi-step forecasting
        y = pd.concat([data[target].shift(-h) for h in [1, 3, 6, 12]], axis=1).dropna().values
        y = y[self.lags:]  # align with X after dropping lags
        X = jnp.asarray(X, dtype=jnp.float32)
        y = jnp.asarray(y, dtype=jnp.float32)

        # training loop
        for epoch in range(self.epochs):
            loss = self._train_step(X, y)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        # get residuals for bootstrap confidence intervals
        self.residuals = nnx.data(np.asarray(y - self(X)))

        return self
    

    def forecast(self, data: pd.DataFrame, n_bootstrap: int = 1000, alpha: float = 0.05) -> pd.DataFrame:
        if self.residuals is None:
            raise ValueError("Model must be fitted before prediction.")
        if n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be positive.")

        # create lags of features for forecasting
        X = pd.concat([data.shift(lag) for lag in range(self.lags + 1)], axis=1).dropna().values
        if len(X) == 0:
            raise ValueError("Data too short to build lagged forecasting features.")
        X = jnp.asarray(X[-1:, :], dtype=jnp.float32)
        # predict next values
        preds = np.asarray(self(X))[0]

        # Bootstrap confidence intervals
        bootstrap_forecasts = np.zeros((n_bootstrap, 4), dtype=float)
        for b in range(n_bootstrap):
            # resample residuals with replacement
            idx = np.random.randint(0, self.residuals.shape[0])
            resampled_residuals = self.residuals[idx]
            # generate bootstrap forecast path
            bootstrap_forecasts[b, :] = preds + resampled_residuals

        upper = np.percentile(bootstrap_forecasts, 100*(1-alpha/2), axis=0).flatten()
        lower = np.percentile(bootstrap_forecasts, 100*alpha/2, axis=0).flatten()

        # get index for future periods
        last_index = data.index[-1]
        idx = pd.Index([last_index + h for h in [1,3,6,12]])

        return pd.DataFrame({"forecast": preds, "lower": lower, "upper": upper}, index=idx)
