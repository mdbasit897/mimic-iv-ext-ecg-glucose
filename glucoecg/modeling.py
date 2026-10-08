"""Model definitions shared by the baseline scripts (hyperparameters as in the v1.0 paper)."""

import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import assert_no_leak

XGB_PARAMS = dict(
    n_estimators=1000, learning_rate=0.05, max_depth=6, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    objective="reg:squarederror", eval_metric="mae", early_stopping_rounds=30,
    tree_method="hist", verbosity=0,
)


def matrix(df, cols):
    assert_no_leak(cols)
    return df[cols].astype(float).to_numpy()


def fit_xgb(X, y, train, val, weights=None, seed: int = 42, n_jobs: int = -1):
    """Fit XGBoost on rows ``train`` with early stopping on rows ``val``; return the model."""
    model = xgb.XGBRegressor(**XGB_PARAMS, random_state=seed, n_jobs=n_jobs)
    model.fit(X[train], y[train], sample_weight=None if weights is None else weights[train],
              eval_set=[(X[val], y[val])], verbose=False)
    return model


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each column to its training [low, high] quantiles (keeps linear models in range)."""

    def __init__(self, low: float = 0.005, high: float = 0.995):
        self.low, self.high = low, high

    def fit(self, X, y=None):
        self.lo_ = np.nanquantile(X, self.low, axis=0)
        self.hi_ = np.nanquantile(X, self.high, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lo_, self.hi_)


def fit_ridge(X, y, train):
    """Median imputation + Ridge(alpha=10), unweighted, as in v1.0. Clipping to the training
    0.5-99.5% range and scaling were added: unclipped, out-of-range CAUTION rows (non-ICU lags
    spanning years) give extrapolated predictions in the thousands of mg/dL."""
    model = make_pipeline(SimpleImputer(strategy="median"), QuantileClipper(), StandardScaler(),
                          Ridge(alpha=10.0))
    model.fit(X[train], y[train])
    return model


def fit_random_forest(X, y, train, seed: int = 42, n_jobs: int = -1):
    """Median imputation + 300 trees, depth 12, min leaf 10, unweighted, as in v1.0."""
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=10,
                              n_jobs=n_jobs, random_state=seed),
    )
    model.fit(X[train], y[train])
    return model


def predict(model, X) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float)


def train_feature_set(df, feature_set: str, train_subset: str = "iu", seed: int = 42, n_jobs: int = -1):
    """
    Train XGBoost on one of features.FEATURE_SETS using TRAIN rows of ``train_subset``
    ('iu' or 'all'), early-stopped on VALIDATION rows of the same subset, with the
    paper's usability weights. Returns (model, predictions for every row of df).
    """
    from .features import FEATURE_SETS, TARGET, train_mask, usability_weights, val_mask

    cols = FEATURE_SETS[feature_set]
    X = matrix(df, cols)
    y = df[TARGET].to_numpy(float)
    model = fit_xgb(X, y, train_mask(df, train_subset), val_mask(df, train_subset),
                    usability_weights(df), seed=seed, n_jobs=n_jobs)
    return model, predict(model, X)


AGE_BINS = [0, 45, 60, 75, 120]
AGE_LABELS = ["<45", "45-59", "60-74", ">=75"]
