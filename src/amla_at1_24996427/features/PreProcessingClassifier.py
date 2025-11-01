import numpy as np
import pandas as pd
from typing import Iterable, Dict, Tuple, Optional, List

from sklearn.base import BaseEstimator, TransformerMixin     
from sklearn.impute import SimpleImputer                     
from sklearn.preprocessing import StandardScaler             
from sklearn.pipeline import Pipeline     

from amla_at1_24996427.models.PerformanceMetrics import *

# Frozen base feature set (14)
FINAL_FEATURES = [
    "soil_temperature_28_to_100cm","diffuse_radiation","soil_moisture_28_to_100cm","soil_moisture_0_to_7cm",
    "wind_speed_10m","soil_temperature_100_to_255cm","soil_moisture_100_to_255cm","global_tilted_irradiance",
    "pressure_msl","relative_humidity_2m","cloud_cover","sunshine_duration","vapour_pressure_deficit","cloud_cover_high"
]

# Fast vs slow variables for lags/rollings
FAST_VARS = [
    "relative_humidity_2m","cloud_cover","cloud_cover_high","pressure_msl",
    "wind_speed_10m","vapour_pressure_deficit","diffuse_radiation",
    "sunshine_duration","global_tilted_irradiance"
]
SLOW_VARS = [
    "soil_temperature_28_to_100cm","soil_temperature_100_to_255cm",
    "soil_moisture_0_to_7cm","soil_moisture_28_to_100cm","soil_moisture_100_to_255cm"
]

# Train-only anomaly variables (help drift control)
ANOMALY_VARS = [
    "relative_humidity_2m","cloud_cover","pressure_msl",
    "diffuse_radiation","global_tilted_irradiance","sunshine_duration",
    "soil_temperature_28_to_100cm","soil_temperature_100_to_255cm",
    "soil_moisture_0_to_7cm","soil_moisture_28_to_100cm","soil_moisture_100_to_255cm"
]

# Skewed positives to log1p
LOG1P_VARS = ("diffuse_radiation","sunshine_duration","global_tilted_irradiance")

# Physical bounds for bounded features
PHYS_BOUNDS = {"cloud_cover": (0, 100), "cloud_cover_high": (0, 100)}

# FINAL_FEATURES_PLUS = ['soil_temperature_28_to_100cm', 'diffuse_radiation', 'soil_moisture_28_to_100cm', 
#                        'soil_moisture_0_to_7cm', 'wind_speed_10m', 'soil_temperature_100_to_255cm', 
#                        'soil_moisture_100_to_255cm', 'global_tilted_irradiance', 'pressure_msl', 
#                        'relative_humidity_2m', 'cloud_cover', 'sunshine_duration', 'vapour_pressure_deficit', 
#                        'cloud_cover_high', 'doy_sin', 'doy_cos', 'relative_humidity_2m_anom', 'cloud_cover_anom', 
#                        'pressure_msl_anom', 'diffuse_radiation_anom', 'global_tilted_irradiance_anom', 
#                        'sunshine_duration_anom', 'soil_temperature_28_to_100cm_anom', 'soil_temperature_100_to_255cm_anom', 
#                        'soil_moisture_0_to_7cm_anom', 'soil_moisture_28_to_100cm_anom', 'soil_moisture_100_to_255cm_anom', 
#                        'relative_humidity_2m_lag1', 'relative_humidity_2m_lag3', 'relative_humidity_2m_lag7', 
#                        'relative_humidity_2m_roll7', 'relative_humidity_2m_roll14', 'cloud_cover_lag1', 
#                        'cloud_cover_lag3', 'cloud_cover_lag7', 'cloud_cover_roll7', 'cloud_cover_roll14', 
#                        'pressure_msl_lag1', 'pressure_msl_lag3', 'pressure_msl_lag7', 'pressure_msl_roll7', 
#                        'pressure_msl_roll14', 'diffuse_radiation_lag1', 'diffuse_radiation_lag3', 
#                        'diffuse_radiation_lag7', 'diffuse_radiation_roll7', 'diffuse_radiation_roll14', 
#                        'global_tilted_irradiance_lag1', 'global_tilted_irradiance_lag3', 'global_tilted_irradiance_lag7', 
#                        'global_tilted_irradiance_roll7', 'global_tilted_irradiance_roll14', 'soil_temperature_28_to_100cm_lag7', 
#                        'soil_temperature_28_to_100cm_roll7', 'soil_temperature_28_to_100cm_roll14', 
#                        'soil_moisture_28_to_100cm_lag7', 'soil_moisture_28_to_100cm_roll7', 'soil_moisture_28_to_100cm_roll14', 
#                        'diffuse_radiation_log1p', 'sunshine_duration_log1p', 'global_tilted_irradiance_log1p']

FINAL_FEATURES_PLUS = ['soil_temperature_28_to_100cm', 'diffuse_radiation', 'soil_moisture_28_to_100cm', 
                       'soil_moisture_0_to_7cm', 'wind_speed_10m', 'soil_temperature_100_to_255cm', 
                       'soil_moisture_100_to_255cm', 'global_tilted_irradiance', 'pressure_msl', 
                       'relative_humidity_2m', 'cloud_cover', 'sunshine_duration', 'vapour_pressure_deficit', 
                       'cloud_cover_high', 'diffuse_radiation_lag1', 'wind_speed_10m_roll7', 
                       'pressure_msl_roll3', 'cloud_cover_high_roll3', 'relative_humidity_2m_anom', 
                       'soil_moisture_0_to_7cm_lag14']


class QuantileCapper(BaseEstimator, TransformerMixin):                      
    """Clip columns to [q_low, q_high] quantiles fitted on TRAIN."""
    def __init__(self, q_low:float=0.01, q_high:float=0.99):
        self.q_low, self.q_high = q_low, q_high
        self.caps_: Optional[Dict[str, Tuple[float, float]]] = None
        self.feature_names_in_: Optional[Iterable[str]] = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan)
        self.feature_names_in_ = list(X.columns)
        self.caps_ = {}
        for c in self.feature_names_in_:
            lo, hi = np.nanquantile(X[c].astype(float), [self.q_low, self.q_high])
            self.caps_[c] = (float(lo), float(hi))
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_).replace([np.inf, -np.inf], np.nan)
        for c, (lo, hi) in self.caps_.items():
            X[c] = X[c].clip(lo, hi)
        return X  # keep as DataFrame here; pipelines can convert later


def make_preprocessor_linear(q_low=0.01, q_high=0.99):  # <<< TestPyPI (also for regression)
    """Capping + median impute + StandardScaler."""
    return Pipeline([
        ("cap", QuantileCapper(q_low, q_high)),
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
    ])

class DOYEncoder(BaseEstimator, TransformerMixin): 
    """Add cyclical day-of-year features (sin/cos) from a datetime column."""
    def __init__(self, date_col: str = "date_local"):
        self.date_col = date_col
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        doy = X[self.date_col].dt.dayofyear.astype(int)
        ang = 2*np.pi*(doy/365.25)
        X["doy_sin"] = np.sin(ang)
        X["doy_cos"] = np.cos(ang)
        self.feature_names_out_ = list(X.columns)
        return X

class FeatureEngineer(BaseEstimator, TransformerMixin):  
    def __init__(
        self,
        date_col: str,
        base_features: List[str],
        fast_vars: List[str],
        slow_vars: List[str],
        lags_fast: Tuple[int, ...] = (1,3,7),
        lags_slow: Tuple[int, ...] = (7,14),
        roll_fast: Tuple[int, ...] = (3,7,14),
        roll_slow: Tuple[int, ...] = (7,14),
        anomaly_vars: Optional[List[str]] = None,
        log1p_vars: Tuple[str, ...] = (),
        physical_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.date_col = date_col
        self.base_features = base_features
        self.fast_vars = fast_vars
        self.slow_vars = slow_vars
        self.lags_fast = lags_fast
        self.lags_slow = lags_slow
        self.roll_fast = roll_fast
        self.roll_slow = roll_slow
        self.anomaly_vars = anomaly_vars or []
        self.log1p_vars = log1p_vars
        self.physical_bounds = physical_bounds or {}
        self.climatology_: Dict[str, pd.Series] = {}
        self.feature_names_out_: List[str] = []

    def _sorted(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(self.date_col).reset_index(drop=True)

    # F.4 — seasonal anomalies (learned on TRAIN)
    def fit(self, X: pd.DataFrame, y=None):
        X = self._sorted(X)
        if self.anomaly_vars:
            doy = X[self.date_col].dt.dayofyear.astype(int)
            for c in self.anomaly_vars:
                if c in X.columns:
                    self.climatology_[c] = X.groupby(doy)[c].mean()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._sorted(X).copy()
        out = pd.DataFrame(index=X.index)

        # carry base (+ DOY) and apply F.5 safety bounds first
        for c in self.base_features:
            if c in X.columns:
                out[c] = X[c]

        for c, (lo, hi) in self.physical_bounds.items():
            if c in X.columns:
                X[c] = X[c].clip(lo, hi)
                out[c] = X[c]

        # F.2 — Lags (fast & slow)
        for c in self.fast_vars:
            if c in X.columns:
                for L in self.lags_fast:
                    out[f"{c}_lag{L}"] = X[c].shift(L)
        for c in self.slow_vars:
            if c in X.columns:
                for L in self.lags_slow:
                    out[f"{c}_lag{L}"] = X[c].shift(L)

        # F.3 — Rolling means, windows close at t-1
        for c in self.fast_vars:
            if c in X.columns:
                s = X[c].shift(1)
                for W in self.roll_fast:
                    out[f"{c}_roll{W}"] = s.rolling(W, min_periods=1).mean()
        for c in self.slow_vars:
            if c in X.columns:
                s = X[c].shift(1)
                for W in self.roll_slow:
                    out[f"{c}_roll{W}"] = s.rolling(W, min_periods=1).mean()

        # F.4 — seasonal anomalies (value - climatology(DOY))
        if self.climatology_:
            doy = X[self.date_col].dt.dayofyear.astype(int)
            for c, clim in self.climatology_.items():
                if c in X.columns:
                    base = doy.map(lambda d: clim.get(d, clim.get(365, np.nan)))
                    out[f"{c}_anom"] = X[c].astype(float) - base.values

        # F.5 — log1p for skewed positives
        for c in self.log1p_vars:
            if c in X.columns:
                out[f"{c}_log1p"] = np.log1p(np.clip(X[c].astype(float), a_min=0, a_max=None))

        self.feature_names_out_ = list(out.columns)
        return out

    def get_feature_names_out(self) -> List[str]:
        return self.feature_names_out_

