from sklearn.preprocessing import (MinMaxScaler,
                                   PowerTransformer,
                                   QuantileTransformer,
                                   RobustScaler,
                                   StandardScaler,
                                   MaxAbsScaler)

from sklearn.base import TransformerMixin
from typing import Literal, Optional
from ..exceptions._exceptions import InvalidScalingError
import numpy as np
implemented_transformers = ["PowerTransformer", "QuantileTransformer"]
implemented_scalers = [
    "RobustScaler", "MaxAbsScaler", "StandardScaler", "MinMaxScaler"
]


class QuantileCapper():
    """Class to use quantile capping according to the sklearn syntax

    Parameters
    ----------
    lower_cap
        quantile that has to be set. Accepts lower quantile
    upper_cap
        upper quantile. If None, is calculated symmetrically from lower cap

    Examples
    --------
    >>> q_capper = QuantileCapper(lower_cap = 0.05)
    >>> q_capper.fit()
    >>> X: np.ndarray = q_capper.transform(X)
    """

    def __init__(self,
                 lower_cap: float,
                 upper_cap: Optional[float] = None):
        self.lower_limit = lower_cap
        self.upper_limit = 1 - lower_cap if upper_cap is None else upper_cap

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def fit(self,
            X: np.ndarray) -> None:
        self.lower_quantiles = np.quantile(X, self.lower_limit, axis = 0)
        self.upper_quantiles = np.quantile(X, self.upper_limit, axis = 0)

    def transform(self,
                  X: np.ndarray):
        return np.clip(X, self.lower_quantiles, self.upper_quantiles)

    def fit_transform(self,
                      X: np.ndarray):
        self.fit(X)
        return self.transform(X)


def scale_data(X: np.ndarray,
               scaler: Optional[
                   Literal["RobustScaler", "MaxAbsScaler",
                           "StandardScaler", "MinMaxScaler"]
               ]
               ) -> tuple[np.ndarray, TransformerMixin]:
    if scaler == "RobustScaler":
        transformer = RobustScaler()
    elif scaler == "MaxAbsScaler":
        transformer = MaxAbsScaler()
    elif scaler == "StandardScaler":
        transformer = StandardScaler()
    elif scaler == "MinMaxScaler":
        transformer = MinMaxScaler()
    else:
        raise InvalidScalingError(scaler)

    X = transformer.fit_transform(X)
    return X, transformer


def transform_data(X: np.ndarray,
                   transformer: Literal[
                       "PowerTransformer", "QuantileTransformer"
                   ]
                   ) -> tuple[np.ndarray, TransformerMixin]:
    if transformer == "PowerTransformer":
        transform_algorithm = PowerTransformer()
    elif transformer == "QuantileTransformer":
        transform_algorithm = QuantileTransformer()
    else:
        raise NotImplementedError()

    X = transform_algorithm.fit_transform(X)

    return X, transform_algorithm


def cap_data(X: np.ndarray,
             quantile_cap: float) -> tuple[np.ndarray, QuantileCapper]:
    transformer = QuantileCapper(quantile_cap)
    X = transformer.fit_transform(X)
    return X, transformer
