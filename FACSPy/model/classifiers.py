import anndata as ad
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import contextlib
from typing import Optional, Union, Literal
from ..utils import create_gate_lut, find_parents_recursively

implemented_estimators = ["DecisionTree", "RandomForest"]

class BaseFACSPyClassifier:

    def __init__(self):
        pass

    def fit(self, X, y, *args, **kwargs):
        return self.estimator.fit(X, y, *args, **kwargs)
    
    def predict(self, X):
        return self.estimator.predict(X)

    def classes_(self):
        if self.estimator:
            return self.estimator.classes_
    
    def set_params(self, **params):
        if not params:
            return self

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError("Parameter you are trying to set on the Classifier does not exist")
                
        self.estimator = self._generate_estimator()
        return self       

class RandomForest(BaseFACSPyClassifier, BaseEstimator):

    def __init__(self,
                 *,
                 base_estimator = RandomForestClassifier(),
                 criterion: str = "gini",
                 max_depth: str|int = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 min_weight_fraction_leaf: float = 0.0,
                 max_features: str|None|int = "sqrt",
                 max_leaf_nodes = None,
                 min_impurity_decrease: float = 0.0,
                 bootstrap: bool = True,
                 oob_score: bool = False,
                 n_jobs: int = None,
                 random_state: int = None,
                 verbose: int|bool = 0,
                 warm_start: bool = False,
                 class_weight = None,
                 ccp_alpha: float = 0.0,
                 max_samples = None,
                 **kwargs
                ) -> None:
        
        self._estimator_type = "classifier"
        self.base_estimator = base_estimator
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples

        self.estimator = self._generate_estimator()
    
    def _generate_estimator(self):
        return RandomForestClassifier(
                n_estimators = self.n_estimators,
                criterion = self.criterion,
                max_depth = self.max_depth,
                min_samples_split = self.min_samples_split,
                min_samples_leaf = self.min_samples_leaf,
                min_weight_fraction_leaf = self.min_weight_fraction_leaf,
                max_features = self.max_features,
                max_leaf_nodes = self.max_leaf_nodes,
                min_impurity_decrease = self.min_impurity_decrease,
                bootstrap = self.bootstrap,
                oob_score = self.oob_score,
                n_jobs = self.n_jobs,
                random_state = self.random_state,
                verbose = self.verbose,
                warm_start = self.warm_start,
                class_weight = self.class_weight,
                ccp_alpha = self.ccp_alpha,
                max_samples = self.max_samples
            )


class DecisionTree(BaseEstimator, BaseFACSPyClassifier):

    def __init__(self,
                 *,
                 base_estimator = DecisionTreeClassifier(),
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 class_weight=None,
                 ccp_alpha=0.0):
        
        self._estimator_type = "classifier"
        self.base_estimator = base_estimator
        self.criterion = criterion
        self.splitter = splitter
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.estimator = self._generate_estimator()

    def _generate_estimator(self):
        return DecisionTreeClassifier(
            criterion = self.criterion,
            splitter = self.splitter,
            max_depth = self.max_depth,
            min_samples_split = self.min_samples_split,
            min_samples_leaf = self.min_samples_leaf,
            min_weight_fraction_leaf = self.min_weight_fraction_leaf,
            max_features = self.max_features,
            random_state = self.random_state,
            max_leaf_nodes = self.max_leaf_nodes,
            min_impurity_decrease = self.min_impurity_decrease,
            class_weight = self.class_weight,
            ccp_alpha = self.ccp_alpha
        )
