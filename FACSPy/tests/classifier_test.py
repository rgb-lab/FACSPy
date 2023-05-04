import pytest

from ..ml_gating.classifiers import FACSPyRandomForest, FACSPyDecisionTree, FACSPyExtraTree, FACSPyExtraTrees
from sklearn.ensemble import RandomForestRegressor
from ..ml_gating.training import MLGatingTrainingData
from ..exceptions.exceptions import (WrongFACSPyHyperparameterMethod,
                                     WrongFACSPyHyperparameterTuningDepth,
                                     HyperparameterTuningCustomError)
import pickle
import os

# TODO: add hyperparameter tuning tests

@pytest.fixture
def mock_train_set():
    with open(f"{os.path.dirname(__file__)}/dummy_MLGatingTrainingSet.pkl", "rb") as file:
        train_set = pickle.load(file)
    return train_set

def test_classifier_check_input_estimator(mock_train_set):
    pass #not possible as inheriting class is called

def test_classifier_input_hyperparameter_tuning_bool(mock_train_set):
    with pytest.raises(TypeError):
        _ = FACSPyRandomForest(training_data = mock_train_set,
                               hyperparameter_tuning = "some_string")
    with pytest.raises(TypeError):
        _ = FACSPyRandomForest(training_data = mock_train_set,
                               hyperparameter_tuning = [True])
    with pytest.raises(TypeError):
        _ = FACSPyRandomForest(training_data = mock_train_set,
                               hyperparameter_tuning = {True})

def test_classifier_input_hyperparameter_tuning_method(mock_train_set):
    with pytest.raises(WrongFACSPyHyperparameterMethod):
        _ = FACSPyRandomForest(training_data = mock_train_set,
                               hyperparameter_tuning = True,
                               hyperparameter_tuning_method = "RandomSearch")
    with pytest.raises(WrongFACSPyHyperparameterMethod):
        _ = FACSPyRandomForest(training_data = mock_train_set,
                               hyperparameter_tuning = True,
                               hyperparameter_tuning_method = "GridSearchCV")
    with pytest.raises(WrongFACSPyHyperparameterMethod):
        _ = FACSPyRandomForest(training_data = mock_train_set,
                               hyperparameter_tuning = True,
                               hyperparameter_tuning_method = ["GridSearch"])

def test_classifier_input_hyperparameter_tuning_depth(mock_train_set):
    with pytest.raises(WrongFACSPyHyperparameterTuningDepth):
        _ = FACSPyRandomForest(training_data = mock_train_set,
                                hyperparameter_tuning = True,
                                hyperparameter_tuning_method = "GridSearch",
                                hyperparameter_tuning_depth = "shallow")
    with pytest.raises(WrongFACSPyHyperparameterTuningDepth):
        _ = FACSPyRandomForest(training_data = mock_train_set,
                                hyperparameter_tuning = True,
                                hyperparameter_tuning_method = "GridSearch",
                                hyperparameter_tuning_depth = 2)
    with pytest.raises(WrongFACSPyHyperparameterTuningDepth):
        _ = FACSPyRandomForest(training_data = mock_train_set,
                                hyperparameter_tuning = True,
                                hyperparameter_tuning_method = "GridSearch",
                                hyperparameter_tuning_depth = ["deep"])

def test_classifier_input_hyperparameter_tuning_custom(mock_train_set):
    with pytest.raises(HyperparameterTuningCustomError):
        _ = FACSPyRandomForest(training_data = mock_train_set,
                               hyperparameter_tuning = True,
                               hyperparameter_tuning_method = "GridSearch",
                               hyperparameter_tuning_depth = "custom",
                               hyperparameter_tuning_grid = {})
    with pytest.raises(HyperparameterTuningCustomError):
        _ = FACSPyRandomForest(training_data = mock_train_set,
                               hyperparameter_tuning = True,
                               hyperparameter_tuning_method = "GridSearch",
                               hyperparameter_tuning_depth = "custom",
                               hyperparameter_tuning_grid = None)

def test_classifier_training_random_forest(mock_train_set):
    a = FACSPyRandomForest(training_data = mock_train_set,
                           n_estimators = 50,
                           criterion = "entropy",
                           **{"max_features": "log2", "bootstrap": False})
    assert a.model.n_estimators == 50
    assert a.model.criterion == "entropy"
    assert a.model.max_features == "log2"
    assert a.model.bootstrap == False
    with pytest.raises(AttributeError):
        _ = a.hyperparameters

def test_classifier_training_extratrees(mock_train_set):
    a = FACSPyExtraTrees(training_data = mock_train_set,
                         n_estimators = 50,
                         criterion = "entropy",
                         **{"max_features": "log2", "bootstrap": False})
    assert a.model.n_estimators == 50
    assert a.model.criterion == "entropy"
    assert a.model.max_features == "log2"
    assert a.model.bootstrap == False
    with pytest.raises(AttributeError):
        _ = a.hyperparameters

def test_classifier_training_extra_tree(mock_train_set):
    a = FACSPyExtraTree(training_data = mock_train_set,
                           criterion = "entropy",
                           **{"max_features": "log2"})
    assert a.model.criterion == "entropy"
    assert a.model.max_features == "log2"
    with pytest.raises(AttributeError):
        _ = a.hyperparameters

def test_classifier_training_decision_tree(mock_train_set):
    a = FACSPyDecisionTree(training_data = mock_train_set,
                           criterion = "entropy",
                           **{"max_features": "log2"})
    assert a.model.criterion == "entropy"
    assert a.model.max_features == "log2"
    with pytest.raises(AttributeError):
        _ = a.hyperparameters
    