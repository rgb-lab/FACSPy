from anndata import AnnData
import numpy as np
from typing import Optional, Union, Literal
from .classifiers import DecisionTree, RandomForest
from sklearn.model_selection import train_test_split
import contextlib
from ..utils import create_gate_lut, find_parents_recursively
from ..exceptions.exceptions import ClassifierNotImplementedError
from .classifiers import implemented_estimators
from scipy.sparse import lil_matrix, csr_matrix



class supervisedGating:

    def __init__(self,
                 adata: AnnData,
                 wsp_group: str,
                 estimator = Literal["DecisionTree", "RandomForest"]):
        
        self.train_sets = adata.uns["train_sets"][wsp_group]
        self._estimator = estimator
        self.adata = adata
        self.wsp_group = wsp_group
        
        self.classifiers = {
            gate_to_train: self._select_classifier(estimator)
            for gate_to_train in self.train_sets
        }

    def tune_hyperparameters(self):

        raise NotImplementedError("Hyperparameter tuning is currently not supported. :(")

    def train(self,
              **kwargs):
        
        for gate_to_train in self.train_sets:
            X_train, X_test, y_train, y_test = self.prepare_training_data(samples = self.train_sets[gate_to_train]["samples"],
                                                                          gate_columns = self.train_sets[gate_to_train]["training_columns"],
                                                                          **kwargs)

            self.classifiers[gate_to_train].fit(X_train, y_train)
            # TODO: print some logging message... maybe even progressbar
            # TODO: update stats, accuracy and such.

    def gate_dataset(self):
        for gate_to_train in self.train_sets:
            gate_indices = self.find_gate_indices(self.train_sets[gate_to_train]["training_columns"])
            self.fill_gates(gate_to_train,
                            gate_indices)

    def find_gate_indices(self,
                          gate_columns):
        return [self.adata.uns["gating_cols"].get_loc(gate) for gate in gate_columns]

    def fill_gates(self) -> None:
        self.adata.obsm["gating"] = self.adata.obsm["gating"].tolil()
        for gate_to_train in self.train_sets:
            non_gated_samples = [sample for sample in self.adata.obs["file_name"].unique()
                                if sample not in self.train_sets[gate_to_train]]
            gate_indices = self.find_gate_indices(gate_columns = self.train_sets[gate_to_train]["training_columns"])
            for sample in non_gated_samples:
                sample_view = self.adata[self.adata.obs["file_name"] == sample,:]
                first_sample_index = self.adata.obs_names.get_loc(sample_view.obs_names[0])
                sample_shape = sample_view.shape[0]
                predictions: np.ndarray = self.classifiers[gate_to_train].predict(sample_view.layers["compensated"])
                self.adata.obsm["gating"][
                    first_sample_index : first_sample_index + sample_shape,
                    gate_indices,
                ] = lil_matrix(predictions,
                               dtype=bool)

        self.adata.obsm["gating"] = self.adata.obsm["gating"].tocsr()

    def get_dataset(self):
        return self.adata

    def subset_anndata(self,
                       samples):
        return self.adata[self.adata.obs["file_name"].isin(samples),:]
    

    def prepare_training_data(self,
                              samples: list[str],
                              gate_columns: list[str],
                              test_size: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
             
        adata_subset = self.subset_anndata(samples)
        assert adata_subset.is_view ##TODO: delete later
        gate_indices = self.find_gate_indices(gate_columns)
        X = adata_subset.layers["compensated"]
        y = adata_subset.obsm["gating"][:, gate_indices].toarray()
        assert self.y_identities_correct(y = y,
                                         gates = adata_subset.obsm["gating"].toarray(),
                                         gate_indices = gate_indices)
        
        (X_train,
         X_test,
         y_train,
         y_test) = train_test_split(X, y, test_size = test_size)
        
        return X_train, X_test, y_train, y_test

    def y_identities_correct(self,
                             y: np.ndarray,
                             gates: np.ndarray,
                             gate_indices: list[int]) -> bool:
        return all(
            np.array_equal(
                np.array(y[:, [i]]).ravel(),
                gates[:, gate_indices[i]].ravel(),
            )
            for i in range(y.shape[1])
        )
    
    def _select_classifier(self,
                           estimator: Literal["DecisionTree", "RandomForest"]):
        if estimator == "DecisionTree":
            return DecisionTree()
        if estimator == "RandomForest":
            return RandomForest()
        else:
            raise ClassifierNotImplementedError(estimator, implemented_estimators)

    @classmethod
    def setup_anndata(cls,
                      dataset: AnnData,
                      wsp_group: Optional[str],
                      training_samples: Union[list[str],
                                              Literal["all_gated"]] = "all_gated"
                     ) -> None:
        
        workspace_subset: dict[str, dict] = dataset.uns["workspace"][wsp_group] 

        gate_lut = create_gate_lut(workspace_subset)

        if training_samples == "all_gated":
            training_samples = [
                sample for sample in workspace_subset if
                workspace_subset[sample]["gates"]
            ]
        else:
            training_samples = training_samples
        
        training_gate_paths = {
            gate_lut[sample][gate]["full_gate_path"]: gate_lut[sample][gate]["dimensions"]
            for sample in training_samples
            for gate in gate_lut[sample].keys()
        }
        
        reverse_lut = {}
        for gate in training_gate_paths:
            gate_name = gate.split("/")[-1]
            parents = [parent for parent in find_parents_recursively(gate) if parent != "root"]
            reverse_lut[gate] = {"dimensions": training_gate_paths[gate],
                                    "samples": [sample for sample in gate_lut.keys() if gate_name in gate_lut[sample].keys()],
                                    "training_columns": parents + [gate],
                                    "parents": parents}

        for gate in list(reverse_lut.keys()):
            with contextlib.suppress(KeyError): ## key errors are expected since we remove keys along the way
                if any(
                    k in reverse_lut
                    for k in reverse_lut[gate]["training_columns"]
                ):
                    for k in reverse_lut[gate]["training_columns"]:
                        if k == gate:
                            continue
                        del reverse_lut[k]

        for gate in list(reverse_lut.keys()):
            with contextlib.suppress(KeyError): ## key errors are expected since we remove keys along the way
                parents = reverse_lut[gate]["parents"]
                other_gates = list(reverse_lut.keys())
                other_gates.remove(gate)
                for other_gate in other_gates:

                    #if reverse_lut[other_gate]["dimensions"] == dimensions and reverse_lut[other_gate]["parents"] == parents:
                    if reverse_lut[other_gate]["parents"] == parents:
                        reverse_lut[gate]["training_columns"] += [other_gate]
                        reverse_lut[gate]["samples"] += reverse_lut[other_gate]["samples"]
                        reverse_lut.pop(other_gate)

        for gate in list(reverse_lut.keys()):
            reverse_lut[gate]["samples"] = set(reverse_lut[gate]["samples"])

        if "train_sets" not in dataset.uns.keys():
            dataset.uns["train_sets"] = {}
        dataset.uns["train_sets"][wsp_group] = reverse_lut
        
        return