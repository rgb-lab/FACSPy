from anndata import AnnData
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import contextlib

from typing import Optional, Union, Literal

from .classifiers import DecisionTree, RandomForest, implemented_estimators
from ..utils import create_gate_lut, find_parents_recursively

from ..utils import (contains_only_fluo,
                     subset_gate,
                     get_idx_loc,
                     find_gate_indices)
from ..exceptions.exceptions import ClassifierNotImplementedError, ParentGateNotFoundError, AnnDataSetupError

"""
TODO: testing of classifier
append data to adata.uns["train_sets"]
gating strategy plot

"""

class BaseGating:
    
    def __init__(self):
        pass

    def get_dataset(self):
        return self.adata

    def subset_anndata_by_sample(self,
                                 samples,
                                 adata: Optional[AnnData] = None,
                                 copy: bool = False):
        if not isinstance(samples, list):
            samples = [samples]
        if adata is not None:
            if copy:
                return adata[adata.obs["file_name"].isin(samples),:].copy()
            else:
                return adata[adata.obs["file_name"].isin(samples),:]
        if copy:
            return self.adata[self.adata.obs["file_name"].isin(samples),:].copy()
        else:
            return self.adata[self.adata.obs["file_name"].isin(samples),:]

    def add_gating_to_input_dataset(self,
                                    subset: AnnData,
                                    predictions: np.ndarray,
                                    gate_indices: list[int]) -> None:
        sample_indices = get_idx_loc(adata = self.adata,
                                     idx_to_loc = subset.obs_names)
        ### otherwise, broadcast error if multiple columns are indexed and sample_indices
        for i, gate_index in enumerate(gate_indices):
            self.adata.obsm["gating"][
                sample_indices,
                gate_index,
            ] = predictions[:, i]

class supervisedGating(BaseGating):

    def __init__(self,
                 adata: AnnData,
                 wsp_group: str,
                 estimator = Literal["DecisionTree", "RandomForest"]):
        if "train_sets" not in adata.uns:
            raise AnnDataSetupError()
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
            print(f"Gating gate {gate_to_train}")
            print("Preparing training data")
            X_train, X_test, y_train, y_test = self.prepare_training_data(samples = self.train_sets[gate_to_train]["samples"],
                                                                          gate_columns = self.train_sets[gate_to_train]["training_columns"],
                                                                          **kwargs)
            print("Fitting classifier...")
            self.classifiers[gate_to_train].fit(X_train, y_train)
            # TODO: print some logging message... maybe even progressbar
            # TODO: update stats, accuracy and such.


    def gate_dataset(self) -> None:
        #self.adata.obsm["gating"] = self.adata.obsm["gating"].tolil()
        self.adata.obsm["gating"] = self.adata.obsm["gating"].todense()
        for gate_to_train in self.train_sets:
            print(f"Gating {gate_to_train}")
            non_gated_samples = [sample for sample in self.adata.obs["file_name"].unique()
                                if sample not in self.train_sets[gate_to_train]]
            gate_indices = self.find_gate_indices(gate_columns = self.train_sets[gate_to_train]["training_columns"])
            for sample in non_gated_samples:
                print(f"Gating sample {sample}...")
                sample_view = self.subset_anndata_by_sample(samples = sample, copy = False)
                predictions: np.ndarray = self.classifiers[gate_to_train].predict(sample_view.layers["compensated"])
                self.add_gating_to_input_dataset(sample_view,
                                                 predictions,
                                                 gate_indices)

        self.adata.obsm["gating"] = csr_matrix(self.adata.obsm["gating"])



    def prepare_training_data(self,
                              samples: list[str],
                              gate_columns: list[str],
                              test_size: float = 0.1) -> tuple[np.ndarray,
                                                               np.ndarray,
                                                               np.ndarray,
                                                               np.ndarray]:
             
        adata_subset = self.subset_anndata_by_sample(samples)
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
            reverse_lut[gate]["samples"] = list(set(reverse_lut[gate]["samples"]))

        if "train_sets" not in dataset.uns.keys():
            dataset.uns["train_sets"] = {}
        dataset.uns["train_sets"][wsp_group] = reverse_lut
        
        return
    

class unsupervisedGating(BaseGating):

    def __init__(self,
                 adata: AnnData,
                 gating_strategy: dict,
                 clustering_algorithm: Literal["leiden", "FlowSOM"],
                 cluster_key: str = None) -> None:
        self.gating_strategy = gating_strategy
        self.clustering_algorithm = clustering_algorithm
        self.adata = adata
        self.cluster_key = cluster_key or "clusters"
        assert contains_only_fluo(self.adata)

    def population_is_already_a_gate(self,
                                     parent_population) -> bool:
        return parent_population in [gate.split("/")[-1] for gate in self.adata.uns["gating_cols"]]

    def process_markers(self,
                        population: str) -> dict[str, list[Optional[str]]]:
        markers = self.gating_strategy[population][1]
        if not isinstance(markers, list):
            markers = [markers]
        marker_dict = {"up": [],
                       "down": []}
        for marker in markers:
            if "+" in marker:
                marker_dict["up"].append(marker.split("+")[0])
            elif "-" in marker:
                marker_dict["down"].append(marker.split("-")[0])
            else:
                marker_dict["up"].append(marker)
        return marker_dict

    def identify_population(self,
                            population: str) -> None:
        
        parent_population = self.gating_strategy[population][0]
        if not self.population_is_already_a_gate(parent_population):
            if parent_population in self.gating_strategy:
                self.identify_population(parent_population)
            else:
                raise ParentGateNotFoundError(parent_population)
        
        markers_of_interest = self.process_markers(population)
        
        ## this handles a weird case where this population already exists...
        ## should potentially throw an error?
        parent_gate_path = [gate_path for gate_path in self.adata.uns["gating_cols"]
                            if gate_path.endswith(parent_population)][0]
        population_gate_path = "/".join([parent_gate_path, population])
        
        if not self.population_is_already_a_gate(population):
            self.append_gate_column_to_adata(population_gate_path)

        gate_indices = find_gate_indices(self.adata, population_gate_path)
        
        if parent_population != "root":
            dataset = subset_gate(self.adata,
                                  gate = parent_population,
                                  as_view = True)
            assert dataset.is_view
        else:
            dataset = self.adata.copy()
        
        
        for sample in dataset.obs["file_name"].unique():
            print(f"Analyzing sample {sample}")
            subset = self.subset_anndata_by_sample(adata = dataset,
                                                   samples = sample,
                                                   copy = False)
            print("... preprocessing")
            subset = self.preprocess_dataset(subset)
            print("... clustering")
            if self.cluster_key not in subset.obs:
                subset = self.cluster_dataset(subset)
            
            cluster_vector = self.identify_clusters_of_interest(subset,
                                                                markers_of_interest)
            
            cell_types = self.map_cell_types_to_cluster(subset,
                                                        cluster_vector,
                                                        population)

            predictions = self.convert_cell_types_to_bool(cell_types)
    
            self.add_gating_to_input_dataset(subset,
                                             predictions,
                                             gate_indices) 
    
    def identify_populations(self):
        self.adata.X = self.adata.layers["transformed"]
        self.adata.obsm["gating"] = self.adata.obsm["gating"].todense()
        for population in self.gating_strategy:
            print(f"Population: {population}")
            self.identify_population(population)
        self.adata.obsm["gating"] = csr_matrix(self.adata.obsm["gating"])

    def convert_cell_types_to_bool(self,
                                   cell_types: list[str]) -> np.ndarray:
        return np.array(
            list(map(lambda x: x != "other", cell_types)),
            dtype=bool
        ).reshape((len(cell_types), 1))
        
    def map_cell_types_to_cluster(self,
                                  subset: AnnData,
                                  cluster_vector: pd.Index,
                                  population: str) -> list[str]:
        # TODO: map function...
        return [population if cluster in cluster_vector else "other" for cluster in subset.obs[self.cluster_key].to_list()]

    def append_gate_column_to_adata(self,
                                    gate_path) -> None:
        self.adata.uns["gating_cols"] = self.adata.uns["gating_cols"].append(pd.Index([gate_path]))
        empty_column = np.zeros(
                shape = (self.adata.obsm["gating"].shape[0],
                         1),
                dtype = bool
            )
        self.adata.obsm["gating"] = np.hstack([self.adata.obsm["gating"], empty_column])
        return
        
    def convert_markers_to_query_string(self,
                                        markers_of_interest: dict[str: list[Optional[str]]]) -> str:
        cutoff = str(np.arcsinh(1))
        up_markers = markers_of_interest["up"]
        down_markers = markers_of_interest["down"]
        query_strings = (
            [f"{marker} > {cutoff}" for marker in up_markers] +
            [f"{marker} < {cutoff}" for marker in down_markers]
        )
        return " & ".join(query_strings)
    
    def identify_clusters_of_interest(self,
                                      dataset: AnnData,
                                      markers_of_interest: dict[str: list[Optional[str]]]) -> list[str]:
        df = dataset.to_df(layer = "transformed")
        df[self.cluster_key] = dataset.obs[self.cluster_key].to_list()
        medians = df.groupby(self.cluster_key).median()
        cells_of_interest: pd.DataFrame = medians.query(self.convert_markers_to_query_string(markers_of_interest))
        return cells_of_interest.index

    def cluster_dataset(self,
                        dataset: AnnData) -> AnnData:
        if self.clustering_algorithm != "leiden":
            raise NotImplementedError("Please select 'leiden' :D")
        sc.tl.leiden(dataset, key_added = self.cluster_key)
        return dataset

    def preprocess_dataset(self,
                           subset: AnnData) -> AnnData:
        sc.pp.pca(subset)
        sc.pp.neighbors(subset)
        return subset
    
    @classmethod
    def setup_anndata(cls):
        pass

