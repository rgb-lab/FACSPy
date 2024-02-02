from anndata import AnnData

import numpy as np
import pandas as pd
import scanpy as sc

from scipy.sparse import csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted

from scipy.spatial import ConvexHull

from typing import Optional, Union, Literal

from ._classifiers import DecisionTree, RandomForest, implemented_estimators
from ._utils import (cap_data,
                    transform_data,
                    scale_data,
                    )

from .._utils import (_transform_gates_according_to_gate_transform,
                      _transform_vertices_according_to_gate_transform,
                      _find_parent_gate,
                      GATE_SEPARATOR)

from ..tools._leiden import leiden
from ..tools._phenograph import phenograph
from ..tools._parc import parc
from ..tools._flowsom import flowsom

from .._utils import (subset_gate,
                      get_idx_loc,
                      _find_gate_indices,
                      _create_gate_lut,
                      _find_parents_recursively,
                      _find_current_population,
                      _find_gate_path_of_gate,
                      _is_valid_filename,
                      _is_valid_sample_ID)

from ..exceptions._exceptions import (ClassifierNotImplementedError,
                                      ParentGateNotFoundError,
                                      AnnDataSetupError)

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

    def _subset_anndata_by_sample(self,
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

    def _add_gating_to_input_dataset(self,
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

    def _append_gate_column_to_adata(self,
                                     gate_path) -> None:
        self.adata.uns["gating_cols"] = self.adata.uns["gating_cols"].append(pd.Index([gate_path]))
        empty_column = np.zeros(
                shape = (self.adata.obsm["gating"].shape[0],
                         1),
                dtype = bool
            )
        self.adata.obsm["gating"] = np.hstack([self.adata.obsm["gating"], empty_column])
        return

class ManualGating(BaseGating):

    def __init__(self,
                 adata: AnnData,
                 gate_coordinates: np.ndarray,
                 x_channel: str,
                 y_channel: str,
                 data_origin: Literal["compensated", "transformed"],
                 parent_population: str,
                 gate_name: str,
                 sample_identifier: Optional[str] = None
                 ) -> None:
        self.adata: AnnData = adata
        
        self.gate_coordinates = self._preprocess_gate_coordinates(gate_coordinates)
        self._create_hull_from_gate_coordinates()
        
        self.subset = self._preprocess_adata(sample_identifier = sample_identifier,
                                             parent_population = parent_population)
        self._cells = self._extract_cells(self.subset,
                                          x_channel,
                                          y_channel,
                                          data_origin)
        self.parent_population = parent_population
        self.gate_name = gate_name
        self.gating_path = self._create_gate_path()

    def _preprocess_adata(self,
                          sample_identifier: Optional[str],
                          parent_population: str) -> AnnData:

        subset = subset_gate(self.adata,
                             gate = parent_population,
                             as_view = True)

        if sample_identifier is None:
            return subset

        if _is_valid_sample_ID(subset, sample_identifier):
            subset = subset[subset.obs["sample_ID"] == str(sample_identifier),:]
        elif _is_valid_filename(subset, sample_identifier):
            subset = subset[subset.obs["file_name"] == str(sample_identifier),:]
        else:
            raise ValueError(f"{sample_identifier} not found")
        return subset 

    def _create_gate_path(self):
        parent_path = _find_gate_path_of_gate(self.adata, self.parent_population)
        return GATE_SEPARATOR.join([parent_path, self.gate_name])

    def _preprocess_gate_coordinates(self,
                                     gate_coordinates: np.ndarray) -> np.ndarray:
        if gate_coordinates.shape[0] <= 2:
            # means we likely have a rectangle range and
            # cannot create a hull from that
            return np.array(
                np.meshgrid(gate_coordinates[:,0], gate_coordinates[:,1])
            ).T.reshape(-1,2)
        return gate_coordinates

    def _extract_cells(self,
                       adata: AnnData,
                       x_channel: str,
                       y_channel: str,
                       data_origin: Literal["compensated", "transformed"]) -> np.ndarray:
        return adata.to_df(layer = data_origin)[[x_channel, y_channel]].values
    
    def _create_hull_from_gate_coordinates(self):
        self.hull = ConvexHull(self.gate_coordinates)
        return
    
    def gate(self):
        self.adata.obsm["gating"] = self.adata.obsm["gating"].todense()
        gating_result = self._points_in_gate()
        
        self._append_gate_column_to_adata(self.gating_path)
        gate_index = _find_gate_indices(self.adata, self.gating_path)
        
        self._add_gating_to_input_dataset(self.subset,
                                          predictions = gating_result,
                                          gate_indices = gate_index)
        
        self.adata.obsm["gating"] = csr_matrix(self.adata.obsm["gating"])
        return

    def _points_in_gate(self,
                        tol: float = 1e-12) -> np.ndarray:
        gate_results: np.ndarray = np.all(
            self.hull.equations[:,:-1] @ self._cells.T + np.repeat(
                self.hull.equations[:,-1][None,:],
                len(self._cells), axis=0).T <= tol, 0
            )
        return gate_results.reshape(gate_results.shape[0], 1)

    


class supervisedGating(BaseGating):
    """
    Class to unify functionality for supervised gating approaches.

    The user inputs the anndata object containing the data as well 
    as the workspace group (corresponding to the FlowJo groups) that
    contains the gates.
    This is meant to account for the fact that the same sample can
    have multiple gating strategies, based on the experiment (which)
    is potentially stored in different workspace groups.
    The train sets (using .setup_anndata()) are created using the following logic:
        - First, all different available gates are stored in a dictionary
        - For each gate, a list is stored with samples that have already been
          assigned a gate with that name (~ training samples)
        - In general, as a presumption, one classifier is used for one gate
        - Multiple gates can be unified into one classifier (which leads to
          a multioutput-situation) if the training samples for two gates are 
          the same. That way, one sample can contain multiple gating strategies
          but multiple samples do not have to share all gating strategies
        - 

    Parameters
    ----------

    Examples
    --------

    """
    def __init__(self,
                 adata: AnnData,
                 wsp_group: str,
                 estimator: Literal["DecisionTree", "RandomForest"] = "DecisionTree",
                 train_on: Optional[Union[Literal["all_gated"], list[str]]] = "all_gated"):
        
        self.adata: AnnData = adata
        self.wsp_group = wsp_group
        self.base_estimator = estimator
        self.train_on = train_on
        self.status = "untrained"
        self.train_accuracy = {}
        self.test_accuracy = {}

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(" + 
            f"{self.train_on}" +
            ")"
        )

    def tune_hyperparameters(self):
        raise NotImplementedError("Hyperparameter tuning is currently not supported. :(")

    def train(self,
              on: Literal["compensated", "transformed"] = "transformed",
              pp_transformator: Optional[Literal["PowerTransformer", "QuantileTransformer"]] = None,
              pp_quantile_capping: Optional[float] = None,
              pp_scaler: Optional[Literal["RobustScaler", "MaxAbsScaler", "StandardScaler", "MinMaxScaler"]] = None,
              test_size: Optional[float] = 0.1,
              **kwargs):
        """Method to train the classifier.
        The training data are prepared by the following logic:
            - the user selects if compensated or transformed data are used
            - the user can select a transformation method
            - the user can select if the data are quantile capped
            - the user can select if the data are scaled to some extent
        The selected preprocessing algorithms (fitted scaler/quantile_caps) are stored
        The preprocessing is carried out:
            - transformation
            - quantile capping
            - scaling
            --- missing algorithms are ignored but the order stays the same
        The classifier is then fit on the data as they are prepared.
        Train and test accuracy are computed and stored for documentation purposes
        
        """
        self.on = on

        self.classifiers = {
            group: self._select_classifier(self.base_estimator)
            for group in self.training_groups
        }

        for gate_group in self.training_groups:
            print(f"Initializing Training on gates: {self.training_groups[gate_group]}")
            print("... Preparing training data")
            self.training_groups[gate_group]["preprocessing"] = {"transformation": pp_transformator,
                                                                 "quantile_capping": pp_quantile_capping,
                                                                 "scaling": pp_scaler}

            X_train, X_test, y_train, y_test = self.prepare_training_data(samples = self.training_groups[gate_group]["samples"],
                                                                          gate_columns = self.training_groups[gate_group]["gates"],
                                                                          group_identifier = gate_group,
                                                                          test_size = test_size,
                                                                          **kwargs)
            print("Fitting classifier...")
            self.classifiers[gate_group] = self.classifiers[gate_group].fit(X_train, y_train)
            print("Calculating accuracies...")
            train_prediction = self.classifiers[gate_group].predict(X_train)
            test_prediction = self.classifiers[gate_group].predict(X_test)
            self.training_groups[gate_group]["metrics"] = {"train_accuracy": accuracy_score(y_train, train_prediction),
                                                           "test_accuracy": accuracy_score(y_test, test_prediction)}
        _ = [check_is_fitted(self.classifiers[gate_group]) for gate_group in self.classifiers]
        self.update_train_groups()

    def update_train_groups(self):
        self.adata.uns["train_sets"][self.wsp_group] = self.training_groups

    def _select_classifier(self,
                           estimator: Literal["DecisionTree", "RandomForest"]):
        if estimator == "DecisionTree":
            return DecisionTree()
        if estimator == "RandomForest":
            return RandomForest()
        else:
            raise ClassifierNotImplementedError(estimator, implemented_estimators)

    def is_fit(self): pass

    def preprocess_data(self,
                        X: np.ndarray,
                        group_identifier: Union[str, int]
                        ) -> np.ndarray:
        preprocessing_dict = self.training_groups[group_identifier]["preprocessing"]
        if preprocessing_dict["transformation"] is not None:
            transformator = preprocessing_dict["transformation"]
            print(f"... transforming using {transformator}")
            if isinstance(transformator, str):
                X, transformer = transform_data(X, transformer = transformator)
                self.training_groups[group_identifier]["preprocessing"]["transformation"] = transformer
            else:
                X = transformator.transform(X)
        
        if preprocessing_dict["quantile_capping"] is not None:
            quantile_capping = preprocessing_dict["quantile_capping"]
            print(f"... quantile capping {quantile_capping} percentile")
            if isinstance(quantile_capping, float):
                X, transformer = cap_data(X, quantile_cap = quantile_capping)
                self.training_groups[group_identifier]["preprocessing"]["quantile_capping"] = transformer
            else:
                X = quantile_capping.transform(X)
        
        if preprocessing_dict["scaling"] is not None:
            scaler = preprocessing_dict["scaling"]
            print(f"... scaling using {scaler}")
            if isinstance(scaler, str):
                X, transformer = scale_data(X, scaler = scaler)
                self.training_groups[group_identifier]["preprocessing"]["scaling"] = transformer
            else:
                X = scaler.transform(X)
            
        return X


    def prepare_training_data(self,
                              samples: list[str],
                              gate_columns: list[str],
                              group_identifier: Union[str, int],
                              test_size: float = 0.1) -> tuple[np.ndarray,
                                                               np.ndarray,
                                                               np.ndarray,
                                                               np.ndarray]:
             
        adata_subset = self._subset_anndata_by_sample(samples)
        assert adata_subset.is_view ##TODO: delete later
        gate_indices = _find_gate_indices(self.adata,
                                          gate_columns)
        X = adata_subset.layers[self.on]
        y = adata_subset.obsm["gating"][:, gate_indices].toarray()
        X = self.preprocess_data(X,
                                 group_identifier)
        assert self.y_identities_correct(y = y,
                                         gates = adata_subset.obsm["gating"].toarray(),
                                         gate_indices = gate_indices)
        
        (X_train,
         X_test,
         y_train,
         y_test) = train_test_split(X, y, test_size = test_size)
        
        return X_train, X_test, y_train, y_test

    def merge_current_and_reference_gates(self,
                                          current_gates: list[str],
                                          reference_gates: list[dict]) -> list[dict]:
        merged_gates = []
        for gate in current_gates:
            gate_name = _find_current_population(gate)
            parent_gate = _find_parent_gate(gate)
            split_gate_path = tuple(parent_gate.split(GATE_SEPARATOR))
            merged_gates.extend(
                ref_gate
                for ref_gate in reference_gates
                if (ref_gate["gate"].gate_name == gate_name)
                and (ref_gate["gate_path"] == split_gate_path)
            )
        return merged_gates

    def add_gating_to_workspace(self,
                                adata: AnnData,
                                group_identifier: Union[str, int, float],
                                current_sample: str) -> None:

        gated_samples = self.training_groups[group_identifier]["samples"]
        current_gates = self.training_groups[group_identifier]["gates"]
        reference_gates = adata.uns["workspace"][self.wsp_group][gated_samples[0]]["gates"]
        gate_template = self.merge_current_and_reference_gates(current_gates, reference_gates)
        for gate in gate_template:
            gate_path = GATE_SEPARATOR.join(list(gate["gate_path"]))
            full_gate_path = GATE_SEPARATOR.join([gate_path, gate["gate"].gate_name])
            gate_index = _find_gate_indices(adata, full_gate_path)
            gate_information = adata.obsm["gating"][:, gate_index].toarray()
            dimensions = [gate["gate"].dimensions[i].id for i, _ in enumerate(gate["gate"].dimensions)]
            dimensions_indices = [adata.var.loc[adata.var["pnn"] == dim, "pns"].iloc[0] for dim in dimensions]
            fluorescence_data = adata[:, dimensions_indices].layers["compensated"]
            fluorescence_data = np.hstack([fluorescence_data, gate_information])
            fluorescence_data = fluorescence_data[fluorescence_data[:,2] == True]
            transformations = adata.uns["workspace"][self.wsp_group][current_sample]["transforms"]
            if fluorescence_data.shape[0] != 0:
                hull = ConvexHull(fluorescence_data[:, [0,1]])
                vertices = fluorescence_data[hull.vertices][:,[0,1]]
            else:
                print("WARNING NO HULL")
                vertices = np.zeros(shape = (2,2))
            if gate["gate"].gate_type == "PolygonGate":
                gate["gate"].vertices = _transform_vertices_according_to_gate_transform(vertices,
                                                                                        transforms = transformations,
                                                                                        gate_channels = dimensions)
            if gate["gate"].gate_type == "RectangleGate":
                gate_dimensions = np.array([[np.min(vertices[:,0]), np.max(vertices[:,0])],
                                            [np.min(vertices[:,1]), np.max(vertices[:,1])]])
                print(gate, "\n", vertices,"\n", gate_dimensions, "\n", dimensions, "\n", dimensions_indices)
                gate_dimensions = _transform_gates_according_to_gate_transform(gate_dimensions,
                                                                               transforms = transformations,
                                                                               gate_channels = dimensions)
                gate["gate"].dimensions[0].min = gate_dimensions[0,0]
                gate["gate"].dimensions[0].max = gate_dimensions[0,1]
                gate["gate"].dimensions[1].min = gate_dimensions[1,0]
                gate["gate"].dimensions[1].max = gate_dimensions[1,1]
        self.adata.uns["workspace"][self.wsp_group][current_sample]["gates"] = gate_template

    def gate_dataset(self) -> None:
        self.adata.obsm["gating"] = self.adata.obsm["gating"].todense()
        for gate_group in self.training_groups:
            print(f"Gating gates {self.training_groups[gate_group]}")
            non_gated_samples = [sample for sample in self.adata.obs["file_name"].unique()
                                 if sample not in self.training_groups[gate_group]["samples"]]
            gate_indices = _find_gate_indices(self.adata,
                                              gate_columns = self.training_groups[gate_group]["gates"])
            for sample in non_gated_samples:
                print(f"Gating sample {sample}...")
                sample_view = self._subset_anndata_by_sample(samples = sample, copy = False)
                X = sample_view.layers[self.on]
                X = self.preprocess_data(X = X,
                                         group_identifier = gate_group)
                print("Predicting")
                predictions: np.ndarray = self.classifiers[gate_group].predict(X)
                self._add_gating_to_input_dataset(sample_view,
                                                  predictions,
                                                  gate_indices)
                self.add_gating_to_workspace(sample_view,
                                             group_identifier = gate_group,
                                             current_sample = sample)

        self.adata.obsm["gating"] = csr_matrix(self.adata.obsm["gating"])



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

    def get_gated_samples(self,
                             gate_lut) -> list[str]:
        """Function checks for all samples that have been gated in some sense

        Parameters
        ----------
        gate_lut:
            the gate lookup table in the form of {file_name: {gate1: gate description, ...}, ...}


        Returns:
        --------
            list of samples that have an entry in the gate lookup table
        """
        return (
            [sample for sample in gate_lut if gate_lut[sample]]
            if self.train_on == "all_gated"
            else self.train_on
        )
    
    def create_gate_lut(self) -> dict[str, dict]:
        """Convenience wrapper around the fp.utils.create_gate_lut function.
        Extracts the wsp_group that has been specified by the user and stored
        in self.wsp_group and creates the corresponding lookup table

        Returns
        -------
        gate lookup table in format dict[str, dict]
        """
        workspace_subset = self.adata.uns["workspace"][self.wsp_group]
        return _create_gate_lut(workspace_subset)

    def gather_gate_paths(self,
                           gate_lut: dict[str: dict],
                           training_samples: list[str]):
        """Function collects all gates that are recorded for one sample
        and creates a dictionary where the full gate path and the dimensions are stored

        Parameters
        ----------
        gate_lut: lookup table for all samples and their gates
        training_samples: file names of the samples that have gates

        Returns
        -------
        Dictionary with the full gate paths and their dimensions

        """

        return {
            gate_lut[sample][gate]["full_gate_path"]: gate_lut[sample][gate]["dimensions"]
            for sample in training_samples
            for gate in gate_lut[sample].keys()
        }

    def create_reverse_lut(cls,
                           gate_lut: dict[str, dict],
                           training_gate_paths: list[str]
                           ) -> dict[str: dict]:
        """ 
        This function generates a dictionary of the form :
        "root/singlets: {'dimensions': ["FSC-A", "FSC-H"],
                         'samples': [filename1, filename2],
                         'training_columns': ["root/singlets"],
                         'parents': []}
        it is supposed to be a lookup table to assemble the necessary
        parameters that are needed to train a classifier for one gate.

        Parameters
        ----------
        gate_lut: lookup table for all samples and their gates
        training_gate_paths: gates with their full path

        Returns
        -------
        dictionary as described above

        """
        reverse_lut = {}
        for gate in training_gate_paths:
            gate_name = _find_current_population(gate)
            parents = [parent for parent in _find_parents_recursively(gate) if parent != "root"]
            reverse_lut[gate] = {
                "dimensions": training_gate_paths[gate],
                "samples": [
                    sample
                    for sample in gate_lut
                    if gate_name in gate_lut[sample].keys()
                ],
                "training_columns": parents + [gate],
                "parents": parents,
            }

        return reverse_lut

    def get_gates_with_sample_sample_set(self,
                                         reverse_lut: dict[str: dict],
                                         current_gate: str):
        return [
            candidate_gate
            for candidate_gate in reverse_lut
            if set(reverse_lut[current_gate]["samples"])
            == set(reverse_lut[candidate_gate]["samples"])
        ]

    def condense_gates(self,
                       reverse_lut: dict[str: dict]) -> dict[str: dict]:
        """
        Gates can be condensed into a multi-output classification problem
        if they share the same training examples (e.g. multiple files with the same gating)
        """

        # tuple conversion necessary due to immutable character (10.07.2023, correspondence with giiiirl)
        unique_sample_sets = set([tuple(reverse_lut[key]["samples"]) for key in reverse_lut])
        training_groups = {}
        for index, group_set in enumerate(unique_sample_sets):
            training_groups[index] = {key: values for
                                      key, values in reverse_lut.items() if
                                      reverse_lut[key]["samples"] == list(group_set)
                                      }
        training_groups = self.tidy_training_groups(training_groups,
                                                    unique_sample_sets)
        return training_groups

    def tidy_training_groups(self,
                             training_groups: dict[int: dict],
                             unique_sample_sets: set[list[str]]) -> dict[int: dict]:
        """Tidies the gate lookup table to contain only relevant information.
        For each training group (where potentially multiple gates are trained for at
        the same time), the full gate names and the corresponding training samples
        are stored.

        Returns:
            tidy_training_groups: dict of form {training_group: {gates: [list[gates], samples: list[file_names]]}}
        """
        tidy_dict = {}
        for group in training_groups:
            tidy_dict[group] = {"gates": list(training_groups[group].keys()),
                                "samples": list(list(unique_sample_sets)[group])}
        
        return tidy_dict

    def append_classifiers(self,
                           training_groups: dict[int: dict]) -> dict[int: dict]:
        for group in training_groups:
            training_groups[group]["classifier"] = self._select_classifier(self.base_estimator)
        return training_groups
    
    def setup_anndata(self):
        
        gate_lut = self.create_gate_lut()

        training_samples = self.get_gated_samples(gate_lut)
        
        training_gate_paths = self.gather_gate_paths(gate_lut, training_samples)
        
        reverse_lut = self.create_reverse_lut(gate_lut, training_gate_paths)

        training_groups = self.condense_gates(reverse_lut)

        if "train_sets" not in self.adata.uns.keys():
            self.adata.uns["train_sets"] = {}
        
        self.training_groups = training_groups
        self.gate_lut = gate_lut

        self.update_train_groups()

class unsupervisedGating(BaseGating):

    def __init__(self,
                 adata: AnnData,
                 gating_strategy: dict,
                 layer: str = None,
                 clustering_algorithm: Literal["leiden", "FlowSOM"] = "leiden",
                 cluster_key: str = None,
                 sensitivity: float = 1,
                 intervals: list[float] = [0.33,0.66]) -> None:
        """
        intervals:
            gives the intervals for lo, int and hi so that the first two numbers
            denote the lo quantile. Defaults to the 33rd percentile and below
        """
        gating_strategy = gating_strategy
        self.gating_strategy = self._preprocess_gating_strategy(gating_strategy)
        self.clustering_algorithm = clustering_algorithm
        self.adata = adata
        self.cluster_key = cluster_key or "clusters"
        self.layer = layer
        # sensitivity controls the cutoff.
        # will be log-transformed so that a sensitivity of 1 leads to cutoff of np.arcsinh(1) (~0.88).
        # and for every log there will be the addition of 0.1. sensitivity of 0.1 therefore
        # leads to np.arcsinh(1) + 0.1*-np.log10(0.1) which is np.arcsinh(1) + 0.1.
        # to increase the sensitivity, choose higher values (100 will result in a decrease of 0.2).
        self.sensitivity = sensitivity # will be logtransformed so that 1 is a cut
        if len(intervals) != 2:
            raise TypeError("Please provide intervals in two steps (e.g. [0.33, 0.66]).")
        self.intervals = intervals
        self._define_disallowed_characters()

    def _define_disallowed_characters(self) -> None:
        disallowed_characters = ["/", "[", "{", "(", ")", "}", "]", ".", "-"]
        replacement_dict = {char: "" for char in disallowed_characters}
        self.transtab = str.maketrans(replacement_dict)
        return

    def _generate_cutoff_table(self,
                               adata: AnnData,
                               layer: str,
                               sensitivity: float,
                               intervals: list[float]) -> dict:
        data_array = adata.layers[layer]
        cutoffs = {}
        for i, marker in enumerate(adata.var_names):
            marker = self._remove_disallowed_character(marker)
            lo_end_cutoff = self._calculate_cutoff_interval(data_array[:,i], intervals[0])
            int_end_cutoff = self._calculate_cutoff_interval(data_array[:,i], intervals[1])
            cutoffs[marker] = {
                "pos": np.arcsinh(1) - (0.1 * np.log10(sensitivity)),
                "lo": (np.min(data_array[:,i]), lo_end_cutoff),
                "int": (lo_end_cutoff, int_end_cutoff),
                "hi": (int_end_cutoff, np.max(data_array[:,i])),
            }
        return cutoffs

    def _calculate_cutoff_interval(self,
                                   data: np.ndarray,
                                   quantile: float) -> tuple[float, float]:
        return np.percentile(data, quantile * 100)

    def identify_populations(self,
                             cluster_kwargs: Optional[dict] = None):
        if cluster_kwargs is None:
            cluster_kwargs = {}
        if isinstance(self.adata.obsm["gating"], csr_matrix):
            self.adata.obsm["gating"] = self.adata.obsm["gating"].todense()
        ### mutable object to keep track of analyzed gates. this is necessary
        ### because if parents are not present immediately, the population
        ### is analyzed beforehand but still a key in the dictionary and
        ### therefore analyzed twice.
        self.already_analyzed = []
        for cell_population in self.gating_strategy:
            self._identify_population(cell_population,
                                      cluster_kwargs)
        self.adata.obsm["gating"] = csr_matrix(self.adata.obsm["gating"])

        return
    
    def _preprocess_gating_strategy(self,
                                    gating_strategy: dict) -> dict:
        parent_populations = {entry[0] for _, entry in gating_strategy.items()}
        return {
            population: [
                [
                    key,
                    value[1],
                ]  ## [CD4CM, ["CD4+", "CD197-"]] instead of [CD4_T_cells, ["CD4+", "CD197-"]]
                for key, value in gating_strategy.items()
                if value[0] == population
            ]
            for population in parent_populations
        }

    def _population_is_already_a_gate(self,
                                      parent_population) -> bool:
        return parent_population in [_find_current_population(gate)
                                     for gate in self.adata.uns["gating_cols"]]
    
    def _process_markers(self,
                         markers: list[str]) -> dict[str, list[Optional[str]]]:
        if not isinstance(markers, list):
            markers = [markers]
        marker_dict = {"up": [],
                       "down": [],
                       "lo": [],
                       "int": [],
                       "hi": []}
        for marker in markers:
            if marker.endswith('+'):
                marker_dict["up"].append(marker.split("+")[0])
            elif marker.endswith('-'):
                marker_dict["down"].append(marker.split("-")[0])
            elif marker.endswith('lo'):
                marker_dict["lo"].append(marker.split('lo')[0])
            elif marker.endswith('int'):
                marker_dict["int"].append(marker.split('int')[0])
            elif marker.endswith('hi'):
                marker_dict["hi"].append(marker.split('hi')[0])
            else:
                marker_dict["up"].append(marker)
        return marker_dict

    def _find_parent_population_in_gating_strategy(self,
                                                   query_population: str) -> str:
        for entry in self.gating_strategy:
            for population in self.gating_strategy[entry]:
                if population[0] == query_population:
                    return entry
        raise ParentGateNotFoundError(query_population)

    def _identify_population(self,
                             population_to_cluster: str,
                             cluster_kwargs: dict) -> None:
        if population_to_cluster in self.already_analyzed:
            return
        print(f"Analyzing population: {population_to_cluster}")

        # handles the case when the parent lives within the gating strategy        
        if not self._population_is_already_a_gate(population_to_cluster):
            parent_of_population_to_cluster = self._find_parent_population_in_gating_strategy(population_to_cluster)
            if parent_of_population_to_cluster in self.gating_strategy:
                self._identify_population(parent_of_population_to_cluster,
                                          cluster_kwargs)
            else:
                raise ParentGateNotFoundError(population_to_cluster)
            
        if self.adata.shape[0] <= 1:
            """
            Handles the case where no parent cells have been found. In order to avoid missing gates,
            empty gates are appended.
            TODO: code doubling...
            """
            for population_list in self.gating_strategy[population_to_cluster]:
                population_name: str = population_list[0]
                print(f"... gating population {population_name}")
                parent_gate_path = _find_gate_path_of_gate(file_subset, population_to_cluster)
                population_gate_path = GATE_SEPARATOR.join([parent_gate_path, population_name])

                if not self._population_is_already_a_gate(population_name):
                    self._append_gate_column_to_adata(population_gate_path)

        else:
            gate_subset = subset_gate(self.adata, gate = population_to_cluster, as_view = True)
            if gate_subset.shape[0] >= 1:
                self.cutoffs = self._generate_cutoff_table(gate_subset,
                                                           layer = self.layer,
                                                           sensitivity = self.sensitivity,
                                                           intervals = self.intervals)
            for sample in self.adata.obs["file_name"].unique():
                print(f"... sample {sample}")
                file_subset = self._subset_anndata_by_sample(adata = self.adata,
                                                             samples = sample,
                                                             copy = True)
                if file_subset.shape[0] <= 1 or self._has_no_cells_in_gate(file_subset,
                                                                           gate = population_to_cluster):
                    """
                    Handles the case where no parent cells have been found. In order to avoid missing gates,
                    empty gates are appended.
                    TODO: code doubling...
                    """
                    for population_list in self.gating_strategy[population_to_cluster]:
                        population_name: str = population_list[0]
                        print(f"... gating population {population_name}")
                        parent_gate_path = _find_gate_path_of_gate(file_subset, population_to_cluster)
                        population_gate_path = GATE_SEPARATOR.join([parent_gate_path, population_name])

                        if not self._population_is_already_a_gate(population_name):
                            self._append_gate_column_to_adata(population_gate_path)
                    continue

                file_subset = self._cluster_dataset(adata = file_subset,
                                                    gate = population_to_cluster,
                                                    layer = self.layer,
                                                    cluster_kwargs = cluster_kwargs)

                for population_list in self.gating_strategy[population_to_cluster]:
                    population_name: str = population_list[0]
                    print(f"     ... gating population {population_name}")
                    
                    ## each entry is a list with the structure [population_name, [marker1, marker2, marker3]]
                    parent_gate_path = _find_gate_path_of_gate(file_subset, population_to_cluster)
                    population_gate_path = GATE_SEPARATOR.join([parent_gate_path, population_name])

                    if not self._population_is_already_a_gate(population_name):
                        self._append_gate_column_to_adata(population_gate_path)
                    
                    gate_index: list[int] = _find_gate_indices(self.adata,
                                                               population_gate_path)

                    markers: list[str] = population_list[1]
                    markers_of_interest = self._process_markers(markers)
                    cluster_vector = self._identify_clusters_of_interest(file_subset,
                                                                         markers_of_interest)
                    cell_types = self._map_cell_types_to_cluster(file_subset,
                                                                 cluster_vector,
                                                                 population_name)
                    predictions = self._convert_cell_types_to_bool(cell_types)
                    self._add_gating_to_input_dataset(file_subset,
                                                      predictions,
                                                      gate_index)
        self.already_analyzed.append(population_to_cluster)
        return 
  
    def _has_no_cells_in_gate(self,
                              adata: AnnData,
                              gate: str):
        gate_subset = subset_gate(adata,
                                  gate,
                                  as_view = True)
        return gate_subset.shape[0] <= 1

    def _convert_cell_types_to_bool(self,
                                    cell_types: list[str]) -> np.ndarray:
        return np.array(
            list(map(lambda x: x != "other", cell_types)),
            dtype=bool
        ).reshape((len(cell_types), 1))
        
    def _map_cell_types_to_cluster(self,
                                   subset: AnnData,
                                   cluster_vector: pd.Index,
                                   population: str) -> list[str]:
        # TODO: map function...
        return [population if cluster in cluster_vector else "other"
                for cluster in subset.obs[self.cluster_key].to_list()]
    
    def _convert_markers_to_query_string(self,
                                         markers_of_interest: dict[str: list[Optional[str]]]) -> str:

        #cutoff = str(np.arcsinh(1) - (0.1 * np.log10(self.sensitivity)))
        cutoffs = self.cutoffs
        up_markers = markers_of_interest["up"]
        down_markers = markers_of_interest["down"]
        lo_markers = markers_of_interest["lo"]
        int_markers = markers_of_interest["int"]
        hi_markers = markers_of_interest["hi"]
        query_strings = (
            [f"{marker} > {cutoffs[marker]['pos']}"
             for marker in up_markers] +
            [f"{marker} < {cutoffs[marker]['pos']}"
             for marker in down_markers] + 
            [f"{cutoffs[marker]['lo'][0]} < {marker} < {cutoffs[marker]['lo'][1]}"
             for marker in lo_markers] + 
            [f"{cutoffs[marker]['int'][0]} < {marker} < {cutoffs[marker]['int'][1]}"
             for marker in int_markers] + 
            [f"{cutoffs[marker]['hi'][0]} < {marker} < {cutoffs[marker]['hi'][1]}"
             for marker in hi_markers]
        )
        return " & ".join(query_strings)

    def _clean_marker_names(self,
                            markers: list[str]) -> list[str]:
        """This function checks for disallowed characters that would otherwise mess up the pd.query function"""
        if isinstance(markers, pd.Index):
            return self._remove_disallowed_character_list(markers)
        # if isinstance(markers, pd.Index):
        #     markers = list(markers)
        #     for i, marker in enumerate(markers):
        #         if any(k in marker for k in disallowed_characters):
        #             transtab = marker.maketrans(replacement_dict)
        #             markers[i] = marker.translate(transtab)
        if isinstance(markers, dict):
            for direction in markers:
                for i, _ in enumerate(markers[direction]):
                    markers[direction][i] = self._remove_disallowed_character(markers[direction][i])
        return markers

    def _remove_disallowed_character_list(self,
                                          str_list: list[str]) -> list[str]:
        return [self._remove_disallowed_character(string) for string in str_list]

    def _remove_disallowed_character(self,
                                     input_str: str) -> str:
        return input_str.translate(self.transtab)

    def _identify_clusters_of_interest(self,
                                       adata: AnnData,
                                       markers_of_interest: dict[str: list[Optional[str]]]) -> list[str]:
        df = adata.to_df(layer = "transformed")
        df.columns = self._clean_marker_names(df.columns)
        markers_of_interest = self._clean_marker_names(markers_of_interest)
        df[self.cluster_key] = adata.obs[self.cluster_key].to_list()
        medians = df.groupby(self.cluster_key).median()
        query = self._convert_markers_to_query_string(markers_of_interest)
        cells_of_interest: pd.DataFrame = medians.query(query) 
        return cells_of_interest.index

    def _cluster_dataset(self,
                         adata: AnnData,
                         gate: str,
                         layer: str,
                         cluster_kwargs: dict) -> AnnData:

        #set reasonable defaults for now:
        if "scaling" not in cluster_kwargs:
            cluster_kwargs["scaling"] = None
        if "key_added" not in cluster_kwargs:
            cluster_kwargs["key_added"] = "clusters"

        if self.clustering_algorithm == "leiden":
            leiden(adata,
                   gate = gate,
                   layer = layer,
                   **cluster_kwargs)
        elif self.clustering_algorithm == "parc":
            parc(adata,
                 gate = gate,
                 layer = layer,
                 **cluster_kwargs)
        elif self.clustering_algorithm == "flowsom":
            flowsom(adata,
                    gate = gate,
                    layer = layer,
                    **cluster_kwargs)
        else:
            phenograph(adata,
                       gate = gate,
                       layer = layer,
                       **cluster_kwargs)
        return adata

    @classmethod
    def setup_anndata(cls):
        pass

