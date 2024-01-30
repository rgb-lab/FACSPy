from anndata import AnnData
import imblearn
import numpy as np
from typing import Optional
from imblearn.over_sampling._smote.base import SMOTE, SMOTEN, SMOTENC, BaseOverSampler
from imblearn.over_sampling._smote.cluster import KMeansSMOTE
from imblearn.over_sampling._adasyn import ADASYN
from imblearn.over_sampling import BorderlineSMOTE, RandomOverSampler
from imblearn.over_sampling._smote.filter import SVMSMOTE
from imblearn.combine import SMOTEENN

from imblearn.under_sampling.base import BaseUnderSampler
from imblearn.under_sampling import RandomUnderSampler

import pandas as pd

class GaussianOverSampler:

    def __init__(self,
                 sampling_strategy: dict,
                 standard_deviation: float = 0.01,
                 random_state: int = 187):
        
        self.sampling_strategy = sampling_strategy
        self.sd = standard_deviation
        np.random.seed(random_state)

    def fit_resample(self,
                     X: np.ndarray,
                     y: np.ndarray):
        X_sampled = np.empty(shape = (0, X.shape[1]), dtype = np.float64)
        y_sampled = np.empty(shape = (0, y.shape[1]), dtype = np.int64)
        
        for _class, _class_count in self.sampling_strategy.items():
            idxs = np.where(y == _class)[0]
            X_ = X[idxs]
            y_ = y[idxs]
            if len(idxs) >= _class_count:
                X_sampled = np.vstack([X_sampled, X_])
                y_sampled = np.vstack([y_sampled, y_])
                continue
            else:
                class_X_sampled = np.empty(shape = (0, X.shape[1]), dtype = np.float64)
                class_y_sampled = np.empty(shape = (0, y.shape[1]), dtype = np.int64)
                class_X_sampled = np.vstack([class_X_sampled, X_])
                class_y_sampled = np.vstack([class_y_sampled, y_])
                while class_X_sampled.shape[0] < _class_count:
                    if (_class_count - class_X_sampled.shape[0]) > X_.shape[0]:
                        class_X_sampled = np.vstack(
                            [class_X_sampled, self._apply_noise(X_)]
                        )
                        class_y_sampled = np.vstack(
                            [class_y_sampled, y_]
                        )
                    else:
                        subsampled_X, subsampled_y = self._random_subsample(X_, y_, (_class_count - class_X_sampled.shape[0]))
                        class_X_sampled = np.vstack(
                            [class_X_sampled, self._apply_noise(subsampled_X)]
                        )
                        class_y_sampled = np.vstack(
                            [class_y_sampled, subsampled_y]
                        )
                assert class_X_sampled.shape[0] == _class_count, class_X_sampled.shape
                assert class_y_sampled.shape[0] == _class_count, class_y_sampled.shape
                X_sampled = np.vstack([X_sampled, class_X_sampled])
                y_sampled = np.vstack([y_sampled, class_y_sampled])
        return X_sampled, y_sampled

    def _random_subsample(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          n: int):
        idxs = np.random.choice(X.shape[0], n, replace = False)
        return X[idxs], y[idxs]

    def _apply_noise(self,
                     X):
        noise = np.random.normal(
            1, self.sd, X.shape[0]*X.shape[1]
        ).reshape(X.shape)
        return X*noise


IMPLEMENTED_OVERSAMPLERS = {
    "SMOTE": SMOTE,
    "SMOTEENN": SMOTEENN,
    "SMOTEN": SMOTEN,
    "SMOTENC": SMOTENC,
    "SVMSMOTE": SVMSMOTE,
    "BorderlineSMOTE": BorderlineSMOTE,
    "RandomOverSampler": RandomOverSampler,
    "ADASYN": ADASYN,
    "KMeansSMOTE": KMeansSMOTE,
    "Gaussian": GaussianOverSampler
}
IMPLEMENTED_UNDERSAMPLERS = {
    "RandomUnderSampler": RandomUnderSampler
}

class GateSampler:

    def __init__(self,
                 adata: AnnData,
                 target_size_per_gate: int,
                 oversampler: str = "Gaussian",
                 undersampler: str = "RandomUnderSampler",
                 rare_cells_cutoff: Optional[int] = None,
                 oversample_rare_cells: bool = False,
                 rare_cells_target_size: Optional[int] = None) -> None:
        
        self.adata = adata

        
        self._validate_input_parameters(oversampler,
                                        undersampler)
        
        self.gating = self._create_gating_frame(adata)
        self.CELL_THRESHOLD = rare_cells_cutoff or 50
        self.oversample_rare_cells = oversample_rare_cells

        self.rare_cells_target_size = rare_cells_target_size or self.CELL_THRESHOLD
        self.target_size = target_size_per_gate
        self.target_size_half = int(np.ceil(self.target_size/2))

        if isinstance(oversampler, str):
            self.oversampler = IMPLEMENTED_OVERSAMPLERS[oversampler]
        else:
            self.oversampler = oversampler

        if isinstance(undersampler, str):
            self.undersampler = IMPLEMENTED_UNDERSAMPLERS[undersampler]
        else:
            self.undersampler = undersampler
        
        self.random_oversampler = IMPLEMENTED_OVERSAMPLERS["RandomOverSampler"]

        self._bit_ranges = None

    def _create_gating_frame(self,
                             adata: AnnData) -> pd.DataFrame:
        return pd.DataFrame(
            data = adata.obsm["gating"].toarray(),
            columns = adata.uns["gating_cols"],
            index = adata.obs_names
        )

    def _validate_input_parameters(self,
                                   oversampler,
                                   undersampler) -> None:
        if oversampler not in IMPLEMENTED_OVERSAMPLERS:
            raise NotImplementedError(f"{oversampler} is not implemented. Please choose from {list(IMPLEMENTED_OVERSAMPLERS.keys())}")
        if undersampler not in IMPLEMENTED_UNDERSAMPLERS:
            raise NotImplementedError(f"{undersampler} is not implemented. Please choose from {list(IMPLEMENTED_UNDERSAMPLERS.keys())}")
        return

    def _subset_arrays_by_idx(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              idxs) -> tuple[np.ndarray, np.ndarray]:
        return X[idxs], y[idxs]

    #def nd_isin(self,
    #            array: np.ndarray,
    #            test_array: np.ndarray) -> np.ndarray:
    #    z = set(map(tuple, test_array))
    #    return np.array([row in z for row in map(tuple, array)])

    def _calculate_threshold_indices(self,
                                     binary_classes: np.ndarray,
                                     binary_classes_frequency: np.ndarray,
                                     binary_gating: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        classes_above_threshold = binary_classes[np.where(binary_classes_frequency >= self.CELL_THRESHOLD)[0]]
        classes_above_threshold_mask = np.isin(binary_gating, classes_above_threshold)
        classes_above_threshold_idxs = classes_above_threshold_mask.nonzero()[0]
        classes_below_threshold_idxs = np.invert(classes_above_threshold_mask).nonzero()[0]
        return classes_above_threshold_idxs, classes_below_threshold_idxs

    def _map_gate_matrix(self,
                         y: np.ndarray) -> tuple[dict, dict]:
        """
        Function maps binary integers to ascending integers.
        We need that in case we have more than 64 gates, because
        the binarized gating strategy will contain two 64bit integers
        """
        if y.shape[1] == 1:
            y = np.squeeze(y)
        uniques = np.unique(y, axis = 0)
        if y.ndim == 1:
            return (
                {uniques[i]: i for i in range(uniques.shape[0])},
                {i: uniques[i] for i in range(uniques.shape[0])}
            )
        else:
            return (
                {tuple(uniques[i]): i for i in range(uniques.shape[0])},
                {i: tuple(uniques[i]) for i in range(uniques.shape[0])}
            )

    def _apply_gate_map(self,
                        y: np.ndarray,
                        gate_map: dict) -> np.ndarray:
        if y.shape[1] == 1:
            y = np.squeeze(y)
        if y.ndim == 1:
            return np.array([gate_map.__getitem__(y[i])
                             for i in range(len(y))
                             if y[i] in gate_map.keys()])
        tuple_list = list(map(tuple, y))
        return np.array(np.array([gate_map.__getitem__(tuple_list[i])
                                  for i in range(len(tuple_list))
                                  if tuple_list[i] in gate_map.keys()]))

    def _calculate_sampling_strategy(self,
                                     y: np.ndarray):
        bin_classes, counts_per_class = np.unique(y,
                                                  return_counts = True)
        counts: dict = {
            int(bin_class): count
            for bin_class, count in
            zip(bin_classes, counts_per_class)
        }
        sum_of_counts = y.shape[0]
        scale_factor = self.target_size_half / sum_of_counts
        sampling_strategy = {
            bin_class: int(np.ceil(scale_factor*count))
            for bin_class, count
            in counts.items()
        }
        return counts, sampling_strategy

    def _oversample(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    sampling_strategy: dict,
                    counts: dict) -> tuple[np.ndarray, np.ndarray]:
        
        sampler: BaseOverSampler = self.oversampler(sampling_strategy = sampling_strategy)
        sampler: RandomOverSampler = RandomOverSampler(sampling_strategy = sampling_strategy)
        X_, y_ = sampler.fit_resample(X, y)
        y_ = y_.reshape(y_.shape[0], 1)
        return X_, y_

    def _random_subsample(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          n: int):
        idxs = np.random.choice(X.shape[0], n, replace = False)
        return X[idxs], y[idxs]

    def _resample(self,
                  X: np.ndarray,
                  y_binary: np.ndarray,
                  X_sampled: np.ndarray,
                  y_binary_sampled: np.ndarray,
                  indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if indices.shape[0] == 0 or \
           indices.shape[0] == self.event_size:
            ## if there are no indices or if it already has the number we want
            ## we just add it to our pre-final sampled arrays
            X_sampled = np.vstack([X_sampled, X[indices]])
            y_binary_sampled = np.vstack([y_binary_sampled, y_binary[indices]])
        elif indices.shape[0] < self.event_size:
            print("case population is smaller ")
            print(y_binary[indices])
            ### if the negative events are too few,
            ### we oversample while keeping the dataset structure
            counts, sampling_strategy = self._calculate_sampling_strategy(y_binary[indices])

            X_, y_ = self._oversample(X[indices],
                                      y_binary[indices],
                                      sampling_strategy,
                                      counts)
            
            X_sampled = np.vstack([X_sampled, X_])
            y_binary_sampled = np.vstack([y_binary_sampled, y_])

        elif indices.shape[0] > self.event_size:
            X_, y_ = self._random_subsample(X[indices],
                                            y_binary[indices],
                                            n = self.target_size_half)
            X_sampled = np.vstack([X_sampled, X_])
            y_binary_sampled = np.vstack([y_binary_sampled, y_])

        return X_sampled, y_binary_sampled
        


    def fit_resample(self,
                     X: np.ndarray,
                     y: np.ndarray,
                     shuffle: bool = False,
                     oversampler_kwargs: Optional[dict] = None,
                     undersampler_kwargs: Optional[dict] = None):
        
        if oversampler_kwargs is None:
            oversampler_kwargs = {}
        if undersampler_kwargs is None:
            undersampler_kwargs = {}

        ## we keep the expression matrix X and the gate matrix y

        ## y is binarized to comply with the dataset heterogeneity
        ## and the non-multioutput-character of imblearn
        y_binary = self._convert_gate_matrix_to_binary(y)

        ## we create empty arrays to store the sampled information
        X_sampled = np.empty(shape = (0, X.shape[1]), dtype = np.float64)
        y_binary_sampled= np.empty(shape = (0, y_binary.shape[1]), dtype = np.int64)

        ## loop through gates
        for gate in range(y.shape[1]):
            print(f"gate {gate}")
#            if gate == 4:
#                break
            negative_indices = np.where(y[:,gate] == 0)[0]
            positive_indices = np.where(y[:,gate] == 1)[0]

            X_sampled, y_binary_sampled = self._resample(X, y_binary, X_sampled, y_binary_sampled,
                                                         indices = negative_indices)
            X_sampled, y_binary_sampled = self._resample(X, y_binary, X_sampled, y_binary_sampled,
                                                         indices = positive_indices)

        y_binary_sampled = y_binary_sampled.astype(np.int64)
        y_sampled = self._convert_binary_to_gate_matrix(y_binary_sampled)



        return X_sampled, y_sampled
    
    def _adjust_cell_count_to_dataset_shape(self, count):
        return int(np.ceil(count/self.gating.shape[0]))
            
    def fit_resample_specific(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              shuffle: bool = True,
                              oversampler_kwargs: Optional[dict] = None,
                              undersampler_kwargs: Optional[dict] = None) -> tuple[np.ndarray, np.ndarray]:
    
        if oversampler_kwargs is None:
            oversampler_kwargs = {}
        if undersampler_kwargs is None:
            undersampler_kwargs = {}

        y_binary = self._convert_gate_matrix_to_binary(y)
        user_defined_gates = self._identify_gates_binaries(self.gating)

        gate_map, reverse_gate_map = self._map_gate_matrix(y_binary)

        y_mapped = self._apply_gate_map(y_binary, gate_map)
        y_mapped = y_mapped.reshape(y_mapped.shape[0], 1)
        user_defined_gates_mapped = self._apply_gate_map(user_defined_gates, gate_map)
        
        binary_classes, binary_classes_frequency = self._calculate_gate_frequencies(y_mapped)
        binary_classes = [bin_class[0] for bin_class in binary_classes]

        X_sampled = np.empty(shape = (0, X.shape[1]), dtype = np.float64)
        y_mapped_sampled = np.empty(shape = (0, y_mapped.shape[1]), dtype = np.int64)

        #above_thresholds = {}
        #below_thresholds = {}
        #below_cutoff = {}
        #gate_counter = 0
        #for bin_class, count in zip(binary_classes, binary_classes_frequency):
        #    if count > self.target_size:
        #        if bin_class in user_defined_gates_mapped:
        #            gate_counter += 1
        #            above_thresholds[bin_class] = self.target_size
        #        else:
        #            above_thresholds[bin_class] = self._adjust_cell_count_to_dataset_shape(count)
        #    elif self.CELL_THRESHOLD < count <= self.target_size:
        #        if bin_class in user_defined_gates_mapped:
        #            gate_counter += 1
        #            below_thresholds[bin_class] = self.target_size
        #        else:
        #            adjusted_count = self._adjust_cell_count_to_dataset_shape(count)
        #            if adjusted_count > count:
        #                below_thresholds[bin_class] = adjusted_count
        #            else:
        #                above_thresholds[bin_class] = adjusted_count
        #    elif count <= self.CELL_THRESHOLD:
        #        if bin_class in user_defined_gates_mapped:
        #            gate_counter += 1
        #            below_cutoff[bin_class] = self.target_size
        #        else:
        #            adjusted_count = self._adjust_cell_count_to_dataset_shape(count)
        #            if adjusted_count > count:
        #                below_cutoff[bin_class] = adjusted_count
        #            else:
        #                above_thresholds[bin_class] = adjusted_count
        #print(gate_counter)
        above_thresholds = {
            bin_class: self.target_size
            for bin_class, count
            in zip(binary_classes, binary_classes_frequency)
            if count > self.target_size
        }
        below_thresholds = {
            bin_class: self.target_size
            for bin_class, count
            in zip(binary_classes, binary_classes_frequency)
            if self.CELL_THRESHOLD < count <= self.target_size
        }
        below_cutoff = {
            bin_class: self.rare_cells_target_size
            for bin_class, count
            in zip(binary_classes, binary_classes_frequency)
            if count <= self.CELL_THRESHOLD

        }

        for bin_class, count in above_thresholds.items():
            idxs = np.where(y_mapped == bin_class)[0]
            X_, y_ = self._random_subsample(X[idxs], y_mapped[idxs], min(self.target_size, count))
            X_sampled = np.vstack([X_sampled, X_])
            y_mapped_sampled = np.vstack([y_mapped_sampled, y_])
        
        if below_thresholds:
            idxs = np.where(np.isin(y_mapped, list(below_thresholds.keys())))[0]
            print(idxs)
            oversampler: BaseOverSampler = self.oversampler(sampling_strategy = below_thresholds)
            X_, y_ = oversampler.fit_resample(X[idxs], y_mapped[idxs])
            y_ = y_.reshape(y_.shape[0], 1)
            X_sampled = np.vstack([X_sampled, X_])
            y_mapped_sampled = np.vstack([y_mapped_sampled, y_])

        if self.oversample_rare_cells and below_cutoff:
            idxs = np.where(np.isin(y_mapped, list(below_cutoff.keys())))[0]
            oversampler: BaseOverSampler = RandomOverSampler(sampling_strategy = below_cutoff)
            X_, y_ = oversampler.fit_resample(X[idxs], y_mapped[idxs])
            y_ = y_.reshape(y_.shape[0], 1)
            X_sampled = np.vstack([X_sampled, X_])
            y_mapped_sampled = np.vstack([y_mapped_sampled, y_])

        y_binary_sampled = self._apply_gate_map(y_mapped_sampled, reverse_gate_map)

        y_sampled = self._convert_binary_to_gate_matrix(y_binary_sampled.astype(np.int64).reshape(y_binary_sampled.shape[0], y_binary.shape[1]))

        return self._shuffle(X_sampled, y_sampled) if shuffle else (X_sampled, y_sampled)

    def fit_resample_old(self,
                     X: np.ndarray,
                     y: np.ndarray,
                     shuffle: bool = True,
                     oversampler_kwargs: Optional[dict] = None,
                     undersampler_kwargs: Optional[dict] = None) -> tuple[np.ndarray, np.ndarray]:
        # sourcery skip: min-max-identity

        if oversampler_kwargs is None:
            oversampler_kwargs = {}
        if undersampler_kwargs is None:
            undersampler_kwargs = {}

        print("Starting Sampling!")
        print(f"Original n_gates: {y.shape[1]}")
        _y = self._convert_gate_matrix_to_binary(y)
        print(f"Found {len(np.unique(_y))} gates after binarization")
        
        gate_map, reverse_gate_map = self._map_gate_matrix(_y)
    
        _y = self._apply_gate_map(_y, gate_map)


        binary_classes, binary_classes_frequency = self._calculate_gate_frequencies(_y)
        classes_above_threshold_idxs, classes_below_threshold_idxs = self._calculate_threshold_indices(binary_classes,
                                                                                                       binary_classes_frequency,
                                                                                                       binary_gating = _y)

        ## first, the above thresholds
        ## threshold refers to the minimum cells that are needed for SMOTE, so not the class size after sampling!
        ## at = above threshold
        at_X, at_y = self._subset_arrays_by_idx(X,
                                                _y,
                                                classes_above_threshold_idxs)

        at_binary_classes, at_binary_classes_frequency = self._calculate_gate_frequencies(at_y)

        if "sampling_strategy" not in undersampler_kwargs:
            undersampler_kwargs["sampling_strategy"] = {
                binary_class: self.target_size if self.target_size < class_count else class_count
                for (binary_class, class_count) in zip(
                    at_binary_classes, at_binary_classes_frequency
                )
            }

        if "sampling_strategy" not in oversampler_kwargs:
            oversampler_kwargs["sampling_strategy"] = {
                binary_class: self.target_size
                for (binary_class, class_count) in zip(
                    at_binary_classes, at_binary_classes_frequency
                )
            }
        print("created sampling strategies")
        ### next, the below thresholds
        bt_X, bt_y = self._subset_arrays_by_idx(X,
                                                _y,
                                                classes_below_threshold_idxs)

        bt_binary_classes, bt_binary_classes_frequency = self._calculate_gate_frequencies(bt_y)
        random_oversampler_kwargs = {
            "sampling_strategy": {
                binary_class: self.target_size
                for (binary_class, class_count) in zip(
                    bt_binary_classes, bt_binary_classes_frequency
                )
            }
        }
        self.oversampler: BaseOverSampler = self.oversampler(**oversampler_kwargs)
        self.undersampler: BaseUnderSampler = self.undersampler(**undersampler_kwargs)

        self.below_threshold_oversampler: BaseOverSampler = self.random_oversampler(**random_oversampler_kwargs)

        at_X, at_y = self._fit_resample_above_threshold(at_X,
                                                        at_y)

        if bt_X.shape[0] > 0:
            bt_X, bt_y = self.below_threshold_oversampler.fit_resample(bt_X,
                                                                       bt_y)
        
        X, y = np.concatenate((at_X, bt_X)), np.concatenate((at_y, bt_y))

        y = np.array(self._apply_gate_map(y, reverse_gate_map), dtype = np.int64)

        y = self._convert_binary_to_gate_matrix(y)
        return self._shuffle(X, y) if shuffle else (X, y)

    def _identify_gates_binaries(self,
                                 gating: pd.DataFrame):
        """
        function to verify wanted gates.
        Everything outside of this
        is an artifact and will be treated as such"""
        from .._utils import _find_parents_recursively, find_parent_gate, _find_children_of_gate
        gates_to_keep = []

        for gate in gating.columns:
            parents = _find_parents_recursively(gate)
            parents.append(gate)

            ### a good gate is a gate where only
            gate_coding = np.array(
                [0 if not gate in parents else 1
                 for gate in gating.columns]
            )
            gates_to_keep.append(
                self._convert_gate_matrix_to_binary(
                    gate_coding.reshape(1, gate_coding.shape[0])
                )[0]
            )
        
        return np.array(gates_to_keep, dtype = np.int64)


    def _fit_resample_above_threshold(self,
                                      X: np.ndarray,
                                      y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, y = self.undersampler.fit_resample(X,
                                              y)
        X, y = self.oversampler.fit_resample(X,
                                             y)
        return X, y

    def _shuffle(self,
                 X: np.ndarray,
                 y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert X.shape[0] == y.shape[0]
        perm = np.random.permutation(X.shape[0])
        return self._subset_arrays_by_idx(X, y, perm)

    def _select_gates_above_threshold(self,
                                      binary_classes: np.ndarray,
                                      binary_classes_frequency: np.ndarray) -> np.ndarray:
        return binary_classes[np.where(binary_classes_frequency > self.CELL_THRESHOLD)[0]]

    def _calculate_gate_frequencies(self,
                                    y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.unique(y, axis = 0, return_counts = True)

    def __bits_to_int(self,
                      y: np.ndarray) -> np.ndarray:
        return y.dot(1 << np.arange(y.shape[-1]-1, -1, -1, dtype = np.int64))
    
    def __int_to_bits(self,
                      y: np.ndarray,
                      bitrange: int) -> np.ndarray:
        return (((y[:,None] & (1 << np.arange(bitrange, dtype = np.int64))[::-1])) > 0).astype(int)

    def _convert_gate_matrix_to_binary(self,
                                       y: np.ndarray) -> np.ndarray:
        ### this line is necessary to set all "sign bits" to 0
        y = np.hstack([np.zeros(shape = (y.shape[0], 1)), y])

        ### we have to convert the matrix into <64 wide columns because
        ### the bit conversion is limited to 64bit numbers
        y_list = np.array_split(y, np.ceil(y.shape[1]/64), axis = 1)
        if self._bit_ranges is None:
            self._bit_ranges = [y_subset.shape[1] for y_subset in y_list]
        return np.vstack([self.__bits_to_int(y_subset) for y_subset in y_list]).astype(np.int64).T

    def _convert_binary_to_gate_matrix(self,
                                       y: np.ndarray) -> np.ndarray:
        if y.shape[1] == 1:
            y = np.squeeze(y)
        if y.ndim == 1:
            return self.__int_to_bits(y, self._bit_ranges[0])[:,1:]
        gate_list = [self.__int_to_bits(y[:,i], self._bit_ranges[i]) for i in range(y.shape[1])]
        return np.hstack(gate_list)[:,1:]

