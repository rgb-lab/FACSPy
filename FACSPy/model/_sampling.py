import imblearn
import numpy as np
from typing import Optional
from imblearn.over_sampling._smote.base import SMOTE, SMOTEN, SMOTENC, BaseOverSampler
from imblearn.over_sampling._smote.cluster import KMeansSMOTE
from imblearn.over_sampling._adasyn import ADASYN
from imblearn.over_sampling import BorderlineSMOTE, RandomOverSampler
from imblearn.over_sampling._smote.filter import SVMSMOTE

from imblearn.under_sampling.base import BaseUnderSampler
from imblearn.under_sampling import RandomUnderSampler

IMPLEMENTED_OVERSAMPLERS = {
    "SMOTE": SMOTE,
    "SMOTEN": SMOTEN,
    "SMOTENC": SMOTENC,
    "SVMSMOTE": SVMSMOTE,
    "BorderlineSMOTE": BorderlineSMOTE,
    "RandomOverSampler": RandomOverSampler
}
IMPLEMENTED_UNDERSAMPLERS = {
    "RandomUnderSampler": RandomUnderSampler
}

class GateSampler:

    def __init__(self,
                 target_size_per_gate: int,
                 oversampler: str = "SMOTE",
                 undersampler: str = "RandomUnderSampler",
                 gate_freq_cutoff: Optional[int] = None) -> None:
        
        self._validate_input_parameters(oversampler,
                                        undersampler)
        
        self.CELL_THRESHOLD = gate_freq_cutoff or 10
        self.target_size = target_size_per_gate

        if isinstance(oversampler, str):
            self.oversampler = IMPLEMENTED_OVERSAMPLERS[oversampler]
        else:
            self.oversampler = oversampler

        if isinstance(undersampler, str):
            self.undersampler = IMPLEMENTED_UNDERSAMPLERS[undersampler]
        else:
            self.undersampler = undersampler
        
        self.random_oversampler = IMPLEMENTED_OVERSAMPLERS["RandomOverSampler"]
    
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
                         y: np.ndarray) -> tuple[np.ndarray, dict]:
        uniques = np.unique(y, axis = 0)
        if y.ndim == 1:
            return {uniques[i]: i for i in range(uniques.shape[0])}, {i: uniques[i] for i in range(uniques.shape[0])}
        else:
            return {tuple(uniques[i]): i for i in range(uniques.shape[0])}, {i: tuple(uniques[i]) for i in range(uniques.shape[0])}

    def _apply_gate_map(self,
                        y: np.ndarray,
                        gate_map: dict) -> np.ndarray:
        if y.ndim == 1:
            return [gate_map.__getitem__(y[i]) for i in range(len(y))]
        tuple_list = list(map(tuple, y))
        return np.array(np.array([gate_map.__getitem__(tuple_list[i]) for i in range(len(tuple_list))]))

    def fit_resample(self,
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

        print("starting...")
        _y = self._convert_gate_matrix_to_binary(y)
        print("converted matrix to binary")
        
        gate_map, reverse_gate_map = self._map_gate_matrix(_y)
    
        _y = self._apply_gate_map(_y, gate_map)
        print("applied gate map")

        binary_classes, binary_classes_frequency = self._calculate_gate_frequencies(_y)
        classes_above_threshold_idxs, classes_below_threshold_idxs = self._calculate_threshold_indices(binary_classes,
                                                                                                       binary_classes_frequency,
                                                                                                       binary_gating = _y)

        print("found threshold indices")
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
        print("start sampling")
        print(oversampler_kwargs, undersampler_kwargs)
        self.oversampler: BaseOverSampler = self.oversampler(**oversampler_kwargs)
        self.undersampler: BaseUnderSampler = self.undersampler(**undersampler_kwargs)

        self.below_threshold_oversampler: BaseOverSampler = self.random_oversampler(**random_oversampler_kwargs)

        at_X, at_y = self._fit_resample_above_threshold(at_X,
                                                        at_y)

        bt_X, bt_y = self.below_threshold_oversampler.fit_resample(bt_X,
                                                                   bt_y)
        
        X, y = np.concatenate((at_X, bt_X)), np.concatenate((at_y, bt_y))

        y = np.array(self._apply_gate_map(y, reverse_gate_map), dtype = np.int64)


        y = self._convert_binary_to_gate_matrix(y)
        return self._shuffle(X, y) if shuffle else (X, y)

    def _fit_resample_above_threshold(self,
                                      X: np.ndarray,
                                      y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        print(X.shape, y.shape)
        X, y = self.undersampler.fit_resample(X,
                                              y)
        print(np.unique(y, return_counts = True))
        print(X.shape, y.shape)
        X, y = self.oversampler.fit_resample(X,
                                             y)
        print(np.unique(y, return_counts = True))
        print(X.shape, y.shape)
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
        print("converting bits to int...")
        ### this line is necessary to set all "sign bits" to 0
        y = np.hstack([np.zeros(shape = (y.shape[0], 1)), y])

        ### we have to convert the matrix into <64 wide columns because
        ### the bit conversion is limited to 64bit numbers
        y_list = np.array_split(y, np.ceil(y.shape[1]/64), axis = 1)
        print("array split")
        self._bit_ranges = [y_subset.shape[1] for y_subset in y_list]
        print("calculated bit ranges", self._bit_ranges)
        return np.vstack([self.__bits_to_int(y_subset) for y_subset in y_list]).T

    def _convert_binary_to_gate_matrix(self,
                                       y: np.ndarray) -> np.ndarray:
        if y.ndim == 1:
            return self.__int_to_bits(y)[:,1:]
        gate_list = [self.__int_to_bits(y[:,i], self._bit_ranges[i]) for i in range(y.shape[1])]
        return np.hstack(gate_list)[:,1:]

