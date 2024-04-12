import numpy as np

from imblearn.over_sampling import BorderlineSMOTE, RandomOverSampler
from imblearn.over_sampling._smote.base import (SMOTE,
                                                SMOTEN,
                                                SMOTENC)
from imblearn.over_sampling._smote.cluster import KMeansSMOTE
from imblearn.over_sampling._adasyn import ADASYN
from imblearn.over_sampling._smote.filter import SVMSMOTE

from imblearn.combine import SMOTEENN

from imblearn.under_sampling import RandomUnderSampler

from typing import Optional


class GaussianOverSampler:

    def __init__(self,
                 sampling_strategy: Optional[dict] = None,
                 standard_deviation: float = 0.01,
                 random_state: int = 187) -> None:
        if sampling_strategy is None:
            sampling_strategy = {}
        self.sampling_strategy = sampling_strategy
        self.sd = standard_deviation
        np.random.seed(random_state)

    def fit_resample(self,
                     X: np.ndarray,
                     y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
                class_X_sampled = np.empty(
                    shape = (0, X.shape[1]),
                    dtype = np.float64
                )
                class_y_sampled = np.empty(
                    shape = (0, y.shape[1]),
                    dtype = np.int64
                )
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
                        subsampled_X, subsampled_y = self._random_subsample(
                            X_, y_, (_class_count - class_X_sampled.shape[0])
                        )
                        class_X_sampled = np.vstack(
                            [class_X_sampled, self._apply_noise(subsampled_X)]
                        )
                        class_y_sampled = np.vstack(
                            [class_y_sampled, subsampled_y]
                        )
                assert class_X_sampled.shape[0] == _class_count, class_X_sampled.shape  # noqa
                assert class_y_sampled.shape[0] == _class_count, class_y_sampled.shape  # noqa
                X_sampled = np.vstack([X_sampled, class_X_sampled])
                y_sampled = np.vstack([y_sampled, class_y_sampled])
        return X_sampled, y_sampled

    def _random_subsample(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          n: int) -> tuple[np.ndarray, np.ndarray]:
        idxs = np.random.choice(X.shape[0], n, replace = False)
        return X[idxs], y[idxs]

    def _apply_noise(self,
                     X: np.ndarray) -> np.ndarray:
        noise = np.random.normal(
            1, self.sd, X.shape[0] * X.shape[1]
        ).reshape(X.shape)
        return X * noise


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
                 target_size: Optional[int] = None,
                 target_size_per_gate: Optional[int] = None,
                 oversampler: str = "Gaussian",
                 undersampler: str = "RandomUnderSampler",
                 oversample_rare_cells: bool = False,
                 rare_cells_cutoff: Optional[int] = None,
                 rare_cells_target_size_per_gate: Optional[int] = None,
                 rare_cells_target_fraction: Optional[float] = None) -> None:

        self.target_size = target_size
        self.target_size_per_gate = target_size_per_gate

        self._rare_cell_cutoff = rare_cells_cutoff
        if self._rare_cell_cutoff is None:
            self.rare_cell_cutoff = 50
        else:
            self.rare_cell_cutoff = self._rare_cell_cutoff
        self.oversample_rare_cells = oversample_rare_cells

        self.rare_cells_target_fraction = rare_cells_target_fraction
        self.rare_cells_target_size_per_gate = rare_cells_target_size_per_gate

        self._oversampler = oversampler
        self._undersampler = undersampler

        self._validate_input_parameters()

        if self.rare_cells_target_fraction is not None:
            assert self.target_size is not None
            self.rare_cells_target_size = int(
                np.ceil(self.target_size * self.rare_cells_target_fraction)
            )
        else:
            assert rare_cells_target_size_per_gate is not None
            self.rare_cells_target_size_per_gate = int(
                rare_cells_target_size_per_gate
            )

        if isinstance(oversampler, str):
            self.oversampler = IMPLEMENTED_OVERSAMPLERS[self._oversampler]
        else:
            self.oversampler = self._oversampler

        if isinstance(undersampler, str):
            self.undersampler = IMPLEMENTED_UNDERSAMPLERS[self._undersampler]
        else:
            self.undersampler = self._undersampler

        self.random_oversampler = IMPLEMENTED_OVERSAMPLERS["RandomOverSampler"]

        self._bit_ranges = None

    def _validate_input_parameters(self):
        if self._oversampler not in IMPLEMENTED_OVERSAMPLERS and \
                isinstance(self._oversampler, str):
            raise NotImplementedError(
                f"{self._oversampler} is not implemented. "
                f"Please choose from {list(IMPLEMENTED_OVERSAMPLERS.keys())}"
            )
        if self._undersampler not in IMPLEMENTED_UNDERSAMPLERS and \
                isinstance(self._undersampler, str):
            raise NotImplementedError(
                f"{self._undersampler} is not implemented. "
                f"Please choose from {list(IMPLEMENTED_UNDERSAMPLERS.keys())}"
            )
        if self.target_size_per_gate and self.target_size:
            raise ValueError(
                "Please provide a target size or a target size per gate"
            )
        if not self.target_size and not self.target_size_per_gate:
            raise ValueError(
                "Please provide target size or target_size_per_gate"
            )
        if self.rare_cells_target_fraction and \
                self.rare_cells_target_size_per_gate:
            raise ValueError(
                "Please provide a target size or a target size for rare cells"
            )
        if self.rare_cells_target_fraction is None \
                and self.rare_cells_target_size_per_gate is None:
            raise ValueError(
                "Please provide target size or target_size_per_gate "
                "for rare cells"
            )
        if self.target_size_per_gate and self.rare_cells_target_fraction:
            raise ValueError("Invalid combination!")
        return

    def _subset_arrays_by_idx(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              idxs) -> tuple[np.ndarray, np.ndarray]:
        return X[idxs], y[idxs]

    def _random_subsample(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          n: int):
        np.random.seed(187)
        idxs = np.random.choice(X.shape[0], n, replace = False)
        return self._subset_arrays_by_idx(X, y, idxs)

    def _shuffle(self,
                 X: np.ndarray,
                 y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert X.shape[0] == y.shape[0]
        return self._random_subsample(X, y, X.shape[0])

    def _map_gate_matrix(self,
                         y: np.ndarray) -> tuple[dict, dict]:
        """
        Function maps binary integers to ascending integers.
        We need that in case we have more than 64 gates, because
        the binarized gating strategy will contain two 64bit integers.
        Returns the gate map and the reverse gate map
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

    def _find_optimal_gate_sample_size(self,
                                       frequencies: np.ndarray):
        assert self.target_size is not None
        total = self.target_size
        rare_cells = frequencies[frequencies <= self.rare_cell_cutoff].shape[0]
        other_cells = frequencies[frequencies > self.rare_cell_cutoff].shape[0]

        if rare_cells and self.oversample_rare_cells:
            self.rare_cells_target_size_per_gate = int(
                np.ceil(self.rare_cells_target_size / rare_cells)
            )
            rare_cells_total = rare_cells * \
                self.rare_cells_target_size_per_gate
            total -= rare_cells_total

        return int(np.ceil(total / other_cells))

    def fit_resample(self,
                     X: np.ndarray,
                     y: np.ndarray,
                     shuffle: bool = True,
                     oversampler_kwargs: Optional[dict] = None,
                     undersampler_kwargs: Optional[dict] = None
                     ) -> tuple[np.ndarray, np.ndarray]:

        if oversampler_kwargs is None:
            oversampler_kwargs = {}
        if undersampler_kwargs is None:
            undersampler_kwargs = {}

        y_binary = self._convert_gate_matrix_to_binary(y)
        gate_map, reverse_gate_map = self._map_gate_matrix(y_binary)

        y_mapped = self._apply_gate_map(y_binary, gate_map)
        y_mapped = y_mapped.reshape(y_mapped.shape[0], 1)

        binary_classes, binary_classes_frequency = \
            self._calculate_gate_frequencies(y_mapped)

        X_sampled = np.empty(shape = (0, X.shape[1]),
                             dtype = np.float64)
        y_mapped_sampled = np.empty(shape = (0, y_mapped.shape[1]),
                                    dtype = np.int64)

        if self.target_size_per_gate is None:
            self.target_size_per_gate = \
                self._find_optimal_gate_sample_size(binary_classes_frequency)

        rare_cell_classes = [cls for
                             cls, count in zip(binary_classes,
                                               binary_classes_frequency)
                             if count <= self.rare_cell_cutoff]
        above_thresholds = {}
        below_thresholds = {}
        for bin_class, count in zip(binary_classes, binary_classes_frequency):
            if bin_class in rare_cell_classes:
                if not self.oversample_rare_cells:
                    continue
                if count > self.rare_cells_target_size_per_gate:
                    above_thresholds[bin_class] = \
                        self.rare_cells_target_size_per_gate
                else:
                    below_thresholds[bin_class] = \
                        self.rare_cells_target_size_per_gate
            else:
                if count > self.target_size_per_gate:
                    above_thresholds[bin_class] = self.target_size_per_gate
                else:
                    below_thresholds[bin_class] = self.target_size_per_gate

        for bin_class, count in above_thresholds.items():
            idxs = np.where(y_mapped == bin_class)[0]
            X_, y_ = self._random_subsample(
                X[idxs], y_mapped[idxs], min(self.target_size_per_gate, count)
            )
            X_sampled = np.vstack([X_sampled, X_])
            y_mapped_sampled = np.vstack([y_mapped_sampled, y_])

        if below_thresholds:
            below_threshold_gates = list(below_thresholds.keys())

            below_k_neighbors_frequencies = {
                _class: _freq
                for _class, _freq
                in zip(binary_classes, binary_classes_frequency)
                if (_class in below_threshold_gates) and (_freq <= 6)
            }
            below_k_neighbors_gates = list(
                below_k_neighbors_frequencies.keys()
            )

            above_k_neighbors_frequencies = {
                _class: _freq
                for _class, _freq
                in zip(binary_classes, binary_classes_frequency)
                if (_class in below_threshold_gates) and (_freq > 6)
            }
            above_k_neighbors_gates = list(
                above_k_neighbors_frequencies.keys()
            )

            idxs = np.where(np.isin(y_mapped, above_k_neighbors_gates))[0]
            if idxs.shape[0] > 0:
                assert not isinstance(self.oversampler, str)
                if np.unique(y_mapped[idxs]).size > 1 or \
                        self.oversampler().__class__.__name__ \
                        not in ["SMOTE", "SVMSMOTE"]:
                    oversampler = self.oversampler(
                        sampling_strategy = {
                            k: v for k, v in below_thresholds.items()
                            if k in above_k_neighbors_frequencies
                        },
                        **oversampler_kwargs
                    )
                    X_, y_ = oversampler.fit_resample(X[idxs], y_mapped[idxs])
                    y_ = y_.reshape(y_.shape[0], 1)
                    X_sampled = np.vstack([X_sampled, X_])
                    y_mapped_sampled = np.vstack([y_mapped_sampled, y_])
                else:
                    print(
                        "Warning... Omitting class because it was the only one"
                    )

            idxs = np.where(np.isin(y_mapped, below_k_neighbors_gates))[0]
            if idxs.shape[0] > 0:
                oversampler = GaussianOverSampler(
                    sampling_strategy = {
                        k: v for k, v in below_thresholds.items()
                        if k in below_k_neighbors_frequencies
                    }
                )

                X_, y_ = oversampler.fit_resample(X[idxs], y_mapped[idxs])
                y_ = y_.reshape(y_.shape[0], 1)
                X_sampled = np.vstack([X_sampled, X_])
                y_mapped_sampled = np.vstack([y_mapped_sampled, y_])

        y_binary_sampled = self._apply_gate_map(
            y_mapped_sampled, reverse_gate_map
        )
        y_sampled = self._convert_binary_to_gate_matrix(
            y_binary_sampled
            .astype(np.int64)
            .reshape(y_binary_sampled.shape[0],
                     y_binary.shape[1])
        )

        return (
            self._shuffle(X_sampled, y_sampled)
            if shuffle else (X_sampled, y_sampled)
        )

    def _calculate_gate_frequencies(self,
                                    y: np.ndarray
                                    ) -> tuple[list, np.ndarray]:
        binary_classes, binary_classes_frequency = np.unique(
            y, axis = 0, return_counts = True
        )
        binary_classes = [
            bin_class[0]
            for bin_class in binary_classes
        ]
        return binary_classes, binary_classes_frequency

    def __bits_to_int(self,
                      y: np.ndarray) -> np.ndarray:
        return y.dot(1 << np.arange(y.shape[-1] - 1, -1, -1, dtype = np.int64))

    def __int_to_bits(self,
                      y: np.ndarray,
                      bitrange: int) -> np.ndarray:
        return (((y[:, None] & (1 << np.arange(bitrange, dtype = np.int64))[::-1])) > 0).astype(int)  # noqa

    def _convert_gate_matrix_to_binary(self,
                                       y: np.ndarray) -> np.ndarray:
        # this line is necessary to set all "sign bits" to 0
        y = np.hstack([np.zeros(shape = (y.shape[0], 1)), y])

        # we have to convert the matrix into <64 wide columns because
        # the bit conversion is limited to 64bit numbers
        y_list = np.array_split(y, np.ceil(y.shape[1] / 64), axis = 1)
        if self._bit_ranges is None:
            self._bit_ranges = [y_subset.shape[1] for y_subset in y_list]
        return np.vstack([self.__bits_to_int(y_subset) for y_subset in y_list]).astype(np.int64).T  # noqa

    def _convert_binary_to_gate_matrix(self,
                                       y: np.ndarray) -> np.ndarray:
        assert self._bit_ranges is not None
        if y.shape[1] == 1:
            y = np.squeeze(y)
        if y.ndim == 1:
            return self.__int_to_bits(y, self._bit_ranges[0])[:, 1:]
        gate_list = [
            self.__int_to_bits(y[:, i], self._bit_ranges[i])
            for i in range(y.shape[1])
        ]
        return np.hstack(gate_list)[:, 1:]
