from anndata import AnnData

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from umap import UMAP

from typing import Optional, Union, Literal

from ._utils import (_save_samplewise_dr_settings,
                     _warn_user_about_changed_setting,
                     _warn_user_about_insufficient_sample_size)

from .._utils import (_fetch_fluo_channels,
                      IMPLEMENTED_SCALERS,
                      reduction_names)
from ..exceptions._exceptions import (AnalysisNotPerformedError,
                                      InvalidScalingError)
from ..plotting._utils import (_scale_data,
                               _select_gate_from_multiindex_dataframe)

def _perform_dr(reduction: Literal["PCA", "MDS", "UMAP", "TSNE"],
                data: np.ndarray,
                n_components: int = 3,
                *args,
                **kwargs) -> np.ndarray:
    if "random_state" not in kwargs:
        kwargs["random_state"] = 187
    if reduction == "PCA":
        return PCA(n_components = n_components,
                   *args,
                   **kwargs).fit_transform(data)
    if reduction == "MDS":
        if "normalized_stress" not in kwargs:
            _warn_user_about_changed_setting("MDS",
                                             "normalized_stress",
                                             "auto",
                                             "this avoids a future warning")
            kwargs["normalized_stress"] = "auto"
        return MDS(n_components = n_components,
                   *args,
                   **kwargs).fit_transform(data)
    if reduction == "TSNE":
        # perplexity has to be smaller than n_samples
        # but is set to 30 by default. Because samples
        # are not as frequent, we set the perplexity to
        # the smallest possible value while still allowing the
        # user to manually overwrite it
        if "perplexity" not in kwargs:
            _warn_user_about_changed_setting("TSNE",
                                             "perplexity",
                                             f"{min(30, data.shape[0]-1)}",
                                             "this avoids a value error")

            kwargs["perplexity"] = min(30, data.shape[0]-1)
        if n_components > 3 and "method" not in kwargs:
            _warn_user_about_changed_setting("TSNE",
                                             "method",
                                             "exact",
                                             "this avoids a value error")
            kwargs["method"] = "exact" # avoids value_error
        if "learning_rate" not in kwargs:
            kwargs["learning_rate"] = "auto"
        if "init" not in kwargs:
            kwargs["init"] = "pca"
        return TSNE(n_components = n_components,
                    *args,
                    **kwargs).fit_transform(data)
    if reduction == "UMAP":
        # you cannot use spectral initialisation of umap if
        # n_components is greater or equal to the number of samples
        # Here, we set the init parameter to random to overcome this
        # while still allowing the override by the user (with accompanying)
        # errors
        # https://github.com/lmcinnes/umap/issues/201
        if n_components >= data.shape[0]:
            if "init" not in kwargs:
                _warn_user_about_changed_setting("UMAP",
                                                "init",
                                                "random",
                                                "this avoids a value error")
                kwargs["init"] = "random"
        return UMAP(n_components = n_components,
                    *args,
                    **kwargs).fit_transform(data)


def _perform_samplewise_dr(adata: AnnData,
                           reduction: Literal["PCA", "MDS", "TSNE", "UMAP"],
                           data_metric: Literal["mfi", "fop"],
                           data_group: str,
                           layer: Union[Literal["compensated", "transformed"], str],
                           use_only_fluo: bool,
                           exclude: Optional[Union[list[str], str]],
                           scaling: Literal["MinMaxScaler", "RobustScaler", "StandardScaler"],
                           n_components: int,
                           *args,
                           **kwargs) -> AnnData:
    
    exclude = [] if exclude is None else exclude

    _save_samplewise_dr_settings(adata = adata,
                                 data_group = data_group,
                                 data_metric = data_metric,
                                 layer = layer,
                                 use_only_fluo = use_only_fluo,
                                 exclude = exclude,
                                 scaling = scaling,
                                 reduction = reduction.lower(),
                                 n_components = n_components,
                                 **kwargs)
    
    if use_only_fluo:
        columns_to_analyze = [
            channel for channel in _fetch_fluo_channels(adata)
            if channel not in exclude
        ]
    else:
        columns_to_analyze = [
            channel for channel in adata.var_names.tolist()
            if channel not in exclude
        ]

    table_identifier = f"{data_metric}_{data_group}_{layer}"

    if table_identifier not in adata.uns:
        raise AnalysisNotPerformedError(analysis = data_metric)
    
    if scaling not in IMPLEMENTED_SCALERS and scaling is not None:
        raise InvalidScalingError(scaler = scaling)

    data: pd.DataFrame = adata.uns[table_identifier]
    return_data = data.copy()

    data = data.loc[:, columns_to_analyze]
    gates = data.index.get_level_values("gate").unique()
    coord_columns = reduction_names[reduction]

    for gate in gates:
        gate_specific_data = _select_gate_from_multiindex_dataframe(data, gate)

        if gate_specific_data.shape[0] <= 1:
            # special case because all the algorithms would fail
            # we set to pd.NA and continue to avoid try/except blocks
            _warn_user_about_insufficient_sample_size(gate,
                                                      gate_specific_data.shape[0],
                                                      n_components)
            return_data.loc[
                return_data.index.get_level_values("gate") == gate,
                coord_columns[:n_components]
            ] = pd.NA
            continue

        if gate_specific_data.shape[0] == 2 and reduction == "UMAP":
            # umap does not allow only two points
            _warn_user_about_insufficient_sample_size(gate,
                                                      gate_specific_data.shape[0],
                                                      n_components)
            return_data.loc[
                return_data.index.get_level_values("gate") == gate,
                coord_columns[:n_components]
            ] = pd.NA
            continue

        if gate_specific_data.shape[0] < n_components:
            _warn_user_about_insufficient_sample_size(gate,
                                                      gate_specific_data.shape[0],
                                                      n_components)
            n_components = gate_specific_data.shape[0]

        gate_specific_data = _scale_data(gate_specific_data, scaling = scaling)
        
        coords = _perform_dr(reduction,
                             gate_specific_data,
                             n_components = n_components,
                             *args,
                             **kwargs)
        return_data.loc[
            return_data.index.get_level_values("gate") == gate,
            coord_columns[:n_components]
        ] = coords

    adata.uns[table_identifier] = return_data

    return return_data
