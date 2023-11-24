from anndata import AnnData

from typing import Literal, Union, Optional

from ._dr_samplewise import _perform_samplewise_dr
from .._utils import _default_layer

@_default_layer
def mds_samplewise(adata: AnnData,
                   data_group: Optional[Union[str, list[str]]] = "sample_ID",
                   data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                   layer: str = None,
                   use_only_fluo: bool = True,
                   exclude: Optional[Union[str, list, str]] = None,
                   scaling: Literal["MinMaxScaler", "RobustScaler", "StandardScaler"] = "MinMaxScaler",
                   n_components: int = 3,
                   copy = False,
                   *args,
                   **kwargs) -> Optional[AnnData]:

    adata = adata.copy() if copy else adata

    adata = _perform_samplewise_dr(adata = adata,
                                   reduction = "MDS",
                                   data_metric = data_metric,
                                   data_group = data_group,
                                   layer = layer,
                                   use_only_fluo = use_only_fluo,
                                   exclude = exclude,
                                   scaling = scaling,
                                   n_components = n_components,
                                   *args,
                                   **kwargs)

    return adata if copy else None

