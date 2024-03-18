from functools import wraps
from anndata import AnnData
from typing import Optional, Union
import warnings
import numpy as np
import pandas as pd
from itertools import combinations

from .exceptions._exceptions import (ChannelSubsetError,
                                     GateNotFoundError,
                                     GateAmbiguityError,
                                     PopulationAsGateError,
                                     ExhaustedGatePathError,
                                     GateNameError)
from .exceptions._utils import GateNotProvidedError, ExhaustedHierarchyError

reduction_names = {
    reduction: [f"{reduction}{i}" for i in range(1,50)] for reduction in ["PCA", "MDS", "UMAP", "TSNE"]
}

GATE_SEPARATOR = "/"

IMPLEMENTED_SAMPLEWISE_DIMREDS = ["MDS", "PCA", "UMAP", "TSNE"]
IMPLEMENTED_SCALERS = ["MinMaxScaler", "RobustScaler", "StandardScaler"]

cytof_technical_channels = ["event_length", "Event_length",
                            "width", "Width",
                            "height", "Height",
                            "center", "Center",
                            "residual", "Residual",
                            "offset", "Offset",
                            "amplitude", "Amplitude",
                            "dna1", "DNA1",
                            "dna2", "DNA2"]

scatter_channels = ["FSC", "SSC", "fsc", "ssc"]
time_channels = ["time", "Time"]
spectral_flow_technical_channels = ["AF"]

def _check_gate_name(gate: str) -> None:
    if gate.startswith(GATE_SEPARATOR) or gate.endswith(GATE_SEPARATOR):
        raise GateNameError(GATE_SEPARATOR)
    if not gate:
        raise GateNotProvidedError(gate)
    
def _check_gate_path(gate_path):
    _check_gate_name(gate_path)
    if not GATE_SEPARATOR in gate_path:
        raise PopulationAsGateError(gate_path)

def _is_parent(adata, gate, parent) -> bool:
    """Substring analysis to see if these are actually children"""
    parent_gate = _find_gate_path_of_gate(adata, parent)
    child_gate = _find_gate_path_of_gate(adata, gate) 
    return parent_gate in child_gate and parent_gate != child_gate

def _find_current_population(gate: str) -> str:
    """Finds current population of a specified gating path

    Parameters
    ----------

    gate: str, no default
        provided gating path

    Examples
    --------
    >>> find_current_population("root/singlets")
    singlets
    >>> find_current_population("root")
    root
    >>> find_current_population("root/singlets/T_cells")
    T_cells
    """
    _check_gate_name(gate)
    return gate.split(GATE_SEPARATOR)[-1]

def _find_gate_path_of_gate(adata: AnnData,
                            gate: str) -> str:
    """\
    Finds the gate path of the specified population
    This function looks into adata.uns["gating_cols"] and selects
    the entry that endswith the provided population

    Parameters
    ----------

    adata: AnnData
        the current dataset with an uns dict
    
    gate: str
        the population that is looked up

    Examples
    --------

    >>> adata = ad.AnnData(uns = {"gating_cols": pd.Index(["root/singlets"])})
    >>> find_gate_path_of_gate(adata, "singlets")
    "root/singlets"

    """
    _check_gate_name(gate)
    if GATE_SEPARATOR in gate:
        n_separators = gate.count(GATE_SEPARATOR)
        n_gates = n_separators + 1
        gates = [gate_path for gate_path in adata.uns["gating_cols"]
                 if _gate_path_length(gate_path) >= n_gates and
                 _extract_partial_gate_path_end(gate_path, n_gates) == gate]
    else:
        gates = [gate_path for gate_path in adata.uns["gating_cols"]
                 if _find_current_population(gate_path) == gate]
    if not gates:
        raise GateNotFoundError(gate)
    if len(gates) > 1:
        raise GateAmbiguityError(gates)
    return gates[0]

def _gate_path_length(gate_path: str) -> int:
    _check_gate_path(gate_path)
    return len(gate_path.split(GATE_SEPARATOR))

def _extract_partial_gate_path_end(gate_path: str,
                                   n_positions: int) -> str:
    _check_gate_path(gate_path)
    gate_components = gate_path.split(GATE_SEPARATOR)
    if len(gate_components) < n_positions:
        raise ExhaustedGatePathError(n_positions, len(gate_components))
    return GATE_SEPARATOR.join(gate_path.split(GATE_SEPARATOR)[-n_positions:])

def _extract_partial_gate_path_start(gate_path: str,
                                     n_positions: int) -> str:
    _check_gate_path(gate_path)
    gate_components = gate_path.split(GATE_SEPARATOR)
    if len(gate_components) < n_positions:
        raise ExhaustedGatePathError(n_positions, len(gate_components))
    return GATE_SEPARATOR.join(gate_path.split("/")[:n_positions])

def _find_gate_indices(adata: AnnData,
                       gate_columns: Union[str, list[str]]) -> list[int]:
    """Finds the index of provided populations in adata.uns["gating_cols"]
    This function is supposed to index columns provided as a string.
    That way, the indices can be used to access the sparse matrix
    in adata.obsm["gating"] that stores the gating values.

    Parameters
    ----------
    adata: AnnData
        the provided dataset
    
    gate_columns: Union[str, list[str]]:
        the gate columns that are supposed to be looked up

    Examples
    --------
    >>> adata = ad.Anndata(uns = {"gating_cols": pd.Index(["root/singlets",
                                                           "root/singlets/T_cells])}
    >>> find_gate_indices(adata, "root/singlets")
    [0]
    >>> find_gate_indices(adata, ["root/singlets", "root/singlets/T_cells"])
    [0,1]

    """

    if not isinstance(gate_columns, list):
        gate_columns = [gate_columns]
    return [adata.uns["gating_cols"].get_loc(gate) for gate in gate_columns]

def _find_parent_gate(gate: str) -> str:
    """Returns the parent gate path of the provided gate

    Parameters
    ----------
    gate: str
        the provided gate path

    Examples
    --------

    >>> find_parent_gate("root/singlets/T_cells")
    root/singlets
    >>> find_parent_gate("root")
    ExhaustedHierarchyError
    >>> find_parent_gate("root/singlets")
    root
    
    """
    _check_gate_name(gate)
    if GATE_SEPARATOR in gate:
        return GATE_SEPARATOR.join(gate.split(GATE_SEPARATOR)[:-1])
    else:
        raise ExhaustedHierarchyError(gate)

def _find_parent_population(gate: str) -> str:
    """Returns the parent population of the provided gate path

    Parameters
    ----------
    gate: str
        the provided gate path

    Examples
    --------
    >>> find_parent_population("root/singlets/T_cells")
    singlets
    >>> find_parent_population("root")
    ExhaustedHierarchyError
    >>> find_parent_population("root/singlets/")
    root
    """

    _check_gate_name(gate)
    if GATE_SEPARATOR in gate:
        return gate.split(GATE_SEPARATOR)[:-1][::-1][0]
    else:
        raise ExhaustedHierarchyError(gate)

def _find_grandparent_gate(gate: str) -> str:
    """Finds the grandparent gating path of a provided gate

    Parameters
    ----------
    gate: str
        the provided gating path

    Examples
    --------

    >>> find_grandparent_gate("root/singlets/T_cells")
    root
    >>> find_grandparent_gate("root/singlets/T_cells/cytotoxic")
    root/singlets
    >>> find_grandparent_gate("root/singlets")
    ExhaustedHieararchyError
    """
    _check_gate_name(gate)
    return _find_parent_gate(_find_parent_gate(gate))

def _find_grandparent_population(gate: str) -> str:
    """Finds the grandparent population of a provided gate

    Parameters
    ----------
    gate: str
        the provided gating path

    Examples
    --------

    >>> find_grandparent_gate("root/singlets/T_cells")
    root
    >>> find_grandparent_gate("root/singlets/T_cells/cytotoxic")
    singlets
    >>> find_grandparent_gate("root/singlets")
    ExhaustedHieararchyError
    """
    _check_gate_name(gate)
    return _find_parent_population(_find_parent_gate(gate))

def _find_parents_recursively(gate: str, parent_list = None) -> list[str]:
    """Finds all parent gates of a specified gate

    Parameters
    ----------
    gate: str
        provided gating path
    parent_list: None
        is instantiated to None because the function is used recursively

    Examples
    --------

    >>> find_parents_recursively("root/singlets/T_cells")
    ["root_singlets", "root"]
    >>> find_parents_recursively("root")
    ExhaustedHierarchyError
    """
    if parent_list is None:
        parent_list = []
    parent = _find_parent_gate(gate)
    parent_list.append(parent)
    if parent != "root":
        return _find_parents_recursively(parent, parent_list)
    return parent_list

def _find_children_of_gate(adata: AnnData,
                           query_gate: str) -> list[str]:
    gates = adata.uns["gating_cols"]
    return [gate for gate in gates if GATE_SEPARATOR.join(gate.split(GATE_SEPARATOR)[:-1]) == query_gate]

def _transform_gates_according_to_gate_transform(vertices: np.ndarray,
                                                 transforms: dict,
                                                 gate_channels: list[str]) -> np.ndarray:
    
    for i, gate_channel in enumerate(gate_channels):
        channel_transforms = [transform for transform in transforms if gate_channel in transform.id]
        if len(channel_transforms) > 1:
            transform = [transform for transform in channel_transforms if "Comp-" in transform.id][0]
        else:
            transform = channel_transforms[0]
        vertices[i] = transform.apply(vertices[i])
    return vertices

def _transform_vertices_according_to_gate_transform(vertices: np.ndarray,
                                                    transforms: dict,
                                                    gate_channels: list[str]) -> np.ndarray:
    for i, gate_channel in enumerate(gate_channels):
        channel_transforms = [transform for transform in transforms if gate_channel in transform.id]
        if len(channel_transforms) > 1:
            transform = [transform for transform in channel_transforms if "Comp-" in transform.id][0]
        else:
            transform = channel_transforms[0]
        vertices[:,i] = transform.apply(vertices[:,i])
    return vertices

def _inverse_transform_gates_according_to_gate_transform(vertices: np.ndarray,
                                                         transforms: dict,
                                                         gate_channels: list[str]) -> np.ndarray:
    
    for i, gate_channel in enumerate(gate_channels):
        channel_transforms = [transform for transform in transforms if gate_channel in transform.id]
        if len(channel_transforms) > 1:
            transform = [transform for transform in channel_transforms if "Comp-" in transform.id][0]
        else:
            transform = channel_transforms[0]
        vertices[i] = transform.inverse(vertices[i])
    return vertices

def _inverse_transform_vertices_according_to_gate_transform(vertices: np.ndarray,
                                                            transforms: dict,
                                                            gate_channels: list[str]) -> np.ndarray:
    
    for i, gate_channel in enumerate(gate_channels):
        channel_transforms = [transform for transform in transforms if gate_channel in transform.id]
        if len(channel_transforms) > 1:
            transform = [transform for transform in channel_transforms if "Comp-" in transform.id][0]
        else:
            transform = channel_transforms[0]
        vertices[:,i] = transform.inverse(vertices[:,i])
    return vertices

def _close_polygon_gate_coordinates(vertices: np.ndarray) -> np.ndarray:
    """Closes a polygon gate by adding the first coordinate to the bottom of the array

    Parameters
    ----------

    vertices: np.ndarray
        the array that contains the gate coordinates

    Examples
    --------
    >>> coordinates = np.array([[1,2], [3,4]])
    >>> close_polygon_gate_coordinates(coordinates)
    np.array([[1,2], [3,4], [1,2]])
    """
    return np.vstack([vertices, vertices[0]])


def _create_gate_lut(wsp_dict: dict[str, dict]) -> dict:
    #TODO: needs check for group...
    _gate_lut = {}
    gated_files = []
    for file in wsp_dict:

        _gate_lut[file] = {}
        gate_list = wsp_dict[file]["gates"]

        if gate_list:
            gated_files.append(file)

        for i, _ in enumerate(gate_list):
            
            gate_name = wsp_dict[file]["gates"][i]["gate"].gate_name.replace(" ", "_")
            _gate_lut[file][gate_name] = {}

            gate_path = GATE_SEPARATOR.join(list(wsp_dict[file]["gates"][i]["gate_path"])).replace(" ", "_")
            gate_channels = [dim.id
                             for dim in wsp_dict[file]["gates"][i]["gate"].dimensions]

            gate_dimensions = np.array([(dim.min, dim.max)
                                         for dim in wsp_dict[file]["gates"][i]["gate"].dimensions],
                                         dtype = np.float32)
            gate_dimensions = _inverse_transform_gates_according_to_gate_transform(gate_dimensions,
                                                                                   wsp_dict[file]["transforms"],
                                                                                   gate_channels)
            
            try:
                vertices = np.array(wsp_dict[file]["gates"][i]["gate"].vertices)
                vertices = _close_polygon_gate_coordinates(vertices)
                vertices = _inverse_transform_vertices_according_to_gate_transform(vertices,
                                                                                   wsp_dict[file]["transforms"],
                                                                                   gate_channels)
            except AttributeError:
                vertices = gate_dimensions

            
            
            _gate_lut[file][gate_name]["parent_path"] = gate_path
            _gate_lut[file][gate_name]["dimensions"] = gate_channels
            _gate_lut[file][gate_name]["full_gate_path"] = GATE_SEPARATOR.join([gate_path, gate_name])
            _gate_lut[file][gate_name]["gate_type"] = wsp_dict[file]["gates"][i]["gate"].__class__.__name__
            _gate_lut[file][gate_name]["gate_dimensions"] = gate_dimensions
            _gate_lut[file][gate_name]["vertices"] = vertices

    return _gate_lut

def _fetch_fluo_channels(adata: AnnData) -> list[str]:
    """compares channel names to a predefined list of common FACS and CyTOF channels"""
    return adata.var.loc[adata.var["type"] == "fluo"].index.tolist()

def subset_fluo_channels(adata: AnnData,
                         as_view: bool = False,
                         copy: bool = False) -> AnnData:
    adata = adata.copy() if copy else adata
    if as_view:
        return adata[:, adata.var["type"] == "fluo"]
    else:
        adata._inplace_subset_var(adata.var[adata.var["type"] == "fluo"].index)
    return adata if copy else None

def subset_channels(adata: AnnData,
                    channels: Optional[list[str]] = None,
                    use_panel: bool = False,
                    keep_state_channels: bool = True,
                    copy: bool = False) -> Optional[AnnData]:
    if not use_panel and channels is None:
        raise ChannelSubsetError
    
    if use_panel: ## overrides channels input.
        channels = adata.uns["panel"].dataframe["antigens"].to_list()
    
    if keep_state_channels:
        state_channels = [channel for channel in adata.var_names if any(k in channel.lower()
                                                                        for k in scatter_channels + time_channels + cytof_technical_channels + spectral_flow_technical_channels)]
        channels += state_channels

    adata = adata.copy() if copy else adata
    adata._inplace_subset_var(adata.var.loc[adata.var["pns"].isin(channels)].index.to_list())
    return adata if copy else None

def subset_gate(adata: AnnData,
                gate: Optional[str] = None,
                gate_path: Optional[str] = None,
                as_view: bool = False,
                copy: bool = False) -> AnnData:
    adata = adata.copy() if copy else adata
    
    if gate is None and gate_path is None:
        raise TypeError("Please provide either a gate name or a gate path.")
    
    gates: list[str] = adata.uns["gating_cols"].to_list()
    
    if gate:
        gate_path = _find_gate_path_of_gate(adata, gate)

    gate_idx = gates.index(gate_path)

    if as_view:
        return  adata[adata.obsm["gating"][:,gate_idx] == True,:]
    ### basically copying the individual steps from AnnData._inplace_subset_var
    ### potentially PR?
    subset = adata[adata.obsm["gating"][:,gate_idx] == True,:].copy()
    adata._init_as_actual(subset, dtype = None)
    return adata if copy else None

def equalize_groups(data: AnnData,
                    fraction: Optional[float] = None,
                    n_obs: Optional[int] = None,
                    on: Union[str, list[str]] = None, 
                    random_state: int = 187,
                    copy: bool = False
                    ) -> Optional[AnnData]:
    #TODO: add "min" as a parameter
    np.random.seed(random_state)
    if n_obs is not None:
        new_n_obs = n_obs
    elif fraction is not None:
        if fraction > 1 or fraction < 0:
            raise ValueError(f'`fraction` needs to be within [0, 1], not {fraction}')
        new_n_obs = int(data.obs.value_counts(on).min() * fraction)
    else:
        raise ValueError('Either pass `n_obs` or `fraction`.')
    
    if on is None:
        warnings.warn("Equalizing... groups to equalize are set to 'sample_ID'")
        on = "sample_ID"

    obs_indices = data.obs.groupby(on).sample(new_n_obs).index.to_numpy()

    if isinstance(data, AnnData):
        if copy:
            return data[obs_indices].copy()
        else:
            data._inplace_subset_obs(obs_indices)
    else:
        X = data
        return X[obs_indices], obs_indices
    
def contains_only_fluo(adata: AnnData) -> bool:
    return all(adata.var["type"] == "fluo")

def get_idx_loc(adata: AnnData,
                idx_to_loc: Union[list[str], pd.Index]) -> np.ndarray:
    return np.array([adata.obs_names.get_loc(idx) for idx in idx_to_loc])

def remove_unnamed_channels(adata: AnnData,
                            copy: bool = False) -> Optional[AnnData]:
    
    unnamed_channels = [channel for channel in adata.var.index if
                        channel not in adata.uns["panel"].dataframe["antigens"].to_list() and
                        adata.var.loc[adata.var["pns"] == channel, "type"].iloc[0] == "fluo"]
    named_channels = [channel for channel in adata.var.index if
                      channel not in unnamed_channels]
    non_fluo_channels = adata.var[adata.var["type"] != "fluo"].index.to_list()

    adata = adata.copy() if copy else adata
    adata._inplace_subset_var(list(set(named_channels + non_fluo_channels)))
    
    return adata if copy else None

def _flatten_nested_list(l):
    return [item for sublist in l for item in sublist]

def _is_valid_sample_ID(adata: AnnData,
                        string_to_check) -> bool:
    return string_to_check in adata.obs["sample_ID"].unique()

def _is_valid_filename(adata: AnnData,
                       string_to_check) -> bool:
    return string_to_check in adata.obs["file_name"].unique()

def is_fluo_channel(adata: AnnData,
                    channel: str) -> bool:
    return adata.var.loc[adata.var["pns"] == channel, "type"].iloc[0] == "fluo"

def _create_comparisons(data: pd.DataFrame,
                        groupby: str,
                        splitby: Optional[str],
                        n: int = 2) -> list[tuple[str, str]]:
    groupby_values = data[groupby].unique()
    if splitby:
        splitby_values = data[splitby].unique()
        vals = [(g, s)
                for g in groupby_values
                for s in splitby_values]
    else:
        vals = groupby_values
    return list(combinations(vals, n))

def convert_cluster_to_gate(adata: AnnData,
                            obs_column: str,
                            positive_cluster: Union[int, str, list[int], list[str]],
                            population_name: Optional[str],
                            parent_name: str,
                            copy: bool = False) -> Optional[AnnData]:
    from scipy.sparse import csr_matrix, hstack
    adata = adata.copy() if copy else adata
    full_parent = _find_gate_path_of_gate(adata, parent_name)
    full_gate = GATE_SEPARATOR.join([full_parent, population_name])
    if full_gate in adata.uns["gating_cols"]:
        raise TypeError("Gate already present. Please choose a different name!")

    if not isinstance(positive_cluster, list):
        positive_cluster = [positive_cluster]
    gate_list = adata.obs[obs_column]
    uniques = gate_list.unique()
    mapping = {cluster: cluster in positive_cluster for cluster in uniques}
    gate_matrix = csr_matrix(gate_list.map(mapping).values.reshape(len(gate_list), 1), dtype = bool)
    adata.obsm["gating"] = hstack([adata.obsm["gating"], gate_matrix])

    adata.uns["gating_cols"] = adata.uns["gating_cols"].append(pd.Index([full_gate]))

    return adata if copy else None
    
def convert_gate_to_obs(adata: AnnData,
                        gate: str,
                        key_added: Optional[str] = None,
                        copy: bool = False) -> Optional[AnnData]:
    adata = adata.copy() if copy else adata

    gate_path = _find_gate_path_of_gate(adata, gate)
    gate_index = _find_gate_indices(adata, gate_path)
    gate_id = key_added or gate
    adata.obs[gate_id] = adata.obsm["gating"][:,gate_index].todense()
    adata.obs[gate_id] = adata.obs[gate_id].map({True: gate_id, False: "other"})
    adata.obs[gate_id] = adata.obs[gate_id].astype("category")
    adata.obs[gate_id] = adata.obs[gate_id].cat.set_categories([gate_id, "other"])
    return adata if copy else None

def convert_gates_to_obs(adata: AnnData,
                         copy: bool = False) -> Optional[AnnData]:
    adata = adata.copy() if copy else adata
    for gate in adata.uns["gating_cols"]:
        convert_gate_to_obs(adata,
                            _find_current_population(gate),
                            copy = False)
    return adata if copy else None

def add_metadata_to_obs(adata: AnnData,
                        metadata_column: str,
                        copy: bool = False) -> Optional[AnnData]:
    adata = adata.copy() if copy else adata
    metadata = adata.uns["metadata"].dataframe.copy()
    metadata["sample_ID"] = metadata["sample_ID"].astype(adata.obs["sample_ID"].cat.categories.dtype)
    metadata = metadata.set_index("sample_ID")
    mapping = metadata.to_dict()
    specific_mapping = mapping[metadata_column]
    adata.obs[metadata_column] = adata.obs["sample_ID"].map(specific_mapping)
    return adata if copy else None

def rename_channel(adata: AnnData,
                   old_channel_name: str,
                   new_channel_name: str,
                   copy: bool = False) -> Optional[AnnData]:
    adata = adata.copy() if copy else adata
    # we need to rename it in the panel, the cofactors and var
    current_var_names = adata.var_names
    new_var_names = [var if var != old_channel_name else new_channel_name for var in current_var_names]
    adata.var.index = new_var_names
    adata.var["pns"] = adata.var.index.to_list()

    if "panel" in adata.uns and len(adata.uns["panel"].dataframe) > 0:
        adata.uns["panel"].rename_channel(old_channel_name, new_channel_name)
    
    if "cofactors" in adata.uns and len(adata.uns["cofactors"].dataframe) > 0:
        adata.uns["cofactors"].rename_channel(old_channel_name, new_channel_name)

    return adata if copy else None

def remove_channel(adata: AnnData,
                   channel: Union[str, list[str]],
                   as_view: bool = False,
                   copy: bool = False) -> Optional[AnnData]:
    if not isinstance(channel, list):
        channel = [channel]
    adata = adata.copy() if copy else adata
    if as_view:
        return adata[:, ~adata.var_names.isin(channel)]
    else:
        adata._inplace_subset_var(
            [var for var in adata.var_names if var not in channel]
        )
    return adata if copy else None

def convert_var_to_panel(adata: AnnData,
                 copy: bool = False) -> Optional[AnnData]:
    from .dataset._supplements import Panel
    adata = adata.copy() if copy else adata
    
    new_panel = pd.DataFrame(data = {"fcs_colname": adata.var["pnn"].to_list(),
                                     "antigens": adata.var["pns"].to_list()})
    adata.uns["panel"] = Panel(panel = new_panel)
    
    return adata if copy else None

def _default_layer(func):
    @wraps(func)
    def __add_default_layer(*args, **kwargs):
        if "layer" in kwargs and kwargs["layer"] is None or "layer" not in kwargs:
            from ._settings import settings
            kwargs["layer"] = settings.default_layer
        return func(*args, **kwargs)
    return __add_default_layer

def _default_gate(func):
    @wraps(func)
    def __add_default_gate(*args, **kwargs):
        if "gate" in kwargs and kwargs["gate"] is None or "gate" not in kwargs:
            from ._settings import settings
            kwargs["gate"] = settings._default_gate
        return func(*args, **kwargs)
    return __add_default_gate

def _default_gate_and_default_layer(func):
    @_default_gate
    @_default_layer
    @wraps(func)
    def __add_default_gate_and_default_layer(*args, **kwargs):
        return func(*args, **kwargs)
    return __add_default_gate_and_default_layer

def _enable_gate_aliases(func):
    @wraps(func)
    def __allow_gate_aliases(*args, **kwargs):
        if "gate" in kwargs:
            gate = kwargs["gate"]
            from ._settings import settings
            if gate in settings.gate_aliases:
                kwargs["gate"] = settings.gate_aliases[gate]
                print(f"Using the provided gate alias {gate} for gate {kwargs['gate']}")
        return func(*args, **kwargs)
    return __allow_gate_aliases

#def _default_layer(func):
#    argspec = inspect.getfullargspec(func)
#    position_count = len(argspec.args) - len(argspec.defaults)
#
#    def add_default_layer(*args, **kwargs):
#        defaults = dict(zip(argspec.args[position_count:], argspec.defaults))
#
#        used_kwargs = kwargs.copy()
#        used_kwargs.update(zip(argspec.args[position_count:], args[position_count:]))
#        
#        # we delete every default that is overwritten by the user
#        defaults = {
#            k: v for (k,v) in defaults.items()
#            if k not in used_kwargs
#        }
#
#        # if its still in defaults, its not set by the user 
#        # and we can set the settings default
#        from . import settings
#        if "layer" in defaults: 
#            defaults["layer"] = settings.default_layer
#        kwargs = {**used_kwargs, **defaults}
#        return func(*args, **kwargs)
#    return add_default_layer
#
#def _default_gate(func):
#    argspec = inspect.getfullargspec(func)
#    position_count = len(argspec.args) - len(argspec.defaults)
#
#    def add_default_gate(*args, **kwargs):
#        defaults = dict(zip(argspec.args[position_count:], argspec.defaults))
#
#        used_kwargs = kwargs.copy()
#        used_kwargs.update(zip(argspec.args[position_count:], args[position_count:]))
#        
#        # we delete every default that is overwritten by the user
#        defaults = {
#            k: v for (k,v) in defaults.items()
#            if k not in used_kwargs
#        }
#
#        # if its still in defaults, its not set by the user 
#        # and we can set the settings default
#        from . import settings
#        if "gate" in defaults: 
#            defaults["gate"] = settings.default_gate
#        kwargs = {**used_kwargs, **defaults}
#        return func(*args, **kwargs)
#    return add_default_gate
#
#def _default_gate_and_default_layer(func):
#    """
#    combines the functionality of _default_gate and _default_layer
#    until we fix this to be a chained decorator we have to live with code duplication...
#    """
#    argspec = inspect.getfullargspec(func)
#    position_count = len(argspec.args) - len(argspec.defaults)
#
#    def add_default_gate(*args, **kwargs):
#        defaults = dict(zip(argspec.args[position_count:], argspec.defaults))
#
#        used_kwargs = kwargs.copy()
#        used_kwargs.update(zip(argspec.args[position_count:], args[position_count:]))
#        
#        # we delete every default that is overwritten by the user
#        defaults = {
#            k: v for (k,v) in defaults.items()
#            if k not in used_kwargs
#        }
#
#        # if its still in defaults, its not set by the user 
#        # and we can set the settings default
#        from . import settings
#        if "gate" in defaults: 
#            defaults["gate"] = settings.default_gate
#        if "layer" in defaults: 
#            defaults["layer"] = settings.default_layer
#        kwargs = {**used_kwargs, **defaults}
#        return func(*args, **kwargs)
#    return add_default_gate
#
#
#