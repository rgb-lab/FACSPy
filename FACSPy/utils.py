from anndata import AnnData
from typing import Optional, Union
import warnings
import numpy as np
import pandas as pd

from .exceptions.exceptions import ChannelSubsetError, GateNotFoundError
from .exceptions.utils import GateNotProvidedError, ExhaustedHierarchyError
from itertools import combinations

from typing import Any

reduction_names = {
    reduction: [f"{reduction}{i}" for i in range(1,4)] for reduction in ["PCA", "MDS", "UMAP", "TSNE"]
}

GATE_SEPARATOR = "/"

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

def find_current_population(gate: str) -> str:
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
    singlets
    """

    if not gate:
        raise GateNotProvidedError(gate)
    return gate.split(GATE_SEPARATOR)[-1]

def find_gate_path_of_gate(adata: AnnData,
                           gate: str) -> str:
    """Finds the gate path of the specified population
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
    try:
        return [gate_path for gate_path in adata.uns["gating_cols"]
                if gate_path.endswith(gate)][0]
    except IndexError as e:
        raise GateNotFoundError(gate) from e

def find_gate_indices(adata: AnnData,
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

def find_parent_gate(gate: str) -> str:
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

    if not gate:
        raise GateNotProvidedError(gate)
    if GATE_SEPARATOR in gate:
        return GATE_SEPARATOR.join(gate.split(GATE_SEPARATOR)[:-1])
    else:
        raise ExhaustedHierarchyError(gate)

def find_parent_population(gate: str) -> str:
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

    if not gate:
        raise GateNotProvidedError(gate)
    if GATE_SEPARATOR in gate:
        return gate.split(GATE_SEPARATOR)[:-1][::-1][0]
    else:
        raise ExhaustedHierarchyError(gate)

def find_grandparent_gate(gate: str) -> str:
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

    return find_parent_gate(find_parent_gate(gate))

def find_grandparent_population(gate: str) -> str:
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
    return find_parent_population(find_parent_gate(gate))

def find_parents_recursively(gate: str, parent_list = None) -> list[str]:
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
    parent = find_parent_gate(gate)
    parent_list.append(parent)
    if parent != "root":
        return find_parents_recursively(parent, parent_list)
    return parent_list
    
def subset_stained_samples(adata: AnnData,
                           copy: bool = False) -> Optional[AnnData]:
    """Subsets all stained samples from a anndata dataset
    
    Parameters
    ----------
    adata:
        the provided dataset
    copy: bool
        whether to copy and return the adata object or subset inplace

    Examples
    --------

    >>> adata = ad.AnnData(obs = pd.DataFrame({"staining": ["stained", "stained", "unstained"]}))
    >>> stained_adata = subset_stained_samples(adata, copy = True)
    >>> len(stained_adata)
    2
    >>> stained_adata.obs["staining"].to_list()
    ["stained", "stained"]

    """
    adata = adata.copy() if copy else adata
    adata._inplace_subset_obs(adata.obs[adata.obs["staining"] == "stained"].index)
    return adata if copy else None

def subset_unstained_samples(adata: AnnData,
                             copy: bool = False) -> Optional[AnnData]:

    """Subsets all unstained samples from a anndata dataset
    
    Parameters
    ----------
    adata:
        the provided dataset
    copy: bool
        whether to copy and return the adata object or subset inplace

    Examples
    --------

    >>> adata = ad.AnnData(obs = pd.DataFrame({"staining": ["stained", "stained", "unstained"]}))
    >>> unstained_adata = subset_unstained_samples(adata, copy = True)
    >>> len(unstained_adata)
    1
    >>> stained_adata.obs["staining"].to_list()
    ["unstained"]

    """    
    adata = adata.copy() if copy else adata
    adata._inplace_subset_obs(adata.obs[adata.obs["staining"] != "stained"].index)
    return adata if copy else None

def transform_gates_according_to_gate_transform(vertices: np.ndarray,
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

def transform_vertices_according_to_gate_transform(vertices: np.ndarray,
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

def inverse_transform_gates_according_to_gate_transform(vertices: np.ndarray,
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

def inverse_transform_vertices_according_to_gate_transform(vertices: np.ndarray,
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

def close_polygon_gate_coordinates(vertices: np.ndarray) -> np.ndarray:
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


def create_gate_lut(wsp_dict: dict[str, dict]) -> dict:
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
            gate_dimensions = inverse_transform_gates_according_to_gate_transform(gate_dimensions,
                                                                                  wsp_dict[file]["transforms"],
                                                                                  gate_channels)
            
            try:
                vertices = np.array(wsp_dict[file]["gates"][i]["gate"].vertices)
                vertices = close_polygon_gate_coordinates(vertices)
                vertices = inverse_transform_vertices_according_to_gate_transform(vertices,
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
    #_gate_lut = _remove_duplicates_from_gate_lut(_gate_lut)

    return _gate_lut

def fetch_fluo_channels(adata: AnnData) -> list[str]:
    """compares channel names to a predefined list of common FACS and CyTOF channels"""
    return adata.var.loc[adata.var["type"] == "fluo"].index.to_list()

def subset_fluo_channels(adata: AnnData,
                         copy: bool = False) -> AnnData:
    adata = adata.copy() if copy else adata
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
        gate_path = find_gate_path_of_gate(adata, gate)

    gate_idx = gates.index(gate_path)

    if as_view:
        return  adata[adata.obsm["gating"][:,gate_idx] == True,:]
    ### basically copying the individual steps from AnnData._inplace_subset_var
    ### potentially PR?
    subset = adata[adata.obsm["gating"][:,gate_idx] == True,:].copy()
    adata._init_as_actual(subset, dtype = None)
    return adata if copy else None

# def subset_gate_as_view(adata: AnnData,
#                         gate: Optional[str] = None,
#                         gate_path: Optional[str] = None,
#                         copy: bool = False) -> AnnData:
#     adata = adata.copy() if copy else adata
    
#     if gate is None and gate_path is None:
#         raise TypeError("Please provide either a gate name or a gate path.")
    
#     gates: list[str] = adata.uns["gating_cols"].to_list()
    
#     if gate:
#         gate_path = [gate_path for gate_path in gates if gate_path.endswith(gate)][0]

#     gate_idx = gates.index(gate_path)

#     return adata[adata.obsm["gating"][:,gate_idx] == True,:]

def equalize_groups(data: AnnData,
                    fraction: Optional[float] = None,
                    n_obs: Optional[int] = None,
                    on: Union[str, list[str]] = None, 
                    random_state:int = 0,
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
    
def annotate_metadata_samplewise(adata: AnnData,
                                 sample_ID: Union[str, int],
                                 annotation: Union[str, int],
                                 factor_name: str,
                                 copy: bool = False) -> Optional[AnnData]:
    
    adata = adata.copy() if copy else adata
    adata.obs.loc[adata.obs["sample_ID"] == sample_ID, factor_name] = annotation
    adata.obs[factor_name] = adata.obs[factor_name].astype("category")

    return adata if copy else None

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

def flatten_nested_list(l):
    return [item for sublist in l for item in sublist]


def get_filename(adata: AnnData,
                 sample_ID: str) -> str:
    return adata.uns["metadata"].dataframe.loc[adata.uns["metadata"].dataframe["sample_ID"] == sample_ID, "file_name"].iloc[0]

def create_comparisons(data: pd.DataFrame,
                       groupby: str,
                       n: int = 2) -> list[tuple[str, str]]:
    return list(combinations(data[groupby].unique(), n))

def ifelse(condition, true_val, false_val) -> Any:
    return true_val if condition else false_val

def convert_cluster_to_gate(adata: AnnData,
                            obs_column: str,
                            positive_cluster: Union[int, str, list[int], list[str]],
                            population_name: Optional[str],
                            parent_name: str,
                            copy: bool = False) -> Optional[AnnData]:
    from scipy.sparse import csr_matrix, hstack
    adata = adata.copy() if copy else adata
    full_parent = find_gate_path_of_gate(adata, parent_name)
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

def convert_obs_to_gate(adata: AnnData,
                        obs_column: str,
                        gate_name: Optional[str] = None,
                        negative_identifier: Optional[str] = "other"
                        ) -> Optional[AnnData]:
    gate_list = adata.obs[obs_column]
    unique_entries = gate_list.unique()
    if len(unique_entries) > 2:
        raise TypeError("To apply for gates, only two values are allowed")
    if negative_identifier not in unique_entries:
        raise TypeError("Please provide a negative identifier for the gate")
    if gate_name is None:
        raise TypeError("Please provide a gate path")
    positive_identifier = [entry for entry in unique_entries if entry != negative_identifier][0]
    mapped_gate = gate_list.map({negative_identifier: False,
                                 positive_identifier: True})
    
    from scipy.sparse import csr_matrix


def convert_gate_to_obs(adata: AnnData,
                        gate: str,
                        key_added: Optional[str] = None,
                        copy: bool = False) -> Optional[AnnData]:
    adata = adata.copy() if copy else adata

    gate_path = find_gate_path_of_gate(adata, gate)
    gate_index = find_gate_indices(adata, gate_path)
    gate_id = key_added or gate
    adata.obs[gate_id] = adata.obsm["gating"][:,gate_index].todense()
    adata.obs[gate_id] = adata.obs[gate_id].map({True: gate, False: "other"})
    adata.obs[gate_id] = adata.obs[gate_id].astype("category")
    return adata if copy else None

def convert_gates_to_obs(adata: AnnData,
                         copy: bool = False) -> Optional[AnnData]:
    adata = adata.copy() if copy else adata
    for gate in adata.uns["gating_cols"]:
        convert_gate_to_obs(adata,
                            find_current_population(gate),
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
                   copy: bool = False) -> Optional[AnnData]:
    if not isinstance(channel, list):
        channel = [channel]
    adata = adata.copy() if copy else adata
    adata._inplace_subset_var(
        [var for var in adata.var_names if var not in channel]
    )

    return adata if copy else None
