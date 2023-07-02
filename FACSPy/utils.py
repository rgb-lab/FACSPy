from anndata import AnnData
from typing import Optional, Union, Literal
import warnings
import numpy as np
import pandas as pd

from .exceptions.exceptions import ChannelSubsetError

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

def find_current_population(gate: str) -> str:
    return gate.split(GATE_SEPARATOR)[-1]

def find_gate_path_of_gate(adata: AnnData,
                           gate: str) -> str:
    return [gate_path for gate_path in adata.uns["gating_cols"]
            if gate_path.endswith(gate)][0]

def find_gate_indices(adata: AnnData,
                      gate_columns):
    if not isinstance(gate_columns, list):
        gate_columns = [gate_columns]
    return [adata.uns["gating_cols"].get_loc(gate) for gate in gate_columns]

def find_parent_gate(gate: str) -> str:
    """returns the parent gate path"""
    """Example: gate = 'root/singlets/T_cells' -> 'root/singlets' """
    return GATE_SEPARATOR.join(gate.split(GATE_SEPARATOR)[:-1])

def find_parent_population(gate: str) -> str:
    """returns the parent population"""
    """Example: gate = 'root/singlets/T_cells' -> 'singlets'"""
    return gate.split(GATE_SEPARATOR)[:-1][::-1][0]

def find_grandparent_gate(gate: str) -> str:
    return find_parent_gate(find_parent_gate(gate))

def find_grandparent_population(gate: str) -> str:
    return find_parent_population(find_parent_population(gate))

def find_parents_recursively(gate: str, parent_list = None):
    if parent_list is None:
        parent_list = []
    parent = find_parent_gate(gate)
    parent_list.append(parent)
    if parent != "root":
        return find_parents_recursively(parent, parent_list)
    return parent_list
    
def subset_stained_samples(adata: AnnData,
                           copy: bool = False) -> Optional[AnnData]:
    adata = adata.copy() if copy else adata
    adata._inplace_subset_obs(adata.obs[adata.obs["staining"] == "stained"].index)
    return adata if copy else None

def subset_unstained_samples(adata: AnnData,
                             copy: bool = False) -> Optional[AnnData]:
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
        vertices[i] = transform.inverse(vertices[i])
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
        vertices[:,i] = transform.inverse(vertices[:,i])
    return vertices

def close_polygon_gate_coordinates(vertices: np.ndarray) -> np.ndarray:
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
            gate_dimensions = transform_gates_according_to_gate_transform(gate_dimensions,
                                                                          wsp_dict[file]["transforms"],
                                                                          gate_channels)
            
            try:
                vertices = np.array(wsp_dict[file]["gates"][i]["gate"].vertices)
                vertices = close_polygon_gate_coordinates(vertices)
                vertices = transform_vertices_according_to_gate_transform(vertices,
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
                                                                        for k in scatter_channels + time_channels + cytof_technical_channels)]
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

def convert_gate_to_obs(adata: AnnData,
                        gate: str,
                        key_added: Optional[str] = None,
                        copy: bool = False) -> Optional[AnnData]:
    gate_path = find_gate_path_of_gate(adata, gate)
    gate_index = find_gate_indices(adata, gate_path)
    adata = adata.copy() if copy else adata
    adata.obs[gate or key_added] = adata.obsm["gating"][:,gate_index].todense()
    adata.obs[gate or key_added] = adata.obs[gate or key_added].map({True: gate, False: "other"})
    adata.obs[gate or key_added] = adata.obs[gate or key_added].astype("category")
    return adata if copy else None