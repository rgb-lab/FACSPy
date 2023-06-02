from anndata import AnnData
from typing import Optional, Union
import warnings
import numpy as np
import pandas as pd

GATE_SEPARATOR = "/"

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
    
def subset_stained_samples(dataset: AnnData,
                           copy: bool = False) -> Optional[AnnData]:
    dataset = dataset.copy() if copy else dataset
    dataset._inplace_subset_obs(dataset.obs[dataset.obs["staining"] == "stained"].index)
    return dataset if copy else None

def subset_unstained_samples(dataset: AnnData,
                             copy: bool = False) -> Optional[AnnData]:
    dataset = dataset.copy() if copy else dataset
    dataset._inplace_subset_obs(dataset.obs[dataset.obs["staining"] != "stained"].index)
    return dataset if copy else None

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

            vertices = np.array([(dim.min, dim.max)
                                 for dim in wsp_dict[file]["gates"][i]["gate"].dimensions],
                                 dtype = np.float32)

            _gate_lut[file][gate_name]["parent_path"] = gate_path
            _gate_lut[file][gate_name]["dimensions"] = gate_channels
            _gate_lut[file][gate_name]["full_gate_path"] = GATE_SEPARATOR.join([gate_path, gate_name])
            _gate_lut[file][gate_name]["gate_type"] = wsp_dict[file]["gates"][i]["gate"].__class__.__name__
            _gate_lut[file][gate_name]["vertices"] = vertices
    #_gate_lut = _remove_duplicates_from_gate_lut(_gate_lut)

    return _gate_lut

def fetch_fluo_channels(dataset: AnnData) -> list[str]:
    return [
        channel
        for channel in dataset.var.index.to_list()
        if all(k not in channel.lower() for k in ["fsc", "ssc", "time"])
    ]

def subset_fluo_channels(dataset: AnnData,
                         copy: bool = False) -> AnnData:
    dataset = dataset.copy() if copy else dataset
    dataset._inplace_subset_var(dataset.var[dataset.var["type"] == "fluo"].index)
    return dataset if copy else None

def subset_channels(adata: AnnData, copy: bool = False) -> Optional[AnnData]:
    pass

def subset_gate(dataset: AnnData,
                gate: Optional[str] = None,
                gate_path: Optional[str] = None,
                as_view: bool = False,
                copy: bool = False) -> AnnData:
    dataset = dataset.copy() if copy else dataset
    
    if gate is None and gate_path is None:
        raise TypeError("Please provide either a gate name or a gate path.")
    
    gates: list[str] = dataset.uns["gating_cols"].to_list()
    
    if gate:
        gate_path = find_gate_path_of_gate(dataset, gate)

    gate_idx = gates.index(gate_path)

    if as_view:
        return  dataset[dataset.obsm["gating"][:,gate_idx] == True,:]
    ### basically copying the individual steps from AnnData._inplace_subset_var
    ### potentially PR?
    subset = dataset[dataset.obsm["gating"][:,gate_idx] == True,:].copy()
    dataset._init_as_actual(subset, dtype = None)
    return dataset if copy else None

# def subset_gate_as_view(dataset: AnnData,
#                         gate: Optional[str] = None,
#                         gate_path: Optional[str] = None,
#                         copy: bool = False) -> AnnData:
#     dataset = dataset.copy() if copy else dataset
    
#     if gate is None and gate_path is None:
#         raise TypeError("Please provide either a gate name or a gate path.")
    
#     gates: list[str] = dataset.uns["gating_cols"].to_list()
    
#     if gate:
#         gate_path = [gate_path for gate_path in gates if gate_path.endswith(gate)][0]

#     gate_idx = gates.index(gate_path)

#     return dataset[dataset.obsm["gating"][:,gate_idx] == True,:]

def equalize_groups(data: AnnData,
                    fraction: Optional[float] = None,
                    n_obs: Optional[int] = None,
                    on: Union[str, list[str]] = None, 
                    random_state:int = 0,
                    copy: bool = False
                    ) -> Optional[AnnData]:
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
    
def annotate_metadata_samplewise(dataset: AnnData,
                                 sample_ID: Union[str, int],
                                 annotation: Union[str, int],
                                 factor_name: str,
                                 copy: bool = False) -> Optional[AnnData]:
    
    dataset = dataset.copy() if copy else dataset
    dataset.obs.loc[dataset.obs["sample_ID"] == sample_ID, factor_name] = annotation
    dataset.obs[factor_name] = dataset.obs[factor_name].astype("category")

    return dataset if copy else None

def contains_only_fluo(dataset: AnnData) -> bool:
    return all(dataset.var["type"] == "fluo")

def get_idx_loc(dataset: AnnData,
                idx_to_loc: Union[list[str], pd.Index]) -> np.ndarray:
    return np.array([dataset.obs_names.get_loc(idx) for idx in idx_to_loc])