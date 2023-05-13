
## gate: "root/singlets/tcells"

import anndata as ad
from typing import Optional, Union, Any

def find_parent_gate(gate: str) -> str:
    return "/".join(gate.split("/")[:-1])

def find_grandparent_gate(gate: str) -> str:
    return find_parent_gate(find_parent_gate(gate))

def find_parents_recursively(gate: str):
    parent = find_parent_gate(gate)
    if parent != "root":
        return find_parents_recursively(parent)

def find_parents_recursively(gate: str, parent_list = None):
    if parent_list is None:
        parent_list = []
    parent = find_parent_gate(gate)
    parent_list.append(parent)
    if parent != "root":
        return find_parents_recursively(parent, parent_list)
    return parent_list
    
def subset_fluo_channels(dataset: ad.AnnData) -> ad.AnnData:
    return dataset[:, dataset.var["type"] == "fluo"]


import warnings
import numpy as np
def equalize_groups(data: ad.AnnData,
              fraction: Optional[float] = None,
              n_obs: Optional[int] = None,
              on: Union[str, list[str]] = None, 
              random_state:int = 0,
              copy: bool = False
              ) -> Optional[ad.AnnData]:
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

    if isinstance(data, ad.AnnData):
        if copy:
            return data[obs_indices].copy()
        else:
            data._inplace_subset_obs(obs_indices)
    else:
        X = data
        return X[obs_indices], obs_indices