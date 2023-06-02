import numpy as np
import pandas as pd

from anndata import AnnData
import scanpy as sc

import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch

from typing import Union, Optional


from ..utils import (create_gate_lut,
                     find_parent_gate,
                     GATE_SEPARATOR,
                     find_parent_population,
                     subset_gate,
                     find_gate_indices,
                     find_gate_path_of_gate)

from .utils import turn_off_missing_plot


class GatingStrategyGrid:
    """Object to create and store a GatingStrategyGrid for one sample"""
    def __init__(self,
                 gate_lut: dict) -> None:
        
        self.gate_lut = gate_lut
        self.gating_grid = self._create_grid()

    def __repr__(self):
        return (f"{self.__class__.__name__}")
    
    def _get_max_gating_depth(self,
                              gating_dimensions: list[tuple[str, int]]) -> int:
        """returns the gate path lengths to determine the maximal strategy 'depth'"""
        return max(entry[1] for entry in gating_dimensions)

    def _get_gate_path_length(self,
                         gate_path: str) -> int:
        """returns the number of element of a gating path"""
        return len(gate_path.split(GATE_SEPARATOR))

    def _get_gating_depths(self) -> list[tuple[str, int]]:
        """returns the number of elements in a gating path per gate"""
        return [
            (gate, self._get_gate_path_length(description["parent_path"]))
            for (gate, description) in self.gate_lut.items()
        ]
    
    def find_gates_at_nth_depth(self,
                           n: int,
                           gating_dimensions: list[tuple[str, int]]) -> list[str]:
        return [gate for gate, gate_length in gating_dimensions if gate_length == n]
    
    def parent_gate_path(self,
                         gate: str) -> str:
        return self.gate_lut[gate]["parent_path"]    
    
    def full_gate_path(self,
                       gate: str) -> str:
        return self.gate_lut[gate]["full_gate_path"]
    

    def _create_gating_hierarchy_map(self,
                                     gating_dimensions: list[tuple[str, int]],
                                     max_gating_strategy_depth: int) -> dict[int: list[str]]:
        
        ## dict to store gates
        hierarchy_map = {k:[] for k in range(max_gating_strategy_depth)}

        for n in range(max_gating_strategy_depth, 1, -1):
            # fetch gates with a specific length from the gate length lookuptable (gate_len_lut)
            gates_at_nth_depth = self.find_gates_at_nth_depth(n = n,
                                                              gating_dimensions = gating_dimensions)
            hierarchy_map[n - 1] += gates_at_nth_depth
            # fetch parent gates from current gates...
            # ...and add them into the level above
            hierarchy_map[n - 2] += [find_parent_population(self.full_gate_path(gate)) for gate in gates_at_nth_depth]

        ## processes the hierarchy map so that empty values are removed and unique gates are kept
        ## example: {0: ['singlets'], 1: ['T_cells', 'B_cells']}
        hierarchy_map = {k: [entry for entry in set(hierarchy_map[k]) if entry] for k in hierarchy_map}
        
        return hierarchy_map

    def _dict_equal_elementwise(self,
                                dict_1: dict,
                                dict_2: dict,
                                keys_to_compare: list[str]) -> bool:
        return all(dict_1[key] == dict_2[key] for key in keys_to_compare)

    def _map_gating_groups(self):
        _gate_group_map = {} # stores potentially duplicate items
        gate_group_map = {}

        # look up every gating strategy that is repeated to map gates into groups
        # a group is characterized by having the same parent and the same dimensions
        # so we can show them in the same plot in the end.
        for i, (gate, _) in enumerate(self.gate_lut.items()):
            parent_gate_of_interest = self.gate_lut[gate]
            repeated_gating_strategies = [key for (key, value) in self.gate_lut.items()
                                          if self._dict_equal_elementwise(value, parent_gate_of_interest, ["parent_path", "dimensions"])]
            if len(repeated_gating_strategies) > 1:
                _gate_group_map[i] = repeated_gating_strategies
        
        # remove duplicate entries
        for key, value in _gate_group_map.items():
            if value not in gate_group_map.values():
                gate_group_map[key] = value

        return gate_group_map

    def _initialize_grid(self,
                         hierarchy_map: dict[int: list[str]]):
        return pd.DataFrame(dict([(k, pd.Series(v)) for k, v in hierarchy_map.items()]))

    def _create_grid(self) -> pd.DataFrame:

        gating_dimensions = self._get_gating_depths()
        max_gating_strategy_depth = self._get_max_gating_depth(gating_dimensions)

        hierarchy_map = self._create_gating_hierarchy_map(gating_dimensions,
                                                          max_gating_strategy_depth)

        self.gate_group_map = self._map_gating_groups()

        idx_map = self._initialize_grid(hierarchy_map)

        if idx_map.shape[1] == 1:
            # meaning that we have a one-dimensional gating strategy that
            # can be plotted in one single row
            return
        
        idx_map = self._replace_group_gates_with_groups(idx_map = idx_map,
                                                        gate_group_map = self.gate_group_map)
        
        
        idx_map = self._merge_groups_to_single_index(idx_map)
        
        #self.idx_map = self._concatenate_single_indices(idx_map)
        
        self.idx_map = self._center_indices_in_map(idx_map)

        idx_map = idx_map.reset_index(drop = True).T.reset_index(drop = True)

        return idx_map

    def _concatenate_single_indices(self,
                                    idx_map: pd.DataFrame) -> pd.DataFrame:
        
        single_values = {}
        for col in idx_map.columns:
            entries = list(idx_map[col].dropna())
            if len(entries) == 1:
                single_values[col] = entries[0]

        gates_to_merge = [value for (key, value) in single_values.items()]
        highest_depth = max(list(single_values))
        merged_column = gates_to_merge + [np.nan for _ in range(len(idx_map)-len(gates_to_merge))]
        idx_map[highest_depth] = merged_column
        idx_map = idx_map.drop(
            [key for key in single_values if key != highest_depth], axis=1
        )

        return idx_map
    
    def _center_indices_in_map(self,
                               idx_map: pd.DataFrame) -> pd.DataFrame:

        for col in idx_map.columns:
            non_nan_entries = list(idx_map[col].dropna())
            final_idx_positions = [int(len(idx_map)/(len(non_nan_entries) + 1) * i) for i in range(1, len(non_nan_entries) + 1)]
            reindexed_col = [np.nan for _ in range(len(idx_map))]
            for i, index_position in enumerate(final_idx_positions):
                reindexed_col[index_position] = non_nan_entries[i]
            idx_map[col] = reindexed_col

        return idx_map
          
    def _merge_groups_to_single_index(self,
                                      idx_map: pd.DataFrame) -> pd.DataFrame:
        
        for col in idx_map.columns:
            unique_entries = set(idx_map[col])
            idx_map[col] = list(unique_entries) + [np.nan for _ in range(len(idx_map)-len(unique_entries))]

        return idx_map.dropna(axis = 0, how = "all")   
    
    def _replace_group_gates_with_groups(self,
                                          idx_map: pd.DataFrame,
                                          gate_group_map: dict) -> pd.DataFrame:
        """maps the gate groups to the individual gates and """
        """assigns a group index if gate is part of a group"""
        for (key, values) in gate_group_map.items():
            for value in values:
                idx_map[idx_map == value] = f"group-{key}"

        return idx_map.sort_values(idx_map.columns.to_list())

def get_quadrant(gate_lut: dict,
                 gate: str) -> int:
    # quadrant naming similar to flowjo:
    #[1,2]
    #[4,3]
    
    vertices = gate_lut[gate]["vertices"]
    if vertices[0,0] is None: # x dimension is from None (-inf)
        if vertices[0,1] is None:
            raise ValueError()
        return 4 if vertices[1,0] is None else 1
    assert vertices[0,1] is None
    return 3 if vertices[1,0] is None else 2

def extract_gate_lut(adata: AnnData,
                     wsp_group: str,
                     file_name: str) -> dict[str: dict[str: Union[list[str], str]]]:
    
    return create_gate_lut(adata.uns["workspace"][wsp_group])[file_name]

def map_sample_ID_to_filename(adata:AnnData,
                              sample_ID: str) -> str:
    metadata = adata.uns["metadata"].to_df()
    return metadata.loc[metadata["sample_ID"] == sample_ID, "file_name"].iloc[0]

def prepare_plot_data(adata: AnnData,
                      parent_gating_path: str,
                      gate_list: Union[list, str],
                      x_channel: str,
                      y_channel: str,
                      sample_size: int) -> pd.DataFrame:
    
    if not isinstance(gate_list, list):
        gate_list = [gate_list]
    
    adata_subset = gate_parent_in_adata(adata,
                                        parent_gating_path)
    if adata_subset.shape[0] > sample_size:
        sc.pp.subsample(adata_subset, n_obs = sample_size)

    gate_list = [find_gate_path_of_gate(adata, gate) for gate in gate_list]
    
    return prepare_plot_dataframe(adata_subset,
                                  gates = gate_list,
                                  x_channel = x_channel,
                                  y_channel = y_channel)

def prepare_plot_dataframe(adata: AnnData,
                           gates: Union[str, list[str]],
                           x_channel: str,
                           y_channel: str) -> pd.DataFrame:
    df = adata.to_df(layer = "compensated")[[x_channel, y_channel]]
    df[gates] = adata.obsm["gating"][:, find_gate_indices(adata, gates)].toarray()
    return df

def gate_parent_in_adata(adata: AnnData,
                         parent_gating_path: str) -> AnnData:
    if parent_gating_path == "root":
        return adata
    return subset_gate(adata,
                       gate_path = parent_gating_path,
                       as_view = True)

def extract_channels_for_gate(gate_lut: dict,
                              gate: str) -> tuple[str, str]:
    return (gate_lut[gate]["dimensions"][0],
            gate_lut[gate]["dimensions"][1])

def group_plot(adata: AnnData,
               idx_map: pd.DataFrame,
               gate_group_map: dict,
               gate_lut: dict[str: dict[str: list[str]]],
               group: str,
               sample_size: int,
               fig: Figure,
               ax: Axes
               ) -> Axes:
    
    group_index = group.split("-")[1]
    gate_list = gate_group_map[group_index]
    reference_gate = gate_list[0] # just to have one single gate to do lookups
    x_channel, y_channel = extract_channels_for_gate(gate_lut, reference_gate)
    
    parent_gating_path = gate_lut[reference_gate]["parent_path"]

    plot_data = prepare_plot_data(adata,
                                  parent_gating_path,
                                  gate_list,
                                  x_channel,
                                  y_channel,
                                  sample_size)
    
    quadrant_map = {gate: get_quadrant(gate_lut, gate) for gate in gate_list}

    plot_params = {
        "x": x_channel,
        "y": y_channel,
        "s": 1,
        "linewidth": 0,
        "ax": ax,
        "rasterized": True
    }

    for i, gate in enumerate(gate_list):
        gate_specific_data = plot_data[plot_data[gate] == True]
        ax = sns.scatterplot(data = gate_specific_data,
                             color = sns.color_palette("Set1")[i],
                             **plot_params)
    

    return ax



def single_plot(adata: AnnData,
                idx_map: pd.DataFrame,
                gate_group_map: dict,
                gate_lut: dict[str: dict[str: list[str]]],
                gate: str,
                sample_size: int,
                fig: Figure,
                ax: Axes
                ) -> Axes:
    parent_gating_path = gate_lut[gate]["parent_path"]
    x_channel, y_channel = extract_channels_for_gate(gate_lut, gate)
    
    parent_gating_path = gate_lut[gate]["parent_path"]

    plot_data = prepare_plot_data(adata,
                                  parent_gating_path,
                                  gate,
                                  x_channel,
                                  y_channel,
                                  sample_size)

    plot_params = {
        "data": plot_data,
        "x": x_channel,
        "y": y_channel,
        "s": 1,
        "linewidth": 0,
        "ax": ax,
        "rasterized": True
    }
    ax = sns.scatterplot(c = plot_data[find_gate_path_of_gate(adata, gate)].map({True: "red", False: "gray"}),
                         **plot_params)

    return ax

def gating_strategy(adata: AnnData,
                    wsp_group: str,
                    sample_ID: Optional[str] = None,
                    file_name: Optional[str] = None,
                    sample_size: Optional[int] = 5_000,
                    return_fig: bool = False,
                    show: bool = True):
    if sample_ID and not file_name:
        file_name = map_sample_ID_to_filename(adata, sample_ID)
    adata = adata[adata.obs["file_name"] == file_name,:]
    gate_lut = extract_gate_lut(adata, wsp_group, file_name)
    gating_strategy_grid = GatingStrategyGrid(gate_lut)
    
    gate_map = gating_strategy_grid.idx_map
    gate_group_map = gating_strategy_grid.gate_group_map
    gate_lut = gating_strategy_grid.gate_lut
    
    gates_to_plot = gate_map.to_numpy().flatten()
    
    ncols = gate_map.shape[1]
    nrows = gate_map.shape[0]
    figsize = (2 * ncols,
               2 * nrows)
    fig, ax = plt.subplots(ncols = ncols,
                           nrows = nrows,
                           figsize = figsize)
    ax = ax.flatten()
    for i, gate in enumerate(gates_to_plot):
        if gate is np.nan:
            ax[i] = turn_off_missing_plot(ax[i])
        if "group-" in gate:
             ax[i] = group_plot(adata = adata,
                                idx_map = gate_map,
                                gate_group_map = gate_group_map,
                                gate_lut = gate_lut,
                                group = gate,
                                sample_size = sample_size,
                                fig = fig,
                                ax = ax[i]
                                )
        else:
            ax[i] = single_plot(adata = adata,
                                idx_map = gate_map,
                                gate_group_map = gate_group_map,
                                gate_lut = gate_lut,
                                gate = gate,
                                sample_size = sample_size,
                                fig = fig,
                                ax = ax[i]
                                )

    ax = np.reshape(ax, (ncols, nrows))

    if return_fig:
        return fig
    
    if not show:
        return ax
    
    plt.tight_layout()
    plt.show()
