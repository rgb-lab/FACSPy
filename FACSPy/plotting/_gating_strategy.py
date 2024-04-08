import numpy as np
import pandas as pd

from anndata import AnnData
import scanpy as sc

import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.patches as patches

from typing import Union, Optional, Literal


from .._utils import (_create_gate_lut,
                      GATE_SEPARATOR,
                      _find_parent_population,
                      subset_gate,
                      _find_gate_indices,
                      _find_gate_path_of_gate)

from ._utils import turn_off_missing_plot, savefig_or_show


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
            hierarchy_map[n - 2] += [_find_parent_population(self.full_gate_path(gate)) for gate in gates_at_nth_depth]

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
                gate_group_map[str(key)] = value

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
        if idx_map.shape[0] == 1:
            
            # meaning that we have a one-dimensional gating strategy that
            # can be plotted in one single row
            return idx_map
        
        idx_map = self._replace_group_gates_with_groups(idx_map = idx_map,
                                                        gate_group_map = self.gate_group_map)
        
        
        idx_map = self._merge_groups_to_single_index(idx_map)
        
        #idx_map = self._concatenate_single_indices(idx_map)
        
        idx_map = self._center_indices_in_map(idx_map)

        idx_map = idx_map.reset_index(drop = True).T.reset_index(drop = True)

        idx_map = idx_map.fillna("NaN")

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

def get_rectangle_quadrant(gate_lut: dict,
                 gate: str) -> int:
    # quadrant naming similar to flowjo:
    #[1,2]
    #[4,3]
    
    vertices = gate_lut[gate]["gate_dimensions"]
    if vertices[0,0] is None: # x dimension is from None (-inf)
        if vertices[0,1] is None:
            raise ValueError()
        return 4 if vertices[1,0] is None else 1
    assert vertices[0,1] is None
    return 3 if vertices[1,0] is None else 2

def extract_gate_lut(adata: AnnData,
                     wsp_group: str,
                     file_name: str) -> dict[str: dict[str: Union[list[str], str]]]:
    return _create_gate_lut(adata.uns["workspace"][wsp_group])[file_name]

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
    if sample_size is not None and adata_subset.shape[0] > sample_size:
        sc.pp.subsample(adata_subset, n_obs = sample_size)
    gate_list = [_find_gate_path_of_gate(adata, gate) for gate in gate_list]
    
    return prepare_plot_dataframe(adata_subset,
                                  gates = gate_list,
                                  x_channel = x_channel,
                                  y_channel = y_channel)

def prepare_plot_dataframe(adata: AnnData,
                           gates: Union[list[str], str],
                           x_channel: str,
                           y_channel: str) -> pd.DataFrame:

    if x_channel == y_channel:
        df = adata.to_df(layer = "compensated")[[x_channel]]
    else:
        df = adata.to_df(layer = "compensated")[[x_channel, y_channel]]
    df[gates] = adata.obsm["gating"][:, _find_gate_indices(adata, gates)].toarray()
    return df

def gate_parent_in_adata(adata: AnnData,
                         parent_gating_path: str) -> AnnData:
    if parent_gating_path == "root":
        return adata
    return subset_gate(adata,
                       gate_path = parent_gating_path,
                       as_view = True)

def extract_channels_for_gate(adata: AnnData,
                              gate_lut: dict,
                              gate: str) -> tuple[str, str]:
    x_channel = gate_lut[gate]["dimensions"][0]
    x_channel_idx = adata.var.loc[adata.var["pnn"] == x_channel, "pns"].iloc[0]
    try:
        y_channel = gate_lut[gate]["dimensions"][1]
    except IndexError:
        y_channel = "SSC-A"
        gate_lut[gate]["vertices"] = np.hstack(
            [np.array(gate_lut[gate]["vertices"]).reshape(2,1),
             np.array([[250_000, 250_000]]).reshape(2,1)]
                                                 ).reshape(2,2).T
        gate_lut[gate]["dimensions"] += ["SSC-A"]
    y_channel_idx = adata.var.loc[adata.var["pnn"] == y_channel, "pns"].iloc[0]
    return (x_channel_idx, y_channel_idx)

def group_plot(adata: AnnData,
               idx_map: pd.DataFrame,
               gate_group_map: dict,
               gate_lut: dict[str: dict[str: list[str]]],
               group: str,
               sample_size: int,
               fig: Figure,
               ax: Axes,
               user_plot_params: dict,
               ) -> Axes:
    
    group_index = group.split("-")[1]
    gate_list = gate_group_map[group_index]
    reference_gate = gate_list[0] # just to have one single gate to do lookups
    x_channel, y_channel = extract_channels_for_gate(adata, gate_lut, reference_gate)
    parent_gating_path = gate_lut[reference_gate]["parent_path"]

    plot_data = prepare_plot_data(adata,
                                  parent_gating_path,
                                  gate_list,
                                  x_channel,
                                  y_channel,
                                  sample_size)

    plot_params = {
        "x": x_channel,
        "y": y_channel,
        "s": 5,
        "linewidth": 0,
        "ax": ax,
        "rasterized": True
    }
    
    if user_plot_params:
        plot_params = merge_plotting_parameters(plot_params, user_plot_params)
    
    for i, gate in enumerate(gate_list):
        gate_specific_data = plot_data[plot_data[_find_gate_path_of_gate(adata, gate)] == True]
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
                ax: Axes,
                user_plot_params: dict,
                ) -> Axes:
    parent_gating_path = gate_lut[gate]["parent_path"]
    x_channel, y_channel = extract_channels_for_gate(adata, gate_lut, gate)
    
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
        "s": 5,
        "linewidth": 0,
        "ax": ax,
        "rasterized": True
    }

    if user_plot_params:
        plot_params = merge_plotting_parameters(plot_params, user_plot_params)

    ax = sns.scatterplot(c = plot_data[_find_gate_path_of_gate(adata, gate)].map({True: "red",
                                                                                 False: "gray"}),
                           **plot_params)

    return ax

def merge_plotting_parameters(facspy_plot_params: dict,
                              user_plot_params: dict) -> dict:
    """ merges plotting parameters. user defined parameters overwrite FACSPy ones """
    for key in user_plot_params:
        facspy_plot_params[key] = user_plot_params[key]
    return facspy_plot_params

def add_gates_to_plot(adata: AnnData,
                      ax: Axes,
                      gate_lut: dict,
                      gates: Union[list[str], str]) -> Axes:
    
    gate_line_params = {
        "marker": ".",
        "markersize": 2,
        "color": "black",
        "linestyle": "-"
    }
    hvline_params = {
        "color": "black",
        "linestyle": "-"
    }
    if not isinstance(gates, list):
        gates = [gates]
    for gate in gates:
        gate_dict = gate_lut[gate]
        vertices = gate_dict["vertices"]
        if gate_dict["gate_type"] == "PolygonGate":
            ax.plot(vertices[:,0],
                    vertices[:,1],
                    **gate_line_params)
        elif gate_dict["gate_type"] == "GMLRectangleGate":
            ### handles the case of quandrant gates
            if np.isnan(vertices).any():
                if any(np.isnan(vertices[:,0])):
                    if all(np.isnan(vertices[:,0])):
                        ax.axvline(x = np.nan,
                                   **hvline_params)
                    else:
                        #TODO incredibly messy...
                        ax.axvline(x = int(vertices[0][~np.isnan(vertices[0])][0]),
                                   **hvline_params)
                if any(np.isnan(vertices[:,1])):
                    if all(np.isnan(vertices[:,1])):
                        ax.axhline(y = np.nan,
                                **hvline_params)
                    else:
                        #TODO incredibly messy...
                        ax.axhline(y = int(vertices[1][~np.isnan(vertices[1])][0]),
                            **hvline_params)              
                continue
            
            patch_starting_point = (vertices[0,0], vertices[1,0])
            height = abs(int(np.diff(vertices[1])))
            width = abs(int(np.diff(vertices[0])))
            ax.add_patch(
                patches.Rectangle(
                    xy = patch_starting_point,
                    width = width,
                    height = height,
                    facecolor = "none",
                    edgecolor = "black",
                    linestyle = "-",
                    linewidth = 1
                )
            )
            ax = adjust_viewlim(ax, patch_starting_point, height, width)

    return ax

def adjust_viewlim(ax: Axes,
                   patch_starting_point: tuple[float, float],
                   height: int,
                   width: int) -> Axes:
    current_viewlim = ax.viewLim
    current_x_lims = current_viewlim._points[:,0]
    current_y_lims = current_viewlim._points[:,1]
    x0 = calculate_range_extension_viewLim(point = min(current_x_lims[0], patch_starting_point[0]),
                                           point_loc = "min")
    y0 = calculate_range_extension_viewLim(point = min(current_y_lims[0], patch_starting_point[1]),
                                           point_loc = "min")
    x1 = calculate_range_extension_viewLim(point = max(current_x_lims[1], patch_starting_point[0] + width),
                                           point_loc = "max")
    y1 = calculate_range_extension_viewLim(point = max(current_y_lims[1], patch_starting_point[1] + height),
                                           point_loc = "max")
    current_viewlim.set_points(np.array([[x0,y0],
                                         [x1, y1]]))
    return ax

def calculate_range_extension_viewLim(point: float,
                                      point_loc: Literal["min", "max"]) -> float:
    if point_loc == "min":
        return point * 1.1 if point < 0 else point * 0.9
    if point_loc == "max":
        return point * 0.9 if point < 0 else point * 1.1

def manage_axis_scale(adata: AnnData,
                      ax: Axes,
                      gate_lut: dict,
                      gates: Union[list[str], str],
                      axis_kwargs: dict) -> Axes:

    x_channel, y_channel = extract_channels_for_gate(adata, gate_lut, gates)
    if x_channel in axis_kwargs:
        if axis_kwargs[x_channel] == "log":
            ax.set_xscale("symlog", linthresh = 1)
    elif adata.var.loc[adata.var["pns"] == x_channel, "type"].iloc[0] == "fluo":
        try:
            cofactor = adata.var.loc[adata.var["pns"] == x_channel, "cofactors"].iloc[0]
        except IndexError:
            cofactor = 5
        ax.set_xscale("symlog", linthresh = float(cofactor))
    
    if y_channel in axis_kwargs:
        if axis_kwargs[y_channel] == "log":
            ax.set_yscale("symlog", linthresh = 1)
    elif adata.var.loc[adata.var["pns"] == y_channel, "type"].iloc[0] == "fluo":
        try:
            cofactor = adata.var.loc[adata.var["pns"] == y_channel, "cofactors"].iloc[0]
        except IndexError:
            print("Index Error...")
            cofactor = 5
        ax.set_yscale("symlog", linthresh = float(cofactor))
    
    return ax


def gating_strategy(adata: AnnData,
                    wsp_group: str,
                    sample_ID: Optional[str] = None,
                    file_name: Optional[str] = None,
                    sample_size: Optional[int] = 5_000,
                    draw_gates: bool = True,
                    axis_kwargs: dict = {},
                    plot_kwargs: dict = {},
                    return_dataframe: bool = False,
                    return_fig: bool = False,
                    save: bool = None,
                    show: bool = None):

    if sample_ID and not file_name:
        file_name = map_sample_ID_to_filename(adata, sample_ID)
    
    adata = adata[adata.obs["file_name"] == file_name,:]
    
    gate_lut = extract_gate_lut(adata, wsp_group, file_name)
    gating_strategy_grid = GatingStrategyGrid(gate_lut)
    
    gate_map = gating_strategy_grid.gating_grid
    gate_group_map = gating_strategy_grid.gate_group_map
    gate_lut = gating_strategy_grid.gate_lut
    
    gates_to_plot = gate_map.to_numpy().flatten()
    
    ncols = gate_map.shape[1]
    nrows = gate_map.shape[0]
    figsize = (3 * ncols,
               3 * nrows)
    
    fig, ax = plt.subplots(ncols = ncols,
                           nrows = nrows,
                           figsize = figsize)
    ax = ax.flatten()
    
    for i, gate in enumerate(gates_to_plot):
        ## TODO: think of a better way...
        adata = adata[adata.obs["file_name"] == file_name,:]
        if gate == "NaN":
            ax[i] = turn_off_missing_plot(ax[i])
            continue

        elif "group-" in gate:
            ax[i] = group_plot(adata = adata,
                               idx_map = gate_map,
                               gate_group_map = gate_group_map,
                               gate_lut = gate_lut,
                               group = gate,
                               sample_size = sample_size,
                               fig = fig,
                               ax = ax[i],
                               user_plot_params = plot_kwargs
                               )
            group_index = gate.split("-")[1]
            gate_list = gate_group_map[group_index]
            if draw_gates:
                for single_gate in gate_list:
                    ax[i] = add_gates_to_plot(adata = adata,
                                    ax = ax[i],
                                    gate_lut = gate_lut,
                                    gates = single_gate
                                    )
            ax[i] = manage_axis_scale(adata = adata,
                                      ax = ax[i],
                                      gate_lut = gate_lut,
                                      gates = gate_list[0],
                                      axis_kwargs = axis_kwargs)
        else:
            ax[i] = single_plot(adata = adata,
                                idx_map = gate_map,
                                gate_group_map = gate_group_map,
                                gate_lut = gate_lut,
                                gate = gate,
                                sample_size = sample_size,
                                fig = fig,
                                ax = ax[i],
                                user_plot_params = plot_kwargs
                                )
            if draw_gates:
                ax[i] = add_gates_to_plot(adata = adata,
                                          ax = ax[i],
                                          gate_lut = gate_lut,
                                          gates = gate)        
            ax[i] = manage_axis_scale(adata = adata,
                                      ax = ax[i],
                                      gate_lut = gate_lut,
                                      gates = gate,
                                      axis_kwargs = axis_kwargs)
        ax[i].set_title(gate)

    if return_fig:
        return fig
    
    
    plt.tight_layout()
    
    savefig_or_show(save = save, show = show)
