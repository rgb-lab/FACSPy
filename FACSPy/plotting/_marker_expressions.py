import pandas as pd
import numpy as np
from anndata import AnnData
from matplotlib import pyplot as plt
import seaborn as sns

from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from typing import Literal, Union, Optional

from ..utils import flatten_nested_list

from .cofactor_plots import calculate_histogram_data

import warnings


def convert_to_mapping(dataframe: pd.DataFrame,
                       key_col: str,
                       value_col: str) -> dict:
    return {key_value: dataframe.loc[dataframe[key_col] == key_value, value_col].iloc[0] for key_value in dataframe[key_col].unique()}

def append_metadata_obs(adata: AnnData,
                        expression_data: pd.DataFrame) -> pd.DataFrame:
    expression_data[adata.obs.columns] = adata.obs
    return expression_data

def convert_expression_to_histogram_data(expression_data: pd.DataFrame,
                                         marker: str,
                                         groupby: str) -> pd.DataFrame:
    group_values = list(expression_data[groupby[0]].unique())
    histogram_df = pd.DataFrame(
        data = {groupby[0]: flatten_nested_list([[group for _ in range (100)] for group in group_values])},
        columns = [groupby[0], "x", "y"],
        index = range(100 * len(group_values))
    )
    
    for group in  group_values:
        group_spec_expression_data = expression_data.loc[expression_data[groupby[0]] == group, [groupby[0], marker]]
        x, y = calculate_histogram_data(group_spec_expression_data,
                                        {"x": marker})
        histogram_df.loc[histogram_df[groupby[0]] == group, ["x", "y"]] = np.vstack([x, y]).T

    return histogram_df

def append_colorby_variable(adata: AnnData,
                            dataframe: pd.DataFrame,
                            colorby: str,
                            groupby: str) -> pd.DataFrame:
    mapping = convert_to_mapping(adata.uns["metadata"].to_df(),
                                 key_col = groupby[0],
                                 value_col = colorby[0])
    dataframe[colorby[0]] = dataframe[groupby[0]].map(mapping)
    return dataframe

#TODO: check if mapping is possible: either the colorby is not in metadata or the grouping by sampleID leads to multiple outputs, not mappable.
def marker_density(adata: AnnData,
                   markers: Union[str, list[str]],
                   colorby: Union[str, list[str]],
                   groupby: Union[str, list[str]] = "sample_ID",
                   ridge: bool = False,
                   cmap: str = "Set1",
                   highlight: Optional[Union[str, list[str]]] = None,
                   linewidth: float = 0.5,
                   on: Literal["compensated", "transformed", "raw"] = "transformed") -> Optional[Figure]:
    """plots histograms per sample and colors by colorby variable"""
    if not isinstance(markers, list):
        markers = [markers]

    if not isinstance(colorby, list):
        colorby = [colorby]

    if not isinstance(groupby, list):
        groupby = [groupby]

    if not isinstance(highlight, list) and highlight is not None:
        highlight = [highlight]

    expression_data = adata.to_df(layer = on)
    expression_data = append_metadata_obs(adata, expression_data)
    ## pot. buggy: groupby only singular at the moment...
    for marker in markers:
        histogram_df = convert_expression_to_histogram_data(expression_data = expression_data,
                                                            marker = marker,
                                                            groupby = groupby)
        
        # histogram_df = append_colorby_variable(adata = adata,
        #                                        dataframe = histogram_df,
        #                                        colorby = colorby,
        #                                        groupby = groupby)
        
        histogram_df[groupby[0]] = histogram_df[groupby[0]].astype("str")
        
        user_defined_cmap = sns.color_palette(cmap, len(list(histogram_df[colorby[0]].unique())))
        
        if highlight is not None:
            colorby_pal = {
                group: "red" if group in highlight else "grey"
                for group in list(histogram_df[colorby[0]].unique())
            }
        else:
            colorby_pal = {group: user_defined_cmap[i]
                for i, group in enumerate(list(histogram_df[colorby[0]].unique()))
            }
        

    
        if ridge:
            if groupby[0] != colorby[0]:
                pal = map_pal_to_groupby(colorby_pal, histogram_df, groupby[0], colorby[0])
            else:
                pal = colorby_pal
            warnings.simplefilter('ignore', category=UserWarning)
            sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
            g = sns.FacetGrid(histogram_df,
                              row = groupby[0],
                              hue = groupby[0],
                              aspect = 5,
                              height = 0.3,
                              palette = pal)
            
            g.set_titles("")
            g.set_ylabels("")
            g.set(yticks = [])
            g.map(sns.lineplot, "x", "y", clip_on = False)
            
            g.map(plt.axvline, x = 0.88, color = "black")#
            
            g.figure.subplots_adjust(hspace = -0.5)

            # Define and use a simple function to label the plot in axes coordinates
            def label(x, color, label):
                ax = plt.gca()
                ax.text(0.9, .2, label, fontsize = 10, color=color, transform=ax.transAxes) #ha="left", va="center", 
            # if colorby[0] != groupby[0]:
            #     g.map(label, groupby[0])
            for axs in g._axes_dict.keys():
                g._axes_dict[axs].fill_between(x = histogram_df.loc[histogram_df[groupby[0]] == axs, "x"].to_numpy(dtype = np.float64),
                                               y1 = histogram_df.loc[histogram_df[groupby[0]] == axs, "y"].to_numpy(dtype = np.float64),
                                               y2 = 0,
                                               alpha = 0.1)
                if groupby[0] != colorby[0]:
                    g._axes_dict[axs].text(0.9, .2, axs, fontsize = 10, color="black", transform = g._axes_dict[axs].transAxes)
            
            
            handles = [Patch(facecolor = colorby_pal[name]) for name in colorby_pal]
            labels = list(colorby_pal.keys())
            sns.reset_orig()
            warnings.resetwarnings()
            group_legend = plt.legend(handles,
                                      labels,
                                      loc = "center right",
                                      title = colorby[0],
                                      bbox_to_anchor = (3, 0.5) if groupby[0] != colorby[0] else (2,0.5),
                                      bbox_transform = g.fig.transFigure)
            
            g.fig.add_artist(group_legend)
            g.set_xlabels(f'{marker}\n{on}\nexpression', fontsize = 10)
            
            # TODO: REMOVE uncomment if in bad mood :)
            #print("Nico war hier möhöhöhö")
            
        else:
            fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (2,2))
            sns.lineplot(data = histogram_df,
                        x = "x",
                        y = "y",
                        hue = colorby[0],
                        style = groupby[0],
                        dashes = False,
                        legend = "auto",
                        palette = colorby_pal,
                        linewidth = linewidth,
                        ax = ax)

            handles, labels = ax.get_legend_handles_labels()
            colorby_labels = np.array([[handle, label] for handle, label in zip(handles, labels) if
                                      any(k in label for k in colorby + list(histogram_df[colorby[0]].unique()))])
            ax.legend(colorby_labels[:,0],
                    colorby_labels[:,1],
                    bbox_to_anchor = (1.05,1))
            ax.set_title(f"Marker expression {marker}\nper sample ID")
            ax.set_ylabel("Density (norm)")
            ax.set_xlabel(f"{on} expression")

            #plt.tight_layout()
            
        plt.show()
        
def map_pal_to_groupby(pal: dict,
                       data: pd.DataFrame,
                       groupby: str,
                       colorby: str) -> dict:
    """maps the original palette to new groupby variable by looking"""
    return {group: pal[data.loc[data[groupby] == group, colorby].iloc[0]] for group in data[groupby].unique()}

def df_generation(adata: AnnData,
                   markers: Union[str, list[str]],
                   colorby: Union[str, list[str]],
                   groupby: Union[str, list[str]] = "sample_ID",
                   ridge: bool = False,
                   highlight: Optional[Union[str, list[str]]] = None,
                   linewidth: float = 0.5,
                   on: Literal["compensated", "transformed", "raw"] = "transformed") -> Optional[Figure]:
    """plots histograms per sample and colors by colorby variable"""
    if not isinstance(markers, list):
        markers = [markers]

    if not isinstance(colorby, list):
        colorby = [colorby]

    if not isinstance(groupby, list):
        groupby = [groupby]

    if not isinstance(highlight, list) and highlight is not None:
        highlight = [highlight]

    expression_data = adata.to_df(layer = on)
    expression_data = append_metadata_obs(adata, expression_data)

    ## pot. buggy: groupby only singular at the moment...
    for marker in markers:
        histogram_df = convert_expression_to_histogram_data(expression_data = expression_data,
                                                            marker = marker,
                                                            groupby = groupby)
        
        histogram_df = append_colorby_variable(adata = adata,
                                               dataframe = histogram_df,
                                               colorby = colorby,
                                               groupby = groupby)
        
        if ridge:
            fig = create_ridge_plot()
        
        else:
            a = 1

        
    return histogram_df

def create_ridge_plot(data: pd.DataFrame,
                      groupby: str,
                      colorby: str,
                      ): pass