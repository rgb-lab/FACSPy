from matplotlib import pyplot as plt

from anndata import AnnData

from ._utils import savefig_or_show

from ._baseplot import (stripboxplot,
                        barplot,
                        label_plot_basic,
                        adjust_legend)
from ._basestats import add_statistic

from typing import Optional

def metadata(adata: AnnData,
             marker: str,
             groupby: str,
             colorby: str,
             figsize: tuple[float, float] = (3,3),
             return_dataframe: bool = False,
             return_fig: bool = False,
             save: Optional[str] = None,
             show: bool = None
             ):
    if not isinstance(groupby, list):
        groupby = [groupby]
    
    if not isinstance(colorby, list):
        colorby = [colorby]
    
    
    df = adata.uns["metadata"].dataframe.copy()

    df[marker] = df[marker].astype("float64")
    if return_dataframe:
        return df
    
    ncols = 1
    nrows = len(groupby)
    figsize = figsize
    plot_params = {
        "data": df,
        "x": groupby[0],
        "y": marker,
        "hue": colorby[0],

    }

    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = figsize)
    if groupby == ["sample_ID"]:
        ax = barplot(ax,
                     plot_params = plot_params)

    else:
        ax = stripboxplot(ax,
                          plot_params = plot_params)
        try:
            ax = add_statistic(ax = ax,
                                test = "Kruskal",
                                dataframe = df,
                                groupby = groupby[0],
                                plot_params = plot_params)
        except ValueError as e:
            if str(e) != "All numbers are identical in kruskal":
                raise ValueError from e
            else:
                print("warning... Values were uniform, no statistics to plot.")

    ax = label_plot_basic(ax = ax,
                          title = f"{marker}\ngrouped by {groupby[0]}",
                          y_label = f"{marker}",
                          x_label = "")
    
    ax = adjust_legend(ax)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = "center")
        
    if return_fig:
        return fig

    plt.tight_layout()
    savefig_or_show(save = save, show = show)
    