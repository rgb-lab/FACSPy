from matplotlib.axes import Axes
from matplotlib.figure import Figure

from matplotlib import pyplot as plt

from anndata import AnnData

from .utils import create_boxplot, savefig_or_show

from typing import Optional

def metadata(adata: AnnData,
             marker: str,
             groupby: str,
             return_dataframe: bool = False,
             return_fig: bool = False,
             save: Optional[str] = None,
             show: bool = False
             ):
    if not isinstance(groupby, list):
        groupby = [groupby]
    
    
    df = adata.uns["metadata"].dataframe.copy()

    df[marker] = df[marker].astype("float64")
    if return_dataframe:
        return df
    
    ncols = 1,
    nrows = len(groupby)
    figsize = (len(df) / 10, 3 * len(groupby)) 
    
    fig, ax = plt.subplots(ncols = 1, nrows = nrows, figsize = figsize)
    for i, grouping in enumerate(groupby):
        plot_params = {
            "data": df,
            "x": "sample_ID" if grouping is None else grouping,
            "y": marker,

        }
        if len(groupby) > 1:
            ax[i] = create_boxplot(ax[i],
                                   grouping,
                                   plot_params)
        else:
            ax = create_boxplot(ax,
                                grouping,
                                plot_params)
        
    if return_fig:
        return fig
    
    plt.tight_layout()
    savefig_or_show(save = save, show = show)
    