from matplotlib.axes import Axes
from anndata import AnnData

from typing import Optional

from ._utils import savefig_or_show
from ._categorical_stripplot import _categorical_strip_box_plot

from .._settings import settings


def metadata(adata: AnnData,
             marker: str,
             groupby: str,
             splitby: str = None,
             cmap: str = None,
             stat_test: str = "Kruskal",
             order: list[str] = None,
             figsize: tuple[float, float] = (3,3),
             return_dataframe: bool = False,
             return_fig: bool = False,
             ax: Axes = None,
             save: Optional[str] = None,
             show: bool = None
             ):
    
    data = adata.uns["metadata"].dataframe.copy()
    try:
        data[marker] = data[marker].astype("float64")
    except ValueError as e:
        print(str(e))
        if "cast" in str(e) or "convert" in str(e):
            raise ValueError("Please provide a numeric variable for the marker")
        else:
            raise Exception from e

    if return_dataframe:
        return data
    
    plot_params = {
        "data": data,
        "x": groupby,
        "y": marker,
        "hue": splitby,
        "palette": cmap or settings.default_categorical_cmap if splitby else None,
        "order": order
    }

    fig, ax = _categorical_strip_box_plot(ax = ax,
                                          data = data,
                                          plot_params = plot_params,
                                          groupby = groupby,
                                          splitby = splitby,
                                          stat_test = stat_test,
                                          figsize = figsize)

    ax.set_title(f"{marker}\ngrouped by {groupby}")
    ax.set_xlabel("")
    ax.set_ylabel(marker)

    if return_fig:
        return fig    

    savefig_or_show(show = show, save = save)

    if show is False:
        return ax