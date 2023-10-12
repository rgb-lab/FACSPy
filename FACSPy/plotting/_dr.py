from anndata import AnnData
import scanpy as sc
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

from typing import Optional, Literal


from .utils import savefig_or_show

def create_dimred_dataframe(adata: AnnData,
                            basis: str,
                            dimred: str,
                            data_origin: Literal["transformed", "compensated"]) -> pd.DataFrame:
    obs_frame = adata.obs
    dimred_coordinates = adata.obsm[basis]
    dimred_cols = [f"{dimred}{i}" for i in range(1, dimred_coordinates.shape[1] + 1)]
    obs_frame[dimred_cols] = dimred_coordinates

    fluo_values = adata.to_df(layer = data_origin)
    
    return pd.concat([obs_frame, fluo_values], axis = 1)


def create_dimred_plot(adata: AnnData,
                       basis: str,
                       dimred: str,
                       figsize: tuple[float, float],
                       layer: str,
                       *args,
                       **kwargs) -> Figure:
    
    color = kwargs.get("color", None)
    
    fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = figsize)
    sc.pl.embedding(adata = adata,
                    basis = basis,
                    layer = layer,
                    ax = ax,
                    show = False,
                    *args,
                    **kwargs)
    ax.set_xlabel(f"{dimred}1")
    ax.set_ylabel(f"{dimred}2")
    ax.set_title(color)

    return fig

def diffmap(adata: AnnData,
            gate: str,
            data_origin: Literal["transformed", "compensated"] = "transformed",
            figsize: tuple[float, float] = (3,3),
            return_fig: bool = False,
            return_dataframe: bool = False,
            save: Optional[bool] = None,
            show: Optional[bool] = None,
            *args,
            **kwargs):
    
    dimred = "DMAP"
    basis = f"X_{gate}_{data_origin}_diffmap"
    neighbors_key = f"{gate}_{data_origin}_neighbors"

    if "neighbors_key" not in kwargs:
        kwargs["neighbors_key"] = neighbors_key

    if return_dataframe:
        return create_dimred_dataframe(adata = adata,
                                       basis = basis,
                                       dimred = dimred,
                                       data_origin = data_origin)

    fig = create_dimred_plot(adata = adata,
                             basis = basis,
                             dimred = dimred,
                             figsize = figsize,
                             layer = data_origin,
                             *args,
                             **kwargs)
    
    if return_fig:
        return fig

    savefig_or_show(show = show, save = save)   

def pca(adata: AnnData,
        gate: str,
        data_origin: Literal["transformed", "compensated"] = "transformed",
        figsize: tuple[float, float] = (3,3),
        return_fig: bool = False,
        return_dataframe: bool = False,
        save: Optional[bool] = None,
        show: Optional[bool] = None,
        *args,
        **kwargs):
    
    dimred = "PCA"
    basis = f"X_{gate}_{data_origin}_pca"
    neighbors_key = f"{gate}_{data_origin}_neighbors"

    if "neighbors_key" not in kwargs:
        kwargs["neighbors_key"] = neighbors_key

    if return_dataframe:
        return create_dimred_dataframe(adata = adata,
                                       basis = basis,
                                       dimred = dimred,
                                       data_origin = data_origin)

    fig = create_dimred_plot(adata = adata,
                             basis = basis,
                             dimred = dimred,
                             figsize = figsize,
                             layer = data_origin,
                             *args,
                             **kwargs)
    
    if return_fig:
        return fig

    savefig_or_show(show = show, save = save)   

  
    
def tsne(adata: AnnData,
         gate: str,
         data_origin: Literal["transformed", "compensated"] = "transformed",
         figsize: tuple[float, float] = (3,3),
         return_fig: bool = False,
         return_dataframe: bool = False,
         save: Optional[bool] = None,
         show: Optional[bool] = None,
         *args,
         **kwargs):
    
    dimred = "TSNE"
    basis = f"X_{gate}_{data_origin}_tsne"
    neighbors_key = f"{gate}_{data_origin}_neighbors"

    if "neighbors_key" not in kwargs:
        kwargs["neighbors_key"] = neighbors_key

    if return_dataframe:
        return create_dimred_dataframe(adata = adata,
                                       basis = basis,
                                       dimred = dimred,
                                       data_origin = data_origin)

    fig = create_dimred_plot(adata = adata,
                             basis = basis,
                             dimred = dimred,
                             figsize = figsize,
                             layer = data_origin,
                             *args,
                             **kwargs)
    
    if return_fig:
        return fig

    savefig_or_show(show = show, save = save)   

def umap(adata: AnnData,
         gate: str,
         data_origin: Literal["transformed", "compensated"] = "transformed",
         figsize: tuple[float, float] = (3,3),
         return_fig: bool = False,
         return_dataframe: bool = False,
         save: Optional[bool] = None,
         show: Optional[bool] = None,
         *args,
         **kwargs):
    
    dimred = "UMAP"
    basis = f"X_{gate}_{data_origin}_umap"
    neighbors_key = f"{gate}_{data_origin}_neighbors"

    if "neighbors_key" not in kwargs:
        kwargs["neighbors_key"] = neighbors_key

    if return_dataframe:
        return create_dimred_dataframe(adata = adata,
                                       basis = basis,
                                       dimred = dimred,
                                       data_origin = data_origin)

    fig = create_dimred_plot(adata = adata,
                             basis = basis,
                             dimred = dimred,
                             figsize = figsize,
                             layer = data_origin,
                             *args,
                             **kwargs)
    
    if return_fig:
        return fig

    savefig_or_show(show = show, save = save)