from anndata import AnnData
import scanpy as sc
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

from typing import Optional, Literal


from ._utils import savefig_or_show

from .._utils import _default_gate_and_default_layer

## BIG TODO!: check for dataset hash before assuming that PCA has been calculated!
def _create_dimred_dataframe(adata: AnnData,
                             basis: str,
                             dimred: str,
                             layer: str) -> pd.DataFrame:
    obs_frame = adata.obs
    dimred_coordinates = adata.obsm[basis]
    dimred_cols = [f"{dimred}{i}" for i in range(1, dimred_coordinates.shape[1] + 1)]
    obs_frame[dimred_cols] = dimred_coordinates

    fluo_values = adata.to_df(layer = layer)
    
    return pd.concat([obs_frame, fluo_values], axis = 1)


def _create_dimred_plot(adata: AnnData,
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

@_default_gate_and_default_layer
def diffmap(adata: AnnData,
            gate: str = None,
            layer: str = None,
            figsize: tuple[float, float] = (3,3),
            return_fig: bool = False,
            return_dataframe: bool = False,
            save: Optional[bool] = None,
            show: Optional[bool] = None,
            *args,
            **kwargs):
    
    dimred = "DMAP"
    basis = f"X_diffmap_{gate}_{layer}"
    neighbors_key = f"{gate}_{layer}_neighbors"

    if "neighbors_key" not in kwargs:
        kwargs["neighbors_key"] = neighbors_key

    if return_dataframe:
        return _create_dimred_dataframe(adata = adata,
                                        basis = basis,
                                        dimred = dimred,
                                        layer = layer)

    fig = _create_dimred_plot(adata = adata,
                              basis = basis,
                              dimred = dimred,
                              figsize = figsize,
                              layer = layer,
                              *args,
                              **kwargs)
    
    if return_fig:
        return fig

    savefig_or_show(show = show, save = save)   

@_default_gate_and_default_layer
def pca(adata: AnnData,
        gate: str = None,
        layer: str = None,
        figsize: tuple[float, float] = (3,3),
        return_fig: bool = False,
        return_dataframe: bool = False,
        save: Optional[bool] = None,
        show: Optional[bool] = None,
        *args,
        **kwargs):
    
    dimred = "PCA"
    basis = f"X_pca_{gate}_{layer}"
    neighbors_key = f"{gate}_{layer}_neighbors"

    if "neighbors_key" not in kwargs:
        kwargs["neighbors_key"] = neighbors_key

    if return_dataframe:
        return _create_dimred_dataframe(adata = adata,
                                        basis = basis,
                                        dimred = dimred,
                                        layer = layer)

    fig = _create_dimred_plot(adata = adata,
                              basis = basis,
                              dimred = dimred,
                              figsize = figsize,
                              layer = layer,
                              *args,
                              **kwargs)
    
    if return_fig:
        return fig

    savefig_or_show(show = show, save = save)   

@_default_gate_and_default_layer
def tsne(adata: AnnData,
         gate: str = None,
         layer: str = None,
         figsize: tuple[float, float] = (3,3),
         return_fig: bool = False,
         return_dataframe: bool = False,
         save: Optional[bool] = None,
         show: Optional[bool] = None,
         *args,
         **kwargs):
    
    dimred = "TSNE"
    basis = f"X_tsne_{gate}_{layer}"
    neighbors_key = f"{gate}_{layer}_neighbors"

    if "neighbors_key" not in kwargs:
        kwargs["neighbors_key"] = neighbors_key

    if return_dataframe:
        return _create_dimred_dataframe(adata = adata,
                                        basis = basis,
                                        dimred = dimred,
                                        layer = layer)

    fig = _create_dimred_plot(adata = adata,
                              basis = basis,
                              dimred = dimred,
                              figsize = figsize,
                              layer = layer,
                              *args,
                              **kwargs)
    
    if return_fig:
        return fig

    savefig_or_show(show = show, save = save)   

@_default_gate_and_default_layer
def umap(adata: AnnData,
         gate: str = None,
         layer: str = None,
         figsize: tuple[float, float] = (3,3),
         return_fig: bool = False,
         return_dataframe: bool = False,
         save: Optional[bool] = None,
         show: Optional[bool] = None,
         *args,
         **kwargs):
    
    dimred = "UMAP"
    basis = f"X_umap_{gate}_{layer}"
    neighbors_key = f"{gate}_{layer}_neighbors"

    if "neighbors_key" not in kwargs:
        kwargs["neighbors_key"] = neighbors_key

    if return_dataframe:
        return _create_dimred_dataframe(adata = adata,
                                        basis = basis,
                                        dimred = dimred,
                                        layer = layer)

    fig = _create_dimred_plot(adata = adata,
                              basis = basis,
                              dimred = dimred,
                              figsize = figsize,
                              layer = layer,
                              *args,
                              **kwargs)
    
    if return_fig:
        return fig

    savefig_or_show(show = show, save = save)