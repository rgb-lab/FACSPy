from anndata import AnnData
import scanpy as sc
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import pandas as pd

from typing import Optional


from ._utils import savefig_or_show

from .._utils import _default_gate_and_default_layer, _enable_gate_aliases

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

def _has_colorbar(axs: Axes):
    return axs._children[0].colorbar

def _create_dimred_plot(adata: AnnData,
                        basis: str,
                        dimred: str,
                        figsize: tuple[float, float],
                        layer: str,
                        ax: Axes,
                        *args,
                        **kwargs) -> Figure:
    
    color = kwargs.get("color", None)
    if ax is None:
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(111)
    axs: Axes = sc.pl.embedding(adata = adata,
                                basis = basis,
                                layer = layer,
                                ax = ax,
                                show = False,
                                *args,
                                **kwargs)
    if _has_colorbar(axs):
        axs._children[0].colorbar.ax.set_ylabel(f"{layer} expression",
                                                rotation = 270,
                                                labelpad = 20)
    axs.set_xlabel(f"{dimred}1")
    axs.set_ylabel(f"{dimred}2")
    axs.set_title(color)

    return axs

@_default_gate_and_default_layer
@_enable_gate_aliases
def diffmap(adata: AnnData,
            gate: str = None,
            layer: str = None,
            dimred: str = "diffmap",
            figsize: tuple[float, float] = (3,3),
            return_fig: bool = False,
            return_dataframe: bool = False,
            save: Optional[bool] = None,
            show: Optional[bool] = None,
            ax: Axes = None,
            *args,
            **kwargs):
    """\
    Plots the diffusion embedding.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels
    gate
        The gate to be analyzed, called by the population name.
        This parameter has a default stored in fp.settings, but
        can be superseded by the user.
    layer
        The layer corresponding to the data matrix. Similar to the
        gate parameter, it has a default stored in fp.settings which
        can be overwritten by user input.
    figsize
        contains the dimensions of the final figure as a tuple of two ints or floats
    title
        sets the figure title. Optional
    show
        whether to show the figure
    save
        expects a file path and a file name. saves the figure to the indicated path
    return_dataframe
        if set to True, returns the raw data that are used for plotting. vmin and vmax
        are not set.
    return_fig
        if set to True, the figure is returned.
    ax
        Optional parameter. Sets user defined ax from for example plt.subplots
    *args
        arguments ultimately passed to sc.pl.diffmap
    **kwargs
        arguments ultimately passed to sc.pl.diffmap

    Returns
    -------
    if `show==False` a :class:`~matplotlib.axes.Axes`
    
    """
    
    basis = f"X_{dimred}_{gate}_{layer}"
    neighbors_key = f"{gate}_{layer}_neighbors"
    dimred = "DMAP"

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
                              ax = ax,
                              *args,
                              **kwargs)
    
    if return_fig:
        return fig

    savefig_or_show(show = show, save = save)

    if show is False:
        return fig

@_default_gate_and_default_layer
@_enable_gate_aliases
def pca(adata: AnnData,
        gate: str = None,
        layer: str = None,
        dimred: str = "pca",
        figsize: tuple[float, float] = (3,3),
        return_fig: bool = False,
        return_dataframe: bool = False,
        save: Optional[bool] = None,
        show: Optional[bool] = None,
        ax: Axes = None,
        *args,
        **kwargs):
    """\
    Plots the PCA embedding.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels
    gate
        The gate to be analyzed, called by the population name.
        This parameter has a default stored in fp.settings, but
        can be superseded by the user.
    layer
        The layer corresponding to the data matrix. Similar to the
        gate parameter, it has a default stored in fp.settings which
        can be overwritten by user input.
    figsize
        contains the dimensions of the final figure as a tuple of two ints or floats
    title
        sets the figure title. Optional
    show
        whether to show the figure
    save
        expects a file path and a file name. saves the figure to the indicated path
    return_dataframe
        if set to True, returns the raw data that are used for plotting. vmin and vmax
        are not set.
    return_fig
        if set to True, the figure is returned.
    ax
        Optional parameter. Sets user defined ax from for example plt.subplots
    *args
        arguments ultimately passed to sc.pl.pca
    **kwargs
        arguments ultimately passed to sc.pl.pca

    Returns
    -------
    if `show==False` a :class:`~matplotlib.axes.Axes`
    
    """
 
    basis = f"X_{dimred}_{gate}_{layer}"
    neighbors_key = f"{gate}_{layer}_neighbors"
    dimred = "PCA"

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
                              ax = ax,
                              *args,
                              **kwargs)
    
    if return_fig:
        return fig

    savefig_or_show(show = show, save = save)   

    if show is False:
        return fig

@_default_gate_and_default_layer
@_enable_gate_aliases
def tsne(adata: AnnData,
         gate: str = None,
         layer: str = None,
         dimred: str = "tsne",
         figsize: tuple[float, float] = (3,3),
         return_fig: bool = False,
         return_dataframe: bool = False,
         save: Optional[bool] = None,
         show: Optional[bool] = None,
         ax: Axes = None,
         *args,
         **kwargs):
    """\
    Plots the TSNE embedding.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels
    gate
        The gate to be analyzed, called by the population name.
        This parameter has a default stored in fp.settings, but
        can be superseded by the user.
    layer
        The layer corresponding to the data matrix. Similar to the
        gate parameter, it has a default stored in fp.settings which
        can be overwritten by user input.
    figsize
        contains the dimensions of the final figure as a tuple of two ints or floats
    title
        sets the figure title. Optional
    show
        whether to show the figure
    save
        expects a file path and a file name. saves the figure to the indicated path
    return_dataframe
        if set to True, returns the raw data that are used for plotting. vmin and vmax
        are not set.
    return_fig
        if set to True, the figure is returned.
    ax
        Optional parameter. Sets user defined ax from for example plt.subplots
    *args
        arguments ultimately passed to sc.pl.tsne
    **kwargs
        arguments ultimately passed to sc.pl.tsne

    Returns
    -------
    if `show==False` a :class:`~matplotlib.axes.Axes`
    
    """
 
    basis = f"X_{dimred}_{gate}_{layer}"
    neighbors_key = f"{gate}_{layer}_neighbors"
    dimred = "TSNE"

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
                              ax = ax,
                              *args,
                              **kwargs)
    
    if return_fig:
        return fig

    savefig_or_show(show = show, save = save)   

    if show is False:
        return fig

@_default_gate_and_default_layer
@_enable_gate_aliases
def umap(adata: AnnData,
         gate: str = None,
         layer: str = None,
         dimred: str = "umap",
         figsize: tuple[float, float] = (3,3),
         return_fig: bool = False,
         return_dataframe: bool = False,
         save: Optional[bool] = None,
         show: Optional[bool] = None,
         ax: Axes = None,
         *args,
         **kwargs):
    """\
    Plots the UMAP embedding.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels
    gate
        The gate to be analyzed, called by the population name.
        This parameter has a default stored in fp.settings, but
        can be superseded by the user.
    layer
        The layer corresponding to the data matrix. Similar to the
        gate parameter, it has a default stored in fp.settings which
        can be overwritten by user input.
    figsize
        contains the dimensions of the final figure as a tuple of two ints or floats
    title
        sets the figure title. Optional
    show
        whether to show the figure
    save
        expects a file path and a file name. saves the figure to the indicated path
    return_dataframe
        if set to True, returns the raw data that are used for plotting. vmin and vmax
        are not set.
    return_fig
        if set to True, the figure is returned.
    ax
        Optional parameter. Sets user defined ax from for example plt.subplots
    *args
        arguments ultimately passed to sc.pl.umap
    **kwargs
        arguments ultimately passed to sc.pl.umap

    Returns
    -------
    if `show==False` a :class:`~matplotlib.axes.Axes`
    
    """
 
    basis = f"X_{dimred}_{gate}_{layer}"
    neighbors_key = f"{gate}_{layer}_neighbors"
    dimred = "UMAP"

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
                              ax = ax,
                              *args,
                              **kwargs)
    
    if return_fig:
        return fig

    savefig_or_show(show = show, save = save)

    if show is False:
        return fig
