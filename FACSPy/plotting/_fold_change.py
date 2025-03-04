from anndata import AnnData
import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from typing import Literal, Union, Optional

from ._utils import savefig_or_show
from ..tools._fold_change import _calculate_fold_changes
from .._utils import (_default_gate_and_default_layer,
                      _enable_gate_aliases,
                      _fetch_fluo_channels)


def _create_custom_cbar(cmap: str,
                        fold_changes: pd.DataFrame,
                        stat: str,
                        min_pval: Optional[float]):
    custom_cmap = matplotlib.colormaps[cmap]
    if min_pval:
           vmin = min_pval
    else:
           if fold_changes[stat].min() >= 0.1:
                  vmin = 1e-5
           else:
                  vmin = fold_changes[stat].min()
    lognorm = LogNorm(vmin = vmin,
                      vmax = 0.1)
    not_sig_cutoff = int(lognorm(0.05) * 256 - 256) * -1
    custom_colors = custom_cmap(np.linspace(0,1,256 - not_sig_cutoff))
    gray = np.array([0.5, 0.5, 0.5, 1])
    custom_colors = np.vstack([custom_colors, np.tile(gray, (not_sig_cutoff,1))])

    sm = plt.cm.ScalarMappable(cmap = ListedColormap(custom_colors),
                               norm = lognorm)
    p_color = sm.cmap(lognorm(fold_changes[stat].tolist()))
    return sm, p_color


@_default_gate_and_default_layer
@_enable_gate_aliases
def fold_change(adata: AnnData,
                gate: str,
                layer: str,
                groupby: str,
                group1: Union[list[Union[str, int]], str],
                group2: Union[list[Union[str, int]], str],
                data_group: str = "sample_ID",
                data_metric: Literal["mfi", "fop"] = "mfi",
                include_technical_channels: bool = False,
                exclude: Optional[Union[list[str], str]] = None,
                stat: Literal["p", "p_adj"] = "p",
                cmap: Optional[str] = "Reds_r",
                test: Literal["Kruskal", "Wilcoxon"] = "Kruskal",
                min_pval: Optional[float] = None,
                comparison_label: Optional[str] = None,
                group1_label: Optional[str] = None,
                group2_label: Optional[str] = None,
                figsize: tuple[float, float] = (4,10),
                return_dataframe: bool = False,
                return_fig: bool = False,
                ax: Optional[Axes] = None,
                show: bool = True,
                save: Optional[str] = None
                ) -> Optional[Union[Figure, Axes, pd.DataFrame]]:
    """\
    Plots the asinh fold change.

    The asinh fold change is calculated from the MFI values of the compensated data.
    It is similar to the log2fc, but since log does not allow negative values,
    we calculate the asinh foldchange as asinh(mean(group2)) - asinh(mean(group1)).
    The p_value is calculated using the specified test and the asinh transformed
    expression values per sample. 

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where Rows correspond to cells and columns to the channels
    gate
        The gate to be analyzed, called by the population name.
        This parameter has a default stored in fp.settings, but
        can be superseded by the user.
    layer
        The layer corresponding to the data matrix. Similar to the
        gate parameter, it has a default stored in fp.settings which
        can be overwritten by user input.
    groupby
        The variable of both groups that are compared
    group1
        The first group to be compared
    group2
        The second group to be compared
    data_group
        When MFIs/FOPs are calculated, and the groupby parameter is used,
        use `data_group` to specify the right dataframe
    data_metric
        One of `mfi` or `fop`. Using a different metric will calculate
        the asinh fold change on mfi and fop values, respectively
    include_technical_channels
        Whether to include technical channels. If set to False, will exclude
        all channels that are not labeled with `type=="fluo"` in adata.var.
    exclude
        Channels to be excluded from plotting.
    stat
        One of `p` or `p_adj`. Specifies whether to show the calculated
        p value or the adjusted p value.
    cmap
        colormap for the p-value colorbar.
    test
        statistical test that is used for the p-value calculation. One of
        `Kruskal` and `Wilcoxon`. Defaults to Kruskal.
    min_pval
        minimum p_value that is still displayed
    comparison_label
        Sets the title for the comparison
    group1_label
        Sets the labeling for the first group
    group2_label
        Sets the labeling for the second group
    figsize
        Contains the dimensions of the final figure as a tuple of two ints or floats.
    return_dataframe
        If set to True, returns the raw data that are used for plotting as a dataframe.
    return_fig
        If set to True, the figure is returned.
    ax
        A :class:`~matplotlib.axes.Axes` created from matplotlib to plot into.
    show
        Whether to show the figure. Defaults to True.
    save
        Expects a file path including the file name.
        Saves the figure to the indicated path. Defaults to None.

    Returns
    -------
    If `show==False` a :class:`~matplotlib.axes.Axes`
    If `return_fig==True` a :class:`~matplotlib.figure.Figure`
    If `return_dataframe==True` a :class:`~pandas.DataFrame` containing the data used for plotting

    Examples
    --------
    .. plot::
        :context: close-figs

        import FACSPy as fp

        dataset = fp.mouse_lineages()

        fp.tl.mfi(dataset)

        fp.pl.fold_change(
            dataset,
            gate = "B_cells",
            groupby = "organ",
            group1 = "BM",
            group2 = "PB",
            figsize = (3,5)
        )

    """
       
    fold_changes = _calculate_fold_changes(adata = adata,
                                           groupby = groupby,
                                           group1 = group1,
                                           group2 = group2,
                                           gate = gate,
                                           data_group = data_group,
                                           data_metric = data_metric,
                                           layer = layer,
                                           test = test)
    fold_changes = fold_changes.sort_values("asinh_fc", ascending = False)
    fold_changes = fold_changes.reset_index()

    if not include_technical_channels:
         fluo_channels = _fetch_fluo_channels(adata)
         fold_changes = fold_changes[fold_changes["index"].isin(fluo_channels)]
    if exclude is not None:
        if not isinstance(exclude, list):
            exclude = [exclude]
        fold_changes = fold_changes[~fold_changes["index"].isin(exclude)]

    if return_dataframe:
        assert isinstance(fold_changes, pd.DataFrame)
        return fold_changes

    colorbar, p_colors = _create_custom_cbar(cmap = cmap,
                                             fold_changes = fold_changes,
                                             stat = stat,
                                             min_pval = min_pval)

    if ax is None:
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(111)
    sns.barplot(data = fold_changes,
                x = "asinh_fc",
                y = "index",
                palette = p_colors,
                ax = ax)
    ax.set_title(f"enriched in\n{comparison_label or groupby}\n{group1_label or group1}       {group2_label or group2}")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = 10)
    ax.set_ylabel("antigen")
    ax.set_xlim(-np.max(np.abs(ax.get_xlim())),
                 np.max(np.abs(ax.get_xlim())))

    cbar = ax.figure.colorbar(colorbar,
                              ax = ax)
    cbar.ax.set_ylabel(f"{stat} value", rotation = 270, labelpad = 25)
    cbar.ax.text(0.55,
                 0.07,
                 "ns",
                 ha='center',
                 va='center',
                 color = "white",
                 weight = "bold")

    if return_fig:
        return fig

    savefig_or_show(save = save, show = show)
    if show is False:
        return ax
