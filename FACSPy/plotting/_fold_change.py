from anndata import AnnData
import matplotlib
import seaborn as sns
import pandas as pd
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from ..tools._fold_change import _calculate_fold_changes

from typing import Literal, Union, Optional

from .._utils import _default_gate_and_default_layer

from ._utils import savefig_or_show

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
def fold_change(adata: AnnData,
                gate: str = None,
                layer: str = None,
                groupby: str = None,
                group1: Union[str, list[Union[str, int]]] = None,
                group2: Union[str, list[Union[str, int]]] = None,
                data_group: Optional[Union[str, list[str]]] = "sample_ID",
                data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                stat: Literal["p", "p_adj"] = "p",
                cmap: str = "Reds_r",
                test: Literal["Kruskal", "Wilcoxon"] = "Kruskal",
                min_pval: float = None,
                figsize: tuple[float, float] = (4,10),
                comparison_label: Optional[str] = None,
                group1_label: Optional[str] = None,
                group2_label: Optional[str] = None,
                return_dataframe: bool = False,
                return_fig: bool = False,
                ax: Axes = None,
                save: bool = None,
                show: bool = None
                ):
       
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

       if return_dataframe:
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