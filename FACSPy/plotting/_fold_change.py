from anndata import AnnData
import matplotlib
import seaborn as sns
import pandas as pd
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib import pyplot as plt
import numpy as np

from matplotlib.patches import Patch

from ..tools._fold_change import _calculate_fold_changes

from typing import Literal, Union, Optional

from .._utils import ifelse, _default_gate_and_default_layer

from ._utils import savefig_or_show

def _create_custom_cbar(cmap: str,
                        fold_changes: pd.DataFrame,
                        stat: str):
       custom_cmap = matplotlib.colormaps[cmap]
       lognorm = LogNorm(vmin = fold_changes[stat].min(), vmax = 0.1)
       not_sig_cutoff = int(lognorm(0.05) * 256 - 256) * -1
       custom_colors = custom_cmap(np.linspace(0,1,256 - not_sig_cutoff))
       gray = np.array([0.5, 0.5, 0.5, 1])
       custom_colors = np.vstack([custom_colors, np.tile(gray, (not_sig_cutoff,1))])

       sm = plt.cm.ScalarMappable(cmap = ListedColormap(custom_colors),
                                  norm = lognorm)
       zeroed = (np.log10(fold_changes[stat].tolist()) - np.min(np.log10(fold_changes[stat].tolist())))
       zeroed = zeroed / np.max(zeroed)
       p_color = sm.cmap(zeroed)
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
                figsize: tuple[float, float] = (4,10),
                return_dataframe: bool = False,
                return_fig: bool = False,
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
                                                stat = stat)

       fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = figsize)
       sns.barplot(data = fold_changes,
                   x = "asinh_fc",
                   y = "index",
                   palette = p_colors,
                   ax = ax)
       ax.set_title(f"enriched in\n{group1}     {group2}")
       ax.set_yticklabels(ax.get_yticklabels(), fontsize = 10)
       ax.set_ylabel("antigen")

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

       plt.tight_layout()
       if return_fig:
              return fig

       savefig_or_show(save = save, show = show)