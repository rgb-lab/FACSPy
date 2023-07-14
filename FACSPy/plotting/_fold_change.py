import pandas as pd
import numpy as np
from anndata import AnnData
from matplotlib import pyplot as plt
import seaborn as sns

from matplotlib.patches import Patch

from ..tools._fold_change import calculate_fold_changes

from typing import Literal, Union

from ..utils import ifelse

from .utils import savefig_or_show

def fold_change(adata: AnnData,
                groupby: str,
                group1: Union[str, list[Union[str, int]]],
                group2: Union[str, list[Union[str, int]]],
                gate: str,
                stat: Literal["p", "p_adj"] = "p",
                cmap: str = "Reds_r",
                test: str = "Kruskal",
                return_dataframe: bool = False,
                return_fig: bool = False,
                save: bool = None,
                show: bool = None
                ):
       
       fold_changes = calculate_fold_changes(adata = adata,
                                                 groupby = groupby,
                                                 group1 = group1,
                                                 group2 = group2,
                                                 gate = gate,
                                                 test = test)
       
       fold_changes = fold_changes.sort_values("asinh_fc", ascending = False)
       fold_changes = fold_changes.reset_index()

       if return_dataframe:
              return fold_changes

       fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (3, len(fold_changes)/5))
       cmap_colors = sns.color_palette(cmap, 4)
       p_color = [ifelse(x < 0.0001, cmap_colors[0],
                            ifelse(x < 0.001, cmap_colors[1],
                                   ifelse(x < 0.01, cmap_colors[2],
                                          ifelse(x < 0.05, cmap_colors[3], "grey")))) for x in fold_changes[stat].to_list()]
                                   

       sns.barplot(data = fold_changes,
                     x = "asinh_fc",
                     y = "index",
                     palette = p_color,
                     ax = ax)
       ax.set_title(f"Comparison\n{group1} vs {group2}")
       
       ax.set_yticklabels(ax.get_yticklabels(), fontsize = 10)
       ax.set_ylabel("antigen")

       group_lut = {"p < 0.0001": cmap_colors[0],
                     "p < 0.001": cmap_colors[1],
                     "p < 0.01": cmap_colors[2],
                     "p < 0.05": cmap_colors[3],
                     "n.s.": "grey"}
       
       handles = [Patch(facecolor = group_lut[name]) for name in group_lut]
       ax.legend(handles,
                     group_lut,
                     bbox_to_anchor = (1.1,0.5),
                     title = "p_signif."
                     )
       
       plt.tight_layout()
       if return_fig:
              return fig

       savefig_or_show(save = save, show = show)
       
       plt.show()