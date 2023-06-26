import pandas as pd
import numpy as np
from anndata import AnnData
from matplotlib import pyplot as plt
import seaborn as sns

from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from ..analysis._fold_change import calculate_fold_changes

from typing import Literal

from ..utils import ifelse

def fold_change(adata: AnnData,
                groupby: str,
                group1: str,
                group2: str,
                gate: str,
                stat: Literal["p", "p_adj"] = "p",
                cmap: Literal["YlOrBr_r"] = "YlOrBr_r"
                ):
    
    fold_changes = calculate_fold_changes(adata = adata,
                                          groupby = groupby,
                                          group1 = group1,
                                          group2 = group2,
                                          gate = gate)
    
    fold_changes = fold_changes.sort_values("asinh_fc", ascending = False)
    fold_changes = fold_changes.reset_index()

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
    plt.show()