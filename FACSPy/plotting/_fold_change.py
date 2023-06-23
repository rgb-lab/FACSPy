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

def fold_change(adata: AnnData,
                groupby: str,
                group1: str,
                group2: str,
                gate: str,
                stat: Literal["p", "p_adj"] = "p"
                ):
    
    fold_changes = calculate_fold_changes(adata = adata,
                                          groupby = groupby,
                                          group1 = group1,
                                          group2 = group2,
                                          gate = gate)
    
    fold_changes = fold_changes.sort_values("asinh_fc", ascending = False)
    fold_changes = fold_changes.reset_index()

    fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (3, len(fold_changes)/5))
    p_color = ['red' if (x < 0.05) else 'grey' for x in fold_changes[stat]]

    sns.barplot(data = fold_changes,
                x = "asinh_fc",
                y = "index",
                palette = p_color,
                ax = ax)
    ax.set_title(f"Comparison\n{group1} vs {group2}")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = 10)
    
    plt.tight_layout()
    plt.show()