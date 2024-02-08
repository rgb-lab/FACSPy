import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
from statannotations.Annotator import Annotator

from typing import Optional

from ._utils import (CATEGORICAL_BOXPLOT_PARAMS,
                     CATEGORICAL_STRIPPLOT_PARAMS)
from .._utils import _create_comparisons

def _add_statistic(ax: Axes,
                   test: str,
                   dataframe: pd.DataFrame,
                   groupby: str,
                   plot_params: dict,
                   splitby: Optional[str] = None) -> Axes:
    
    pairs = _create_comparisons(dataframe, groupby, splitby)
    annotator = Annotator(ax,
                          pairs,
                          **plot_params,
                          verbose = False)
    annotator.configure(test = test, text_format = "star", loc = "inside")
    annotator.apply_and_annotate()

    return ax

def _categorical_strip_box_plot(ax: Optional[Axes],
                                data: pd.DataFrame,
                                plot_params: dict,
                                groupby: str,
                                splitby: str,
                                stat_test: Optional[str],
                                figsize: tuple[float, float]):

    if ax is None:
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(111)
    else:
        fig = None

    if groupby == "sample_ID":
        if plot_params["hue"]:
            raise TypeError("You selected a splitby parameter while plotting sample ID. Don't.")
        sns.barplot(**plot_params)

    else:
        sns.stripplot(**plot_params,
                      **CATEGORICAL_STRIPPLOT_PARAMS)
        handles, labels = ax.get_legend_handles_labels()
        sns.boxplot(**plot_params,
                    **CATEGORICAL_BOXPLOT_PARAMS)

        if stat_test:
            try:
                ax = _add_statistic(ax = ax,
                                    test = stat_test,
                                    dataframe = data,
                                    groupby = groupby,
                                    splitby = splitby,
                                    plot_params = plot_params)
            except ValueError as e:
                if str(e) != "All numbers are identical in kruskal":
                    raise ValueError(str(e)) from e
                else:
                    print("warning... Values were uniform, no statistics to plot.")

    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = "center")

    if splitby is not None:
        ax.legend(handles,
                  labels,
                  bbox_to_anchor = (1.1, 0.5),
                  loc = "center left",
                  title = splitby or None)
    else:
        ax.legend().remove()
    
    return fig, ax        
