from matplotlib.axes import Axes
import pandas as pd

from ..utils import create_comparisons

from statannotations.Annotator import Annotator


def add_statistic(ax: Axes,
                  test: str,
                  dataframe: pd.DataFrame,
                  groupby: str,
                  plot_params) -> Axes:
    
    pairs = create_comparisons(dataframe, groupby)
    annotator = Annotator(ax,
                          pairs,
                          **plot_params,
                          verbose = False)
    annotator.configure(test = test, text_format = "star", loc = "inside")
    annotator.apply_and_annotate()

    return ax