from matplotlib.axes import Axes
import seaborn as sns

def adjust_legend(ax: Axes) -> Axes:
    ax.legend(loc = "center left",
              bbox_to_anchor = (1.01, 0.5))
    return ax


def label_plot_basic(ax: Axes,
                     title: str,
                     x_label: str,
                     y_label: str) -> Axes:
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    return ax

def barplot(ax: Axes,
            plot_params: dict) -> Axes:
    sns.barplot(**plot_params,
                ax = ax,
                dodge = False)
    return ax

def stripboxplot(ax: Axes,
                 plot_params: dict) -> Axes:
    sns.stripplot(**plot_params,
                    dodge = False,
                    jitter = True,
                    linewidth = 1,
                    ax = ax)
    plot_params["hue"] = None
    sns.boxplot(**plot_params,
                boxprops = dict(facecolor = "white"),
                whis = (0,100),
                ax = ax)
    
    return ax