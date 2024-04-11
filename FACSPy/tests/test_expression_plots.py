import pytest

import os
import FACSPy as fp
from anndata import AnnData
import pandas as pd

from pathlib import Path
from matplotlib.testing.decorators import image_comparison

import matplotlib
matplotlib.use("agg")

HERE: Path = Path(__file__).parent
ROOT = os.path.join(HERE, "_images")

IMG_COMP_KWARGS = {
    "extensions": ['png'],
    "style": 'mpl20',
    "tol": 2
}

@pytest.fixture
def mouse_data() -> AnnData:
    adata = fp.mouse_lineages()
    fp.tl.gate_frequencies(adata)
    fp.tl.mfi(adata, layer = "compensated")
    fp.tl.fop(adata, layer = "compensated")
    gate = "Neutrophils"
    layer = "compensated"
    fp.tl.pca(adata, gate = gate, layer = layer)
    fp.tl.neighbors(adata, gate = gate, layer = layer)
    fp.tl.leiden(adata, gate = gate, layer = layer)
    fp.tl.mfi(adata,
              groupby = "Neutrophils_compensated_leiden",
              aggregate = False)
    fp.tl.fop(adata,
              groupby = "Neutrophils_compensated_leiden",
              aggregate = False)

    return adata


# fold change


@image_comparison(baseline_images = ['fold_change_mfi'],
                  **IMG_COMP_KWARGS)
def test_fold_change(mouse_data):
    fp.pl.fold_change(mouse_data,
                      gate = "Neutrophils",
                      layer = "compensated",
                      groupby = "sex",
                      group1 = "m",
                      group2 = "f",
                      figsize = (6, 9),
                      show = False)


@image_comparison(baseline_images = ['fold_change_plot_annotation'],
                  **IMG_COMP_KWARGS)
def test_fold_change_plot_annotation(mouse_data):
    fp.pl.fold_change(mouse_data,
                      gate = "Neutrophils",
                      layer = "compensated",
                      groupby = "sex",
                      group1 = "m",
                      group2 = "f",
                      comparison_label = "COMPARISON",
                      group1_label = "male",
                      group2_label = "female",
                      figsize = (6, 9),
                      show = False)


def test_fold_change_dataframe(mouse_data):
    df = fp.pl.fold_change(mouse_data,
                           gate = "Neutrophils",
                           layer = "compensated",
                           groupby = "sex",
                           group1 = "m",
                           group2 = "f",
                           comparison_label = "COMPARISON",
                           group1_label = "male",
                           group2_label = "female",
                           show = False,
                           return_dataframe = True)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["index", "group1", "group2", "asinh_fc", "p", "p_adj"])


# marker expressions


@image_comparison(baseline_images = ['marker_density'],
                  **IMG_COMP_KWARGS)
def test_marker_density(mouse_data):
    fp.pl.marker_density(mouse_data,
                         gate = "Neutrophils",
                         layer = "compensated",
                         marker = "Ly6G",
                         groupby = "experiment",
                         ridge = True,
                         plot_aspect = 2,
                         figsize = (5, 4),
                         show = False)


@image_comparison(baseline_images = ['marker_density_line'],
                  **IMG_COMP_KWARGS)
def test_marker_density_line(mouse_data):
    fp.pl.marker_density(mouse_data,
                         gate = "Neutrophils",
                         layer = "compensated",
                         marker = "Ly6G",
                         groupby = "experiment",
                         ridge = False,
                         figsize = (5, 5),
                         show = False)
