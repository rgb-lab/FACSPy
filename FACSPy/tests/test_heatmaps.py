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
    "savefig_kwarg": {"bbox_inches": "tight"}
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

# expression_heatmap


@image_comparison(baseline_images = ['expression_heatmap_mfi'],
                  **IMG_COMP_KWARGS)
def test_expression_heatmap_mfi(mouse_data):
    fp.pl.expression_heatmap(mouse_data,
                             gate = "Neutrophils",
                             layer = "compensated",
                             metadata_annotation = ["organ", "sex", "experiment"],
                             marker_annotation = "Ly6G",
                             metaclusters = 3,
                             show = False)


@image_comparison(baseline_images = ['expression_heatmap_technicals'],
                  **IMG_COMP_KWARGS)
def test_expression_heatmap_mfi_technicals(mouse_data):
    fp.pl.expression_heatmap(mouse_data,
                             gate = "Neutrophils",
                             layer = "compensated",
                             include_technical_channels = True,
                             metadata_annotation = ["organ", "sex", "experiment"],
                             marker_annotation = "Ly6G",
                             metaclusters = 3,
                             show = False)


@image_comparison(baseline_images = ['expression_heatmap_fop'],
                  **IMG_COMP_KWARGS)
def test_expression_heatmap_fop(mouse_data):
    fp.pl.expression_heatmap(mouse_data,
                             gate = "Neutrophils",
                             layer = "compensated",
                             data_metric = "fop",
                             metadata_annotation = ["organ", "sex", "experiment"],
                             marker_annotation = "Ly6G",
                             metaclusters = 3,
                             show = False)
    assert "metacluster_sc" in mouse_data.uns["metadata"].dataframe.columns


@image_comparison(baseline_images = ['expression_heatmap_figsize'],
                  **IMG_COMP_KWARGS)
def test_expression_heatmap_figsize(mouse_data):
    fp.pl.expression_heatmap(mouse_data,
                             gate = "Neutrophils",
                             layer = "compensated",
                             data_metric = "mfi",
                             metadata_annotation = ["organ", "sex", "experiment"],
                             marker_annotation = "Ly6G",
                             metaclusters = 3,
                             label_metaclusters_key = "test_mc",
                             figsize = (7, 5),
                             show = False)
    assert "test_mc" in mouse_data.uns["metadata"].dataframe.columns


def test_expression_heatmap_dataframe(mouse_data):
    df = fp.pl.expression_heatmap(mouse_data,
                                  gate = "Neutrophils",
                                  layer = "compensated",
                                  data_metric = "mfi",
                                  include_technical_channels = True,
                                  metadata_annotation = ["organ", "sex", "experiment"],
                                  marker_annotation = "Ly6G",
                                  figsize = (7, 5),
                                  show = False,
                                  return_dataframe = True)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID", "gate"] +
               mouse_data.uns["metadata"].get_factors() +
               mouse_data.var_names.tolist())
    assert df["gate"].nunique() == 1


# cluster heatmap

@image_comparison(baseline_images = ['cluster_heatmap_mfi'],
                  **IMG_COMP_KWARGS)
def test_cluster_heatmap_mfi(mouse_data):
    fp.pl.cluster_heatmap(mouse_data,
                          gate = "Neutrophils",
                          layer = "compensated",
                          cluster_key = "Neutrophils_compensated_leiden",
                          annotate = "Ly6G",
                          show = False)


@image_comparison(baseline_images = ['cluster_heatmap_technicals'],
                  **IMG_COMP_KWARGS)
def test_cluster_heatmap_heatmap_mfi_technicals(mouse_data):
    fp.pl.cluster_heatmap(mouse_data,
                          gate = "Neutrophils",
                          layer = "compensated",
                          include_technical_channels = True,
                          cluster_key = "Neutrophils_compensated_leiden",
                          annotate = "Ly6G",
                          show = False)


@image_comparison(baseline_images = ['cluster_heatmap_fop'],
                  **IMG_COMP_KWARGS)
def test_cluster_heatmap_fop(mouse_data):
    fp.pl.cluster_heatmap(mouse_data,
                          gate = "Neutrophils",
                          layer = "compensated",
                          data_metric = "fop",
                          cluster_key = "Neutrophils_compensated_leiden",
                          annotate = "Ly6G",
                          show = False)



@image_comparison(baseline_images = ['cluster_heatmap_figsize'],
                  **IMG_COMP_KWARGS)
def test_cluster_heatmap_figsize(mouse_data):
    fp.pl.cluster_heatmap(mouse_data,
                          gate = "Neutrophils",
                          layer = "compensated",
                          data_metric = "fop",
                          figsize = (7, 5),
                          cluster_key = "Neutrophils_compensated_leiden",
                          annotate = "Ly6G",
                          show = False)


@image_comparison(baseline_images = ['cluster_heatmap_aggregated'],
                  **IMG_COMP_KWARGS)
def test_cluster_heatmap_aggregated(mouse_data):
    fp.tl.fop(mouse_data,
              groupby = "Neutrophils_compensated_leiden",
              aggregate = True)
    fp.pl.cluster_heatmap(mouse_data,
                          gate = "Neutrophils",
                          layer = "compensated",
                          data_metric = "fop",
                          figsize = (7, 5),
                          cluster_key = "Neutrophils_compensated_leiden",
                          annotate = "Ly6G",
                          show = False)


def test_cluster_heatmap_dataframe(mouse_data):
    df = fp.pl.cluster_heatmap(mouse_data,
                               gate = "Neutrophils",
                               layer = "compensated",
                               data_metric = "fop",
                               include_technical_channels = True,
                               cluster_key = "Neutrophils_compensated_leiden",
                               return_dataframe = True,
                               annotate = "Ly6G",
                               show = False)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID", "gate", "Neutrophils_compensated_leiden"] +
               mouse_data.uns["metadata"].get_factors() +
               mouse_data.var_names.tolist())
    assert df["gate"].nunique() == 1


# sample distance


@image_comparison(baseline_images = ['sample_dist_mfi'],
                  **IMG_COMP_KWARGS)
def test_sample_dist_mfi(mouse_data):
    fp.pl.sample_distance(mouse_data,
                          gate = "Neutrophils",
                          layer = "compensated",
                          metadata_annotation = ["organ", "sex", "experiment"],
                          metaclusters = 3,
                          show = False)


@image_comparison(baseline_images = ['sample_dist_technicals'],
                  **IMG_COMP_KWARGS)
def test_sample_dist_mfi_technicals(mouse_data):
    fp.pl.sample_distance(mouse_data,
                          gate = "Neutrophils",
                          layer = "compensated",
                          include_technical_channels = True,
                          metadata_annotation = ["organ", "sex", "experiment"],
                          metaclusters = 3,
                          show = False)


@image_comparison(baseline_images = ['sample_dist_heatmap_fop'],
                  **IMG_COMP_KWARGS)
def test_sample_dist_fop(mouse_data):
    fp.pl.sample_distance(mouse_data,
                          gate = "Neutrophils",
                          layer = "compensated",
                          data_metric = "fop",
                          metadata_annotation = ["organ", "sex", "experiment"],
                          metaclusters = 3,
                          show = False)
    assert "sample_distance_metaclusters" in mouse_data.uns["metadata"].dataframe.columns


@image_comparison(baseline_images = ['sample_dist_heatmap_figsize'],
                  **IMG_COMP_KWARGS)
def test_sample_dist_figsize(mouse_data):
    fp.pl.sample_distance(mouse_data,
                          gate = "Neutrophils",
                          layer = "compensated",
                          data_metric = "fop",
                          metadata_annotation = ["organ", "sex", "experiment"],
                          metaclusters = 3,
                          figsize = (3, 3),
                          label_metaclusters_key = "test_mc",
                          show = False)
    assert "test_mc" in mouse_data.uns["metadata"].dataframe.columns


def test_sample_dist_dataframe(mouse_data):
    df = fp.pl.sample_distance(mouse_data,
                               gate = "Neutrophils",
                               layer = "compensated",
                               data_metric = "fop",
                               metadata_annotation = ["organ", "sex", "experiment"],
                               figsize = (3, 3),
                               show = False,
                               return_dataframe = True)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID"] +
               mouse_data.uns["metadata"].get_factors())


# marker correlation

@image_comparison(baseline_images = ['marker_corr_mfi'],
                  **IMG_COMP_KWARGS)
def test_marker_corr_mfi(mouse_data):
    fp.pl.marker_correlation(mouse_data,
                             gate = "Neutrophils",
                             layer = "compensated",
                             show = False)


@image_comparison(baseline_images = ['marker_corr_technicals'],
                  **IMG_COMP_KWARGS)
def test_marker_corr_technicals(mouse_data):
    fp.pl.marker_correlation(mouse_data,
                             gate = "Neutrophils",
                             layer = "compensated",
                             include_technical_channels = True,
                             show = False)

# sample correlation


@image_comparison(baseline_images = ['sample_corr_mfi'],
                  **IMG_COMP_KWARGS)
def test_sample_corr_mfi(mouse_data):
    fp.pl.sample_correlation(mouse_data,
                             gate = "Neutrophils",
                             layer = "compensated",
                             metadata_annotation = ["organ", "sex", "experiment"],
                             metaclusters = 3,
                             show = False)


@image_comparison(baseline_images = ['sample_corr_technicals'],
                  **IMG_COMP_KWARGS)
def test_sample_corr_mfi_technicals(mouse_data):
    fp.pl.sample_correlation(mouse_data,
                             gate = "Neutrophils",
                             layer = "compensated",
                             include_technical_channels = True,
                             metadata_annotation = ["organ", "sex", "experiment"],
                             metaclusters = 3,
                             show = False)


@image_comparison(baseline_images = ['sample_corr_heatmap_fop'],
                  **IMG_COMP_KWARGS)
def test_sample_corr_fop(mouse_data):
    fp.pl.sample_correlation(mouse_data,
                             gate = "Neutrophils",
                             layer = "compensated",
                             data_metric = "fop",
                             metadata_annotation = ["organ", "sex", "experiment"],
                             metaclusters = 3,
                             show = False)
    assert "metacluster_sc" in mouse_data.uns["metadata"].dataframe.columns


@image_comparison(baseline_images = ['sample_corr_figsize'],
                  **IMG_COMP_KWARGS)
def test_sample_corr_figsize(mouse_data):
    fp.pl.sample_correlation(mouse_data,
                             gate = "Neutrophils",
                             layer = "compensated",
                             data_metric = "fop",
                             metadata_annotation = ["organ", "sex", "experiment"],
                             metaclusters = 3,
                             figsize = (3, 3),
                             label_metaclusters_key = "test_mc",
                             show = False)
    assert "test_mc" in mouse_data.uns["metadata"].dataframe.columns


def test_sample_corr_dataframe(mouse_data):
    df = fp.pl.sample_correlation(mouse_data,
                                  gate = "Neutrophils",
                                  layer = "compensated",
                                  data_metric = "fop",
                                  metadata_annotation = ["organ", "sex", "experiment"],
                                  figsize = (3, 3),
                                  show = False,
                                  return_dataframe = True)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID"] +
               mouse_data.uns["metadata"].get_factors())
