import pytest

from matplotlib import pyplot as plt
import os
import FACSPy as fp
from anndata import AnnData
import pandas as pd

from pathlib import Path
from matplotlib.testing.decorators import image_comparison
import numpy as np

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


# MFI Plot

@image_comparison(baseline_images = ['mfi_compensated_sex'],
                  **IMG_COMP_KWARGS)
def test_mfi_plot(mouse_data):
    fp.pl.mfi(mouse_data,
              marker = "Ly6G",
              gate = "Neutrophils",
              layer = "compensated",
              groupby = "sex",
              show = False)


@image_comparison(baseline_images = ['mfi_compensated_sampleID'],
                  **IMG_COMP_KWARGS)
def test_mfi_barplot(mouse_data):
    fp.pl.mfi(mouse_data,
              marker = "Ly6G",
              gate = "Neutrophils",
              layer = "compensated",
              groupby = "sample_ID",
              show = False)


@image_comparison(baseline_images = ['mfi_compensated_sex_figsize'],
                  **IMG_COMP_KWARGS)
def test_mfi_plot_figsize(mouse_data):
    fp.pl.mfi(mouse_data,
              marker = "Ly6G",
              gate = "Neutrophils",
              layer = "compensated",
              groupby = "sex",
              figsize = (1, 1),
              show = False)


@image_comparison(baseline_images = ['mfi_compensated_sex_experiment'],  # noqa
                  **IMG_COMP_KWARGS)
def test_mfi_plot_splitby(mouse_data):
    fp.pl.mfi(mouse_data,
              marker = "Ly6G",
              gate = "Neutrophils",
              layer = "compensated",
              groupby = "sex",
              splitby = "experiment",
              show = False)


@image_comparison(baseline_images = ['mfi_compensated_sex_experiment_set2'],
                  **IMG_COMP_KWARGS)
def test_mfi_plot_splitby_cmap(mouse_data):
    fp.pl.mfi(mouse_data,
              marker = "Ly6G",
              gate = "Neutrophils",
              layer = "compensated",
              groupby = "sex",
              cmap = "Set2",
              splitby = "experiment",
              show = False)


@image_comparison(baseline_images = ['mfi_compensated_ax_return'],
                  **IMG_COMP_KWARGS)
def test_mfi_plot_ax_return(mouse_data):
    _, ax = plt.subplots(ncols = 1,
                         nrows = 1,
                         figsize = (3, 4))
    fp.pl.mfi(mouse_data,
              marker = "Ly6G",
              gate = "Neutrophils",
              layer = "compensated",
              groupby = "sex",
              cmap = "Set2",
              splitby = "experiment",
              show = False,
              ax = ax)
    ax.set_title("test_plot")


@image_comparison(baseline_images = ['mfi_compensated_ax_return_double'],
                  **IMG_COMP_KWARGS)
def test_mfi_plot_ax_return_double(mouse_data):
    _, ax = plt.subplots(ncols = 2,
                         nrows = 1,
                         figsize = (4, 2))
    fp.pl.mfi(mouse_data,
              marker = "Ly6G",
              gate = "Neutrophils",
              layer = "compensated",
              groupby = "sex",
              cmap = "Set2",
              splitby = "experiment",
              show = False,
              ax = ax[0])
    ax[0].set_title("left plot")
    fp.pl.mfi(mouse_data,
              marker = "Ly6G",
              gate = "Neutrophils",
              layer = "compensated",
              groupby = "sex",
              cmap = "Set2",
              splitby = "experiment",
              show = False,
              ax = ax[1])
    ax[1].set_title("right plot")


def test_mfi_plot_dataframe(mouse_data):
    df = fp.pl.mfi(mouse_data,
                   marker = "Ly6G",
                   gate = "Neutrophils",
                   layer = "compensated",
                   groupby = "sex",
                   return_dataframe = True)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID", "gate"] +
               mouse_data.uns["metadata"].get_factors() +
               mouse_data.var_names.tolist())
    assert df["gate"].nunique() == 1
    neus = fp.subset_gate(mouse_data, "Neutrophils", copy = True)
    assert isinstance(neus, AnnData)
    s3: pd.DataFrame = neus[
        neus.obs["sample_ID"] == "3"
    ].to_df(layer = "compensated")
    medians: pd.Series = s3.median(axis = 0)
    median_fsca = medians.loc[medians.index == "FSC-A"].iloc[0]
    assert df.loc[
        df["sample_ID"] == "3",
        "FSC-A"
    ].iloc[0] == median_fsca


def test_mfi_plot_dataframe_splitby(mouse_data):
    df = fp.pl.mfi(mouse_data,
                   marker = "Ly6G",
                   gate = "Neutrophils",
                   layer = "compensated",
                   groupby = "sex",
                   splitby = "experiment",
                   return_dataframe = True)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID", "gate"] +
               mouse_data.uns["metadata"].get_factors() +
               mouse_data.var_names.tolist())
    assert df["gate"].nunique() == 1
    neus = fp.subset_gate(mouse_data, "Neutrophils", copy = True)
    assert isinstance(neus, AnnData)
    s3: pd.DataFrame = neus[
        neus.obs["sample_ID"] == "3"
    ].to_df(layer = "compensated")
    medians: pd.Series = s3.median(axis = 0)
    median_fsca = medians.loc[medians.index == "FSC-A"].iloc[0]
    assert df.loc[
        df["sample_ID"] == "3",
        "FSC-A"
    ].iloc[0] == median_fsca


# FOP Plot

@image_comparison(baseline_images = ['fop_compensated_sex'],
                  **IMG_COMP_KWARGS)
def test_fop_plot(mouse_data):
    fp.pl.fop(mouse_data,
              marker = "Ly6G",
              gate = "Neutrophils",
              layer = "compensated",
              groupby = "sex",
              show = False)


@image_comparison(baseline_images = ['fop_compensated_sampleID'],
                  **IMG_COMP_KWARGS)
def test_fop_barplot(mouse_data):
    fp.pl.fop(mouse_data,
              marker = "Ly6G",
              gate = "Neutrophils",
              layer = "compensated",
              groupby = "sample_ID",
              show = False)


@image_comparison(baseline_images = ['fop_compensated_sex_figsize'],
                  **IMG_COMP_KWARGS)
def test_fop_plot_figsize(mouse_data):
    fp.pl.fop(mouse_data,
              marker = "Ly6G",
              gate = "Neutrophils",
              layer = "compensated",
              groupby = "sex",
              figsize = (1, 1),
              show = False)


@image_comparison(baseline_images = ['fop_compensated_ax_return'],
                  **IMG_COMP_KWARGS)
def test_fop_plot_ax_return(mouse_data):
    _, ax = plt.subplots(ncols = 1,
                         nrows = 1,
                         figsize = (3, 3))
    fp.pl.fop(mouse_data,
              marker = "Ly6G",
              gate = "Neutrophils",
              layer = "compensated",
              groupby = "sex",
              cmap = "Set2",
              splitby = "experiment",
              show = False,
              ax = ax)
    ax.set_title("test_plot")


@image_comparison(baseline_images = ['fop_compensated_ax_return_double'],
                  **IMG_COMP_KWARGS)
def test_fop_plot_ax_return_double(mouse_data):
    _, ax = plt.subplots(ncols = 2,
                         nrows = 1,
                         figsize = (4, 2))
    fp.pl.fop(mouse_data,
              marker = "Ly6G",
              gate = "Neutrophils",
              layer = "compensated",
              groupby = "sex",
              cmap = "Set2",
              splitby = "experiment",
              show = False,
              ax = ax[0])
    ax[0].set_title("left plot")
    fp.pl.fop(mouse_data,
              marker = "Ly6G",
              gate = "Neutrophils",
              layer = "compensated",
              groupby = "sex",
              cmap = "Set2",
              splitby = "experiment",
              show = False,
              ax = ax[1])
    ax[1].set_title("right_plot")


@image_comparison(baseline_images = ['fop_compensated_sex_experiment'],  # noqa
                  **IMG_COMP_KWARGS)
def test_fop_plot_splitby(mouse_data):
    fp.pl.fop(mouse_data,
              marker = "Ly6G",
              gate = "Neutrophils",
              layer = "compensated",
              groupby = "sex",
              splitby = "experiment",
              show = False)


@image_comparison(baseline_images = ['fop_compensated_sex_experiment_set2'],
                  **IMG_COMP_KWARGS)
def test_fop_plot_splitby_cmap():
    adata = fp.mouse_lineages()
    fp.tl.fop(adata, layer = "compensated")
    fp.pl.fop(adata,
              marker = "Ly6G",
              gate = "Neutrophils",
              layer = "compensated",
              groupby = "sex",
              cmap = "Set2",
              splitby = "experiment",
              show = False)


def test_fop_plot_dataframe(mouse_data):
    df = fp.pl.fop(mouse_data,
                   marker = "Ly6G",
                   gate = "Neutrophils",
                   layer = "compensated",
                   groupby = "sex",
                   return_dataframe = True)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID", "gate"] +
               mouse_data.uns["metadata"].get_factors() +
               mouse_data.var_names.tolist())
    assert df["gate"].nunique() == 1
    neus = fp.subset_gate(mouse_data, "Neutrophils", copy = True)
    assert isinstance(neus, AnnData)
    s3: pd.DataFrame = neus[
        neus.obs["sample_ID"] == "3"
    ].to_df(layer = "compensated")
    pos = s3 > neus.var["cofactors"].astype(np.float64).tolist()
    freq_pos = pos.sum(axis = 0) / pos.shape[0]
    fop = freq_pos.loc[freq_pos.index == "Ly6G"].iloc[0]
    assert df.loc[
        df["sample_ID"] == "3",
        "Ly6G"
    ].iloc[0] == fop


def test_fop_plot_dataframe_splitby(mouse_data):
    df = fp.pl.fop(mouse_data,
                   marker = "Ly6G",
                   gate = "Neutrophils",
                   layer = "compensated",
                   groupby = "sex",
                   splitby = "experiment",
                   return_dataframe = True)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID", "gate"] +
               mouse_data.uns["metadata"].get_factors() +
               mouse_data.var_names.tolist())
    assert df["gate"].nunique() == 1
    neus = fp.subset_gate(mouse_data, "Neutrophils", copy = True)
    assert isinstance(neus, AnnData)
    s3: pd.DataFrame = neus[
        neus.obs["sample_ID"] == "3"
    ].to_df(layer = "compensated")
    pos = s3 > neus.var["cofactors"].astype(np.float64).tolist()
    freq_pos = pos.sum(axis = 0) / pos.shape[0]
    fop = freq_pos.loc[freq_pos.index == "Ly6G"].iloc[0]
    assert df.loc[
        df["sample_ID"] == "3",
        "Ly6G"
    ].iloc[0] == fop


# cluster mfi

@image_comparison(baseline_images = ['cluster_mfi'],
                  **IMG_COMP_KWARGS)
def test_cluster_mfi(mouse_data):
    fp.pl.cluster_mfi(mouse_data,
                      gate = "Neutrophils",
                      layer = "compensated",
                      cluster_key = "Neutrophils_compensated_leiden",
                      marker = "Ly6G",
                      stat_test = False,
                      show = False)


@image_comparison(baseline_images = ['cluster_mfi_figsize'],
                  **IMG_COMP_KWARGS)
def test_cluster_mfi_figsize(mouse_data):
    fp.pl.cluster_mfi(mouse_data,
                      gate = "Neutrophils",
                      layer = "compensated",
                      cluster_key = "Neutrophils_compensated_leiden",
                      marker = "Ly6G",
                      figsize = (2, 2),
                      stat_test = False,
                      show = False)


@image_comparison(baseline_images = ['cluster_mfi_ax_return'],
                  **IMG_COMP_KWARGS)
def test_cluster_mfi_ax_return(mouse_data):
    _, ax = plt.subplots(ncols = 1,
                         nrows = 1,
                         figsize = (3, 3))
    fp.pl.cluster_mfi(mouse_data,
                      gate = "Neutrophils",
                      layer = "compensated",
                      cluster_key = "Neutrophils_compensated_leiden",
                      marker = "Ly6G",
                      stat_test = False,
                      show = False,
                      ax = ax)
    ax.set_title("test_plot")


@image_comparison(baseline_images = ['cluster_mfi_ax_return_double'],
                  **IMG_COMP_KWARGS)
def test_cluster_mfi_ax_return_double(mouse_data):
    _, ax = plt.subplots(ncols = 2,
                         nrows = 1,
                         figsize = (4, 2))
    fp.pl.cluster_mfi(mouse_data,
                      gate = "Neutrophils",
                      layer = "compensated",
                      cluster_key = "Neutrophils_compensated_leiden",
                      stat_test = False,
                      marker = "Ly6G",
                      show = False,
                      ax = ax[0])
    ax[0].set_title("left plot")
    fp.pl.cluster_mfi(mouse_data,
                      gate = "Neutrophils",
                      layer = "compensated",
                      cluster_key = "Neutrophils_compensated_leiden",
                      stat_test = False,
                      marker = "CD3",
                      show = False,
                      ax = ax[1])
    ax[1].set_title("right_plot")


@image_comparison(baseline_images = ['cluster_mfi_splitby'],  # noqa
                  **IMG_COMP_KWARGS)
def test_cluster_mfi_splitby(mouse_data):
    fp.pl.cluster_mfi(mouse_data,
                      gate = "Neutrophils",
                      layer = "compensated",
                      cluster_key = "Neutrophils_compensated_leiden",
                      marker = "Ly6G",
                      stat_test = False,
                      splitby = "experiment",
                      show = False)


@image_comparison(baseline_images = ['cluster_mfi_cmap'],
                  **IMG_COMP_KWARGS)
def test_cluster_mfi_cmap(mouse_data):
    fp.pl.cluster_mfi(mouse_data,
                      gate = "Neutrophils",
                      layer = "compensated",
                      cluster_key = "Neutrophils_compensated_leiden",
                      marker = "Ly6G",
                      stat_test = False,
                      cmap = "Set2",
                      show = False)


def test_cluster_mfi_typerror(mouse_data):
    mouse_data = mouse_data.copy()
    fp.tl.mfi(mouse_data, groupby = "Neutrophils_compensated_leiden",
              aggregate = True)
    with pytest.raises(TypeError):
        fp.pl.cluster_mfi(mouse_data,
                          gate = "Neutrophils",
                          layer = "compensated",
                          cluster_key = "Neutrophils_compensated_leiden",
                          marker = "Ly6G",
                          stat_test = False,
                          cmap = "Set2",
                          show = False)


def test_cluster_mfi_dataframe(mouse_data):
    df = fp.pl.cluster_mfi(mouse_data,
                           gate = "Neutrophils",
                           layer = "compensated",
                           cluster_key = "Neutrophils_compensated_leiden",
                           marker = "Ly6G",
                           stat_test = False,
                           cmap = "Set2",
                           show = False,
                           return_dataframe = True)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID", "gate", "Neutrophils_compensated_leiden"] +
               mouse_data.uns["metadata"].get_factors() +
               mouse_data.var_names.tolist())
    assert df["gate"].nunique() == 1
    neus = fp.subset_gate(mouse_data, "Neutrophils", copy = True)
    assert isinstance(neus, AnnData)
    s3: pd.DataFrame = neus[
        neus.obs["sample_ID"] == "3"
    ].to_df(layer = "compensated")
    s3[neus.obs.columns] = neus.obs
    s3["Neutrophils_compensated_leiden"] = s3["Neutrophils_compensated_leiden"].astype(np.int8)
    medians: pd.DataFrame = s3.groupby(["Neutrophils_compensated_leiden"]).median(["Ly6G"])
    median_ly6g = medians.loc[medians.index == 0, "Ly6G"].iloc[0]
    assert df.loc[
        (df["sample_ID"] == "3") &
        (df["Neutrophils_compensated_leiden"] == "0"),
        "Ly6G"
    ].iloc[0] == median_ly6g


def test_cluster_mfi_dataframe_splitby(mouse_data):
    df = fp.pl.cluster_mfi(mouse_data,
                           gate = "Neutrophils",
                           layer = "compensated",
                           cluster_key = "Neutrophils_compensated_leiden",
                           marker = "Ly6G",
                           cmap = "Set2",
                           stat_test = False,
                           show = False,
                           splitby = "experiment",
                           return_dataframe = True)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID", "gate", "Neutrophils_compensated_leiden"] +
               mouse_data.uns["metadata"].get_factors() +
               mouse_data.var_names.tolist())
    assert df["gate"].nunique() == 1
    neus = fp.subset_gate(mouse_data, "Neutrophils", copy = True)
    assert isinstance(neus, AnnData)
    s3: pd.DataFrame = neus[
        neus.obs["sample_ID"] == "3"
    ].to_df(layer = "compensated")
    s3[neus.obs.columns] = neus.obs
    s3["Neutrophils_compensated_leiden"] = s3["Neutrophils_compensated_leiden"].astype(np.int8)
    medians: pd.DataFrame = s3.groupby(["Neutrophils_compensated_leiden"]).median(["Ly6G"])
    median_ly6g = medians.loc[medians.index == 0, "Ly6G"].iloc[0]
    assert df.loc[
        (df["sample_ID"] == "3") &
        (df["Neutrophils_compensated_leiden"] == "0"),
        "Ly6G"
    ].iloc[0] == median_ly6g


# cluster fop

@image_comparison(baseline_images = ['cluster_fop'],
                  **IMG_COMP_KWARGS)
def test_cluster_fop(mouse_data):
    fp.pl.cluster_fop(mouse_data,
                      gate = "Neutrophils",
                      layer = "compensated",
                      cluster_key = "Neutrophils_compensated_leiden",
                      marker = "Ly6G",
                      stat_test = False,
                      show = False)


@image_comparison(baseline_images = ['cluster_fop_figsize'],
                  **IMG_COMP_KWARGS)
def test_cluster_fop_figsize(mouse_data):
    fp.pl.cluster_fop(mouse_data,
                      gate = "Neutrophils",
                      layer = "compensated",
                      cluster_key = "Neutrophils_compensated_leiden",
                      marker = "Ly6G",
                      figsize = (2, 2),
                      stat_test = False,
                      show = False)


@image_comparison(baseline_images = ['cluster_fop_ax_return'],
                  **IMG_COMP_KWARGS)
def test_cluster_fop_ax_return(mouse_data):
    _, ax = plt.subplots(ncols = 1,
                         nrows = 1,
                         figsize = (3, 3))
    fp.pl.cluster_fop(mouse_data,
                      gate = "Neutrophils",
                      layer = "compensated",
                      cluster_key = "Neutrophils_compensated_leiden",
                      marker = "Ly6G",
                      show = False,
                      stat_test = False,
                      ax = ax)
    ax.set_title("test_plot")


@image_comparison(baseline_images = ['cluster_fop_ax_return_double'],
                  **IMG_COMP_KWARGS)
def test_cluster_fop_ax_return_double(mouse_data):
    _, ax = plt.subplots(ncols = 2,
                         nrows = 1,
                         figsize = (4, 2))
    fp.pl.cluster_fop(mouse_data,
                      gate = "Neutrophils",
                      layer = "compensated",
                      cluster_key = "Neutrophils_compensated_leiden",
                      marker = "Ly6G",
                      stat_test = False,
                      show = False,
                      ax = ax[0])
    ax[0].set_title("left plot")
    fp.pl.cluster_fop(mouse_data,
                      gate = "Neutrophils",
                      layer = "compensated",
                      cluster_key = "Neutrophils_compensated_leiden",
                      marker = "CD3",
                      stat_test = False,
                      show = False,
                      ax = ax[1])
    ax[1].set_title("right_plot")


@image_comparison(baseline_images = ['cluster_fop_splitby'],  # noqa
                  **IMG_COMP_KWARGS)
def test_cluster_fop_splitby(mouse_data):
    fp.pl.cluster_fop(mouse_data,
                      gate = "Neutrophils",
                      layer = "compensated",
                      cluster_key = "Neutrophils_compensated_leiden",
                      marker = "Ly6G",
                      splitby = "experiment",
                      stat_test = False,
                      show = False)


@image_comparison(baseline_images = ['cluster_fop_cmap'],
                  **IMG_COMP_KWARGS)
def test_cluster_fop_cmap(mouse_data):
    fp.pl.cluster_fop(mouse_data,
                      gate = "Neutrophils",
                      layer = "compensated",
                      cluster_key = "Neutrophils_compensated_leiden",
                      marker = "Ly6G",
                      stat_test = False,
                      cmap = "Set2",
                      show = False)


def test_cluster_fop_typerror(mouse_data):
    mouse_data = mouse_data.copy()
    fp.tl.fop(mouse_data, groupby = "Neutrophils_compensated_leiden",
              aggregate = True)
    with pytest.raises(TypeError):
        fp.pl.cluster_fop(mouse_data,
                          gate = "Neutrophils",
                          layer = "compensated",
                          cluster_key = "Neutrophils_compensated_leiden",
                          marker = "Ly6G",
                          stat_test = False,
                          cmap = "Set2",
                          show = False)


def test_cluster_fop_dataframe(mouse_data):
    df = fp.pl.cluster_fop(mouse_data,
                           gate = "Neutrophils",
                           layer = "compensated",
                           cluster_key = "Neutrophils_compensated_leiden",
                           marker = "Ly6G",
                           cmap = "Set2",
                           stat_test = False,
                           show = False,
                           return_dataframe = True)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID", "gate", "Neutrophils_compensated_leiden"] +
               mouse_data.uns["metadata"].get_factors() +
               mouse_data.var_names.tolist())
    assert df["gate"].nunique() == 1
    neus = fp.subset_gate(mouse_data, "Neutrophils", copy = True)
    assert isinstance(neus, AnnData)
    s3: pd.DataFrame = neus[
        neus.obs["sample_ID"] == "3"
    ].to_df(layer = "compensated")
    pos = s3 > neus.var["cofactors"].astype(np.float64).tolist()
    pos["Neutrophils_compensated_leiden"] = neus.obs["Neutrophils_compensated_leiden"].astype(np.int8)
    freq_pos = pos.groupby("Neutrophils_compensated_leiden").sum() / pos.groupby("Neutrophils_compensated_leiden").count()
    fop = freq_pos.loc[
        (freq_pos.index == 0),
        "Ly6G"
    ].iloc[0]
    assert df.loc[
        (df["sample_ID"] == "3") &
        (df["Neutrophils_compensated_leiden"] == "0"),
        "Ly6G"
    ].iloc[0] == fop


def test_cluster_fop_dataframe_splitby(mouse_data):
    df = fp.pl.cluster_fop(mouse_data,
                           gate = "Neutrophils",
                           layer = "compensated",
                           cluster_key = "Neutrophils_compensated_leiden",
                           marker = "Ly6G",
                           cmap = "Set2",
                           show = False,
                           stat_test = False,
                           splitby = "experiment",
                           return_dataframe = True)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID", "gate", "Neutrophils_compensated_leiden"] +
               mouse_data.uns["metadata"].get_factors() +
               mouse_data.var_names.tolist())
    assert df["gate"].nunique() == 1
    neus = fp.subset_gate(mouse_data, "Neutrophils", copy = True)
    assert isinstance(neus, AnnData)
    s3: pd.DataFrame = neus[
        neus.obs["sample_ID"] == "3"
    ].to_df(layer = "compensated")
    pos = s3 > neus.var["cofactors"].astype(np.float64).tolist()
    pos["Neutrophils_compensated_leiden"] = neus.obs["Neutrophils_compensated_leiden"].astype(np.int8)
    freq_pos = pos.groupby("Neutrophils_compensated_leiden").sum() / pos.groupby("Neutrophils_compensated_leiden").count()
    fop = freq_pos.loc[
        (freq_pos.index == 0),
        "Ly6G"
    ].iloc[0]
    assert df.loc[
        (df["sample_ID"] == "3") &
        (df["Neutrophils_compensated_leiden"] == "0"),
        "Ly6G"
    ].iloc[0] == fop


# cofactor distribution

@image_comparison(baseline_images = ['cofac_distrib'],
                  **IMG_COMP_KWARGS)
def test_cofactor_distribution_plot(mouse_data):
    fp.pl.cofactor_distribution(mouse_data,
                                marker = "Ly6G",
                                groupby = "sex",
                                show = False)


@image_comparison(baseline_images = ['cofac_distrib_barplot'],
                  **IMG_COMP_KWARGS)
def test_cofactor_distribution_barplot(mouse_data):
    fp.pl.cofactor_distribution(mouse_data,
                                marker = "Ly6G",
                                groupby = "sample_ID",
                                show = False)


@image_comparison(baseline_images = ['cofac_distrib_sex_figsize'],
                  **IMG_COMP_KWARGS)
def test_cofactor_distribution_plot_figsize(mouse_data):
    fp.pl.cofactor_distribution(mouse_data,
                                marker = "Ly6G",
                                groupby = "sex",
                                figsize = (1, 1),
                                show = False)


@image_comparison(baseline_images = ['cofac_distrib_sex_experiment'],  # noqa
                  **IMG_COMP_KWARGS)
def test_cofactor_distribution_splitby(mouse_data):
    fp.pl.cofactor_distribution(mouse_data,
                                marker = "Ly6G",
                                groupby = "sex",
                                splitby = "experiment",
                                figsize = (1, 1),
                                show = False)


@image_comparison(baseline_images = ['cofac_distrib_sex_experiment_set2'],
                  **IMG_COMP_KWARGS)
def test_cofactor_distribution_splitby_cmap(mouse_data):
    fp.pl.cofactor_distribution(mouse_data,
                                marker = "Ly6G",
                                groupby = "sex",
                                splitby = "experiment",
                                cmap = "Set2",
                                figsize = (1, 1),
                                show = False)


@image_comparison(baseline_images = ['cofac_distrib_ax_return'],
                  **IMG_COMP_KWARGS)
def test_cofactor_distribution_ax_return(mouse_data):
    _, ax = plt.subplots(ncols = 1,
                         nrows = 1,
                         figsize = (3, 3))
    fp.pl.cofactor_distribution(mouse_data,
                                marker = "Ly6G",
                                groupby = "sex",
                                splitby = "experiment",
                                cmap = "Set2",
                                ax = ax,
                                show = False)
    ax.set_title("test_plot")


@image_comparison(baseline_images = ['cofac_distrib_ax_return_double'],
                  **IMG_COMP_KWARGS)
def test_cofactor_distribution_ax_return_double(mouse_data):
    _, ax = plt.subplots(ncols = 2,
                         nrows = 1,
                         figsize = (4, 2))
    fp.pl.cofactor_distribution(mouse_data,
                                marker = "Ly6G",
                                groupby = "sex",
                                splitby = "experiment",
                                cmap = "Set2",
                                ax = ax[0],
                                show = False)
    ax[0].set_title("left plot")
    fp.pl.cofactor_distribution(mouse_data,
                                marker = "CD3",
                                groupby = "sex",
                                splitby = "experiment",
                                cmap = "Set2",
                                ax = ax[1],
                                show = False)
    ax[1].set_title("right plot")


def test_cofactor_distribution_dataframe(mouse_data):
    df = fp.pl.cofactor_distribution(mouse_data,
                                     marker = "Ly6G",
                                     groupby = "sex",
                                     show = False,
                                     return_dataframe = True)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID", "Ly6G"] +
               mouse_data.uns["metadata"].get_factors())
    raw_cof = mouse_data.uns["raw_cofactors"].reset_index(names = "file_name")
    raw_cof = raw_cof.sort_values("file_name")

    df = df.sort_values("file_name")
    assert all(df["Ly6G"].to_numpy() == raw_cof["Ly6G"].to_numpy())


def test_cofactor_distribution_dataframe_splitby(mouse_data):
    df = fp.pl.cofactor_distribution(mouse_data,
                                     marker = "Ly6G",
                                     groupby = "sex",
                                     splitby = "experiment",
                                     show = False,
                                     return_dataframe = True)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID", "Ly6G"] +
               mouse_data.uns["metadata"].get_factors())
    raw_cof = mouse_data.uns["raw_cofactors"].reset_index(names = "file_name")
    raw_cof = raw_cof.sort_values("file_name")

    df = df.sort_values("file_name")
    assert all(df["Ly6G"].to_numpy() == raw_cof["Ly6G"].to_numpy())


# cluster frequency


@image_comparison(baseline_images = ['clus_freq'],
                  **IMG_COMP_KWARGS)
def test_cluster_frequency_plot(mouse_data):
    fp.pl.cluster_frequency(mouse_data,
                            gate = "Neutrophils",
                            cluster_key = "Neutrophils_compensated_leiden",
                            cluster = "0",
                            groupby = "sex",
                            show = False)


@image_comparison(baseline_images = ['clus_freq_barplot'],
                  **IMG_COMP_KWARGS)
def test_cluster_frequency_barplot(mouse_data):
    fp.pl.cluster_frequency(mouse_data,
                            gate = "Neutrophils",
                            cluster_key = "Neutrophils_compensated_leiden",
                            cluster = "0",
                            groupby = "sample_ID",
                            show = False)


@image_comparison(baseline_images = ['clus_freq_sex_figsize'],
                  **IMG_COMP_KWARGS)
def test_cluster_frequency_plot_figsize(mouse_data):
    fp.pl.cluster_frequency(mouse_data,
                            gate = "Neutrophils",
                            cluster_key = "Neutrophils_compensated_leiden",
                            cluster = "0",
                            groupby = "sex",
                            figsize = (2, 2),
                            show = False)


@image_comparison(baseline_images = ['clus_freq_sex_experiment'],  # noqa
                  **IMG_COMP_KWARGS)
def test_cluster_frequency_splitby(mouse_data):
    fp.pl.cluster_frequency(mouse_data,
                            gate = "Neutrophils",
                            cluster_key = "Neutrophils_compensated_leiden",
                            cluster = "0",
                            groupby = "sex",
                            splitby = "experiment",
                            show = False)


@image_comparison(baseline_images = ['clus_freq_sex_experiment_set2'],
                  **IMG_COMP_KWARGS)
def test_cluster_frequency_splitby_cmap(mouse_data):
    fp.pl.cluster_frequency(mouse_data,
                            gate = "Neutrophils",
                            cluster_key = "Neutrophils_compensated_leiden",
                            cluster = "0",
                            groupby = "sex",
                            splitby = "experiment",
                            cmap = "Set2",
                            show = False)


@image_comparison(baseline_images = ['clus_freq_ax_return'],
                  **IMG_COMP_KWARGS)
def test_cluster_frequency_distribution_ax_return(mouse_data):
    _, ax = plt.subplots(ncols = 1,
                         nrows = 1,
                         figsize = (3, 3))
    fp.pl.cluster_frequency(mouse_data,
                            gate = "Neutrophils",
                            cluster_key = "Neutrophils_compensated_leiden",
                            cluster = "0",
                            groupby = "sex",
                            splitby = "experiment",
                            ax = ax,
                            show = False)
    ax.set_title("test_plot")


@image_comparison(baseline_images = ['clus_freq_ax_return_double'],
                  **IMG_COMP_KWARGS)
def test_cluster_frequency_ax_return_double(mouse_data):
    _, ax = plt.subplots(ncols = 2,
                         nrows = 1,
                         figsize = (4, 2))
    fp.pl.cluster_frequency(mouse_data,
                            gate = "Neutrophils",
                            cluster_key = "Neutrophils_compensated_leiden",
                            cluster = "0",
                            groupby = "sex",
                            splitby = "experiment",
                            ax = ax[0],
                            show = False)
    ax[0].set_title("left plot")
    fp.pl.cluster_frequency(mouse_data,
                            gate = "Neutrophils",
                            cluster_key = "Neutrophils_compensated_leiden",
                            cluster = "0",
                            groupby = "sex",
                            splitby = "experiment",
                            ax = ax[1],
                            show = False)
    ax[1].set_title("right plot")


def test_cluster_frequency_dataframe(mouse_data):
    df = fp.pl.cluster_frequency(mouse_data,
                                 gate = "Neutrophils",
                                 cluster_key = "Neutrophils_compensated_leiden",
                                 cluster = "0",
                                 groupby = "sex",
                                 splitby = "experiment",
                                 show = False,
                                 return_dataframe = True)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID", "sex", "count"])
    neus: AnnData = fp.subset_gate(mouse_data, "Neutrophils", copy = True)
    s3: pd.DataFrame = neus[
        neus.obs["sample_ID"] == "3"
    ]
    cluster_count = s3.obs.groupby(["sample_ID", "Neutrophils_compensated_leiden"]).count().reset_index()
    spec_count = cluster_count.loc[
        (cluster_count["sample_ID"] == "3") &
        (cluster_count["Neutrophils_compensated_leiden"] == "0"),
        "file_name"
    ].iloc[0]
    assert df.loc[
        (df["sample_ID"] == "3") &
        (df["Neutrophils_compensated_leiden"] == "0"),
        "count"
    ].iloc[0] == spec_count


def test_cluster_frequency_dataframe_splitby(mouse_data):
    df = fp.pl.cluster_frequency(mouse_data,
                                 gate = "Neutrophils",
                                 cluster_key = "Neutrophils_compensated_leiden",
                                 cluster = "0",
                                 groupby = "sex",
                                 splitby = "experiment",
                                 show = False,
                                 return_dataframe = True)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID", "sex", "count", "experiment"])
    neus: AnnData = fp.subset_gate(mouse_data, "Neutrophils", copy = True)
    s3: pd.DataFrame = neus[
        neus.obs["sample_ID"] == "3"
    ]
    cluster_count = s3.obs.groupby(["sample_ID", "Neutrophils_compensated_leiden"]).count().reset_index()
    spec_count = cluster_count.loc[
        (cluster_count["sample_ID"] == "3") &
        (cluster_count["Neutrophils_compensated_leiden"] == "0"),
        "file_name"
    ].iloc[0]
    assert df.loc[
        (df["sample_ID"] == "3") &
        (df["Neutrophils_compensated_leiden"] == "0"),
        "count"
    ].iloc[0] == spec_count


# cluster abundance


@image_comparison(baseline_images = ['clus_abund'],
                  **IMG_COMP_KWARGS)
def test_cluster_abundance_plot(mouse_data):
    fp.pl.cluster_abundance(mouse_data,
                            cluster_key = "Neutrophils_compensated_leiden",
                            groupby = "sex",
                            show = False)


@image_comparison(baseline_images = ['clus_abund_sex_figsize'],
                  **IMG_COMP_KWARGS)
def test_cluster_abundance_plot_figsize(mouse_data):
    fp.pl.cluster_abundance(mouse_data,
                            cluster_key = "Neutrophils_compensated_leiden",
                            groupby = "sex",
                            figsize = (2, 2),
                            show = False)



@image_comparison(baseline_images = ['clus_abund_ax_return'],
                  **IMG_COMP_KWARGS)
def test_cluster_abundance_distribution_ax_return(mouse_data):
    _, ax = plt.subplots(ncols = 1,
                         nrows = 1,
                         figsize = (3, 3))
    fp.pl.cluster_abundance(mouse_data,
                            cluster_key = "Neutrophils_compensated_leiden",
                            groupby = "sex",
                            ax = ax,
                            show = False)
    ax.set_title("test_plot")


@image_comparison(baseline_images = ['clus_abund_ax_return_double'],
                  **IMG_COMP_KWARGS)
def test_cluster_abundance_ax_return_double(mouse_data):
    _, ax = plt.subplots(ncols = 2,
                         nrows = 1,
                         figsize = (4, 2))
    fp.pl.cluster_abundance(mouse_data,
                            cluster_key = "Neutrophils_compensated_leiden",
                            groupby = "sex",
                            ax = ax[0],
                            show = False)
    ax[0].set_title("left plot")
    fp.pl.cluster_abundance(mouse_data,
                            cluster_key = "Neutrophils_compensated_leiden",
                            groupby = "sex",
                            ax = ax[1],
                            show = False)
    ax[1].set_title("right plot")


def test_cluster_abundance_dataframe(mouse_data):
    df = fp.pl.cluster_abundance(mouse_data,
                                 cluster_key = "Neutrophils_compensated_leiden",
                                 groupby = "sex",
                                 show = False,
                                 return_dataframe = True)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in mouse_data.uns["metadata"].dataframe["sex"].unique().tolist())
    neus: AnnData = fp.subset_gate(mouse_data, "Neutrophils", copy = True)
    cluster_count = neus.obs.groupby(["sex", "Neutrophils_compensated_leiden"]).count() / neus.obs.groupby(["Neutrophils_compensated_leiden"]).count()
    cluster_count = cluster_count.drop("sex", axis = 1)
    cluster_count = cluster_count.reset_index()
    spec_count = cluster_count.loc[
        (cluster_count["Neutrophils_compensated_leiden"] == "0") &
        (cluster_count["sex"] == "f"),
        "sample_ID"
    ].iloc[0]
    print(spec_count)
    assert df.loc[
        (df.index == "0"),
        "f"
    ].iloc[0] == spec_count


# metadata

@image_comparison(baseline_images = ['metadata'],
                  **IMG_COMP_KWARGS)
def test_metadata(mouse_data):
    fp.pl.metadata(mouse_data,
                   marker = "age",
                   groupby = "sex",
                   show = False)


@image_comparison(baseline_images = ['metadata_figsize'],
                  **IMG_COMP_KWARGS)
def test_metadata_figsize(mouse_data):
    fp.pl.metadata(mouse_data,
                   marker = "age",
                   groupby = "sex",
                   figsize = (1, 1),
                   show = False)


@image_comparison(baseline_images = ['metadata_splitby'],  # noqa
                  **IMG_COMP_KWARGS)
def test_metadata_splitby(mouse_data):
    fp.pl.metadata(mouse_data,
                   marker = "age",
                   groupby = "sex",
                   splitby = "experiment",
                   figsize = (1, 1),
                   show = False)


@image_comparison(baseline_images = ['metadata_splitby_cmap'],
                  **IMG_COMP_KWARGS)
def test_metadata_splitby_cmap(mouse_data):
    fp.pl.metadata(mouse_data,
                   marker = "age",
                   groupby = "sex",
                   splitby = "experiment",
                   cmap = "Set2",
                   figsize = (1, 1),
                   show = False)


@image_comparison(baseline_images = ['metadata_ax_return'],
                  **IMG_COMP_KWARGS)
def test_metadata_ax_return(mouse_data):
    _, ax = plt.subplots(ncols = 1,
                         nrows = 1,
                         figsize = (3, 4))
    fp.pl.metadata(mouse_data,
                   marker = "age",
                   groupby = "sex",
                   splitby = "experiment",
                   cmap = "Set2",
                   ax = ax,
                   show = False)
    ax.set_title("test_plot")


@image_comparison(baseline_images = ['metadata_ax_return_double'],
                  **IMG_COMP_KWARGS)
def test_metadata_ax_return_double(mouse_data):
    _, ax = plt.subplots(ncols = 2,
                         nrows = 1,
                         figsize = (4, 2))
    fp.pl.metadata(mouse_data,
                   marker = "age",
                   groupby = "sex",
                   ax = ax[0],
                   show = False)
    ax[0].set_title("left plot")
    fp.pl.metadata(mouse_data,
                   marker = "age",
                   groupby = "experiment",
                   ax = ax[1],
                   show = False)
    ax[1].set_title("right plot")


def test_metadata_dataframe(mouse_data):
    df = fp.pl.metadata(mouse_data,
                        marker = "age",
                        groupby = "experiment",
                        show = False,
                        return_dataframe = True)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in mouse_data.uns["metadata"].get_factors())


def test_metadata_dataframe_splitby(mouse_data):
    df = fp.pl.metadata(mouse_data,
                        marker = "age",
                        groupby = "experiment",
                        splitby = "sex",
                        show = False,
                        return_dataframe = True)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in mouse_data.uns["metadata"].get_factors())


# gate frequency


@image_comparison(baseline_images = ['gate_freq'],
                  **IMG_COMP_KWARGS)
def test_gate_frequency_plot(mouse_data):
    fp.pl.gate_frequency(mouse_data,
                         gate = "Neutrophils",
                         freq_of = "parent",
                         groupby = "experiment",
                         stat_test = False,
                         show = False)


@image_comparison(baseline_images = ['gate_freq_sample_ID'],
                  **IMG_COMP_KWARGS)
def test_gate_frequency_barplot(mouse_data):
    fp.pl.gate_frequency(mouse_data,
                         gate = "Neutrophils",
                         freq_of = "parent",
                         groupby = "sample_ID",
                         show = False)


@image_comparison(baseline_images = ['gate_freq_figsize'],
                  **IMG_COMP_KWARGS)
def test_gate_frequency_figsize(mouse_data):
    fp.pl.gate_frequency(mouse_data,
                         gate = "Neutrophils",
                         freq_of = "parent",
                         groupby = "experiment",
                         figsize = (2, 2),
                         show = False)


@image_comparison(baseline_images = ['gate_freq_splitby'],  # noqa
                  **IMG_COMP_KWARGS)
def test_gate_frequency_splitby(mouse_data):
    fp.pl.gate_frequency(mouse_data,
                         gate = "Neutrophils",
                         freq_of = "parent",
                         groupby = "experiment",
                         splitby = "age",
                         figsize = (2, 2),
                         show = False)


@image_comparison(baseline_images = ['gate_freq_splitby_cmap'],
                  **IMG_COMP_KWARGS)
def test_gate_frequency_splitby_cmap(mouse_data):
    fp.pl.gate_frequency(mouse_data,
                         gate = "Neutrophils",
                         freq_of = "parent",
                         groupby = "experiment",
                         splitby = "age",
                         cmap = "Set2",
                         figsize = (2, 2),
                         show = False)


@image_comparison(baseline_images = ['gate_freq_ax_return'],
                  **IMG_COMP_KWARGS)
def test_gate_frequency_ax_return(mouse_data):
    _, ax = plt.subplots(ncols = 1,
                         nrows = 1,
                         figsize = (3, 4))
    fp.pl.gate_frequency(mouse_data,
                         gate = "Neutrophils",
                         freq_of = "parent",
                         groupby = "experiment",
                         splitby = "age",
                         ax = ax,
                         show = False)
    ax.set_title("test_plot")


@image_comparison(baseline_images = ['gate_frequency_ax_return_double'],
                  **IMG_COMP_KWARGS)
def test_gate_frequency_ax_return_double(mouse_data):
    _, ax = plt.subplots(ncols = 2,
                         nrows = 1,
                         figsize = (4, 2))
    fp.pl.gate_frequency(mouse_data,
                         gate = "Neutrophils",
                         freq_of = "parent",
                         groupby = "experiment",
                         ax = ax[0],
                         show = False)
    ax[0].set_title("left plot")
    fp.pl.gate_frequency(mouse_data,
                         gate = "Neutrophils",
                         freq_of = "parent",
                         groupby = "experiment",
                         ax = ax[1],
                         show = False)
    ax[1].set_title("right plot")


def test_gate_frequency_dataframe(mouse_data):
    df = fp.pl.gate_frequency(mouse_data,
                              gate = "Neutrophils",
                              freq_of = "parent",
                              groupby = "experiment",
                              return_dataframe = True,
                              show = False)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID", "gate", "experiment", "freq_of", "freq"])
    assert df["gate"].nunique() == 1
    assert df["freq_of"].nunique() == 1
    neus = mouse_data.copy()
    assert isinstance(neus, AnnData)
    fp.convert_gate_to_obs(neus, "Neutrophils", key_added = "neus")
    fp.convert_gate_to_obs(neus,
                           fp._utils._find_parent_gate(
                               fp._utils._find_gate_path_of_gate(
                                   neus, "Neutrophils"
                               )
                           ), key_added = "parent")
    neus = neus[neus.obs["sample_ID"] == "3"]
    neus = neus[neus.obs["parent"] == "parent"]
    neu_freq = neus.obs["neus"].value_counts() / neus.obs.shape[0]
    freq = neu_freq.loc[neu_freq.index == "neus"].iloc[0]
    assert df.loc[
        df["sample_ID"] == "3",
        "freq"
    ].iloc[0] == freq


def test_gate_frequency_dataframe(mouse_data):
    df = fp.pl.gate_frequency(mouse_data,
                              gate = "Neutrophils",
                              freq_of = "parent",
                              groupby = "sex",
                              return_dataframe = True,
                              show = False)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID", "gate", "sex", "freq_of", "freq"])
    assert df["gate"].nunique() == 1
    assert df["freq_of"].nunique() == 1
    neus = mouse_data.copy()
    assert isinstance(neus, AnnData)
    fp.convert_gate_to_obs(neus, "Neutrophils", key_added = "neus")
    fp.convert_gate_to_obs(neus,
                           fp._utils._find_parent_gate(
                               fp._utils._find_gate_path_of_gate(
                                   neus, "Neutrophils"
                               )
                           ), key_added = "parent")
    neus = neus[neus.obs["sample_ID"] == "3"]
    neus = neus[neus.obs["parent"] == "parent"]
    neu_freq = neus.obs["neus"].value_counts() / neus.obs.shape[0]
    freq = neu_freq.loc[neu_freq.index == "neus"].iloc[0]
    assert df.loc[
        df["sample_ID"] == "3",
        "freq"
    ].iloc[0] == freq



def test_gate_frequency_dataframe_splitby(mouse_data):
    df = fp.pl.gate_frequency(mouse_data,
                              gate = "Neutrophils",
                              freq_of = "parent",
                              groupby = "experiment",
                              splitby = "sex",
                              return_dataframe = True,
                              show = False)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID", "gate", "experiment", "freq_of", "freq", "sex"])
    assert df["gate"].nunique() == 1
    assert df["freq_of"].nunique() == 1
    neus = mouse_data.copy()
    assert isinstance(neus, AnnData)
    fp.convert_gate_to_obs(neus, "Neutrophils", key_added = "neus")
    fp.convert_gate_to_obs(neus,
                           fp._utils._find_parent_gate(
                               fp._utils._find_gate_path_of_gate(
                                   neus, "Neutrophils"
                               )
                           ), key_added = "parent")
    neus = neus[neus.obs["sample_ID"] == "3"]
    neus = neus[neus.obs["parent"] == "parent"]
    neu_freq = neus.obs["neus"].value_counts() / neus.obs.shape[0]
    freq = neu_freq.loc[neu_freq.index == "neus"].iloc[0]
    assert df.loc[
        df["sample_ID"] == "3",
        "freq"
    ].iloc[0] == freq


# cell counts


@image_comparison(baseline_images = ['cell_count'],
                  **IMG_COMP_KWARGS)
def test_cell_counts_plot(mouse_data):
    fp.pl.cell_counts(mouse_data,
                      gate = "Neutrophils",
                      groupby = "sex",
                      show = False)


@image_comparison(baseline_images = ['cell_count_sample_ID'],
                  **IMG_COMP_KWARGS)
def test_cell_count_barplot(mouse_data):
    fp.pl.cell_counts(mouse_data,
                      gate = "Neutrophils",
                      groupby = "sample_ID",
                      show = False)


@image_comparison(baseline_images = ['cell_count_figsize'],
                  **IMG_COMP_KWARGS)
def test_cell_count_figsize(mouse_data):
    fp.pl.cell_counts(mouse_data,
                      gate = "Neutrophils",
                      groupby = "sample_ID",
                      figsize = (1, 1),
                      show = False)


@image_comparison(baseline_images = ['cell_count_splitby'],  # noqa
                  **IMG_COMP_KWARGS)
def test_cell_count_splitby(mouse_data):
    fp.pl.cell_counts(mouse_data,
                      gate = "Neutrophils",
                      groupby = "sex",
                      splitby = "experiment",
                      show = False)


@image_comparison(baseline_images = ['cell_count_splitby_cmap'],
                  **IMG_COMP_KWARGS)
def test_cell_count_splitby_cmap(mouse_data):
    fp.pl.cell_counts(mouse_data,
                      gate = "Neutrophils",
                      groupby = "sex",
                      splitby = "experiment",
                      cmap = "Set2",
                      show = False)


@image_comparison(baseline_images = ['cell_count_ax_return'],
                  **IMG_COMP_KWARGS)
def test_cell_count_ax_return(mouse_data):
    _, ax = plt.subplots(ncols = 1,
                         nrows = 1,
                         figsize = (3, 4))
    fp.pl.cell_counts(mouse_data,
                      gate = "Neutrophils",
                      groupby = "sex",
                      splitby = "experiment",
                      ax = ax,
                      show = False)
    ax.set_title("test_plot")


@image_comparison(baseline_images = ['cell_count_ax_return_double'],
                  **IMG_COMP_KWARGS)
def test_cell_count_ax_return_double(mouse_data):
    _, ax = plt.subplots(ncols = 2,
                         nrows = 1,
                         figsize = (4, 2))
    fp.pl.cell_counts(mouse_data,
                      gate = "Neutrophils",
                      groupby = "sex",
                      splitby = "experiment",
                      ax = ax[0],
                      show = False)
    ax[0].set_title("left plot")
    fp.pl.cell_counts(mouse_data,
                      gate = "Neutrophils",
                      groupby = "experiment",
                      splitby = "sex",
                      ax = ax[1],
                      show = False)
    ax[1].set_title("right plot")


def test_cell_counts_dataframe(mouse_data):
    df = fp.pl.cell_counts(mouse_data,
                           gate = "Neutrophils",
                           groupby = "experiment",
                           return_dataframe = True,
                           show = False)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID", "counts", "experiment"])
    neus = fp.subset_gate(mouse_data, "Neutrophils", copy = True)
    freq = neus[neus.obs["sample_ID"] == "3"].shape[0]
    assert isinstance(neus, AnnData)
    assert df.loc[
        df["sample_ID"] == "3",
        "counts"
    ].iloc[0] == freq


def test_cell_counts_dataframe_splitby(mouse_data):
    df = fp.pl.cell_counts(mouse_data,
                           gate = "Neutrophils",
                           groupby = "experiment",
                           splitby = "sex",
                           return_dataframe = True,
                           show = False)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID", "counts", "experiment", "sex"])
    neus = fp.subset_gate(mouse_data, "Neutrophils", copy = True)
    freq = neus[neus.obs["sample_ID"] == "3"].shape[0]
    assert isinstance(neus, AnnData)
    assert df.loc[
        df["sample_ID"] == "3",
        "counts"
    ].iloc[0] == freq

