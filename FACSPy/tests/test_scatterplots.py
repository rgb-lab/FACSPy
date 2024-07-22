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
    "savefig_kwarg": {"bbox_inches": "tight"},
    "tol": 2
}

# biax_plot

@image_comparison(baseline_images = ['biax'],
                  **IMG_COMP_KWARGS)
def test_biax_plot(mouse_data):
    mouse_data = mouse_data.copy()
    fp.pl.biax(mouse_data,
               gate = "CD45+",
               layer = "compensated",
               x_channel = "Ly6G",
               y_channel = "SSC-A",
               x_scale = "biex",
               y_scale = "linear",
               show = False)


@image_comparison(baseline_images = ['biax_color'],
                  **IMG_COMP_KWARGS)
def test_biax_color(mouse_data):
    mouse_data = mouse_data.copy()
    fp.pl.biax(mouse_data,
               gate = "CD45+",
               layer = "compensated",
               x_channel = "Ly6G",
               y_channel = "SSC-A",
               x_scale = "biex",
               y_scale = "linear",
               color_scale = "biex",
               color = "Ly6C",
               show = False)


@image_comparison(baseline_images = ['biax_figsize'],
                  **IMG_COMP_KWARGS)
def test_biax_figsize(mouse_data):
    mouse_data = mouse_data.copy()
    fp.pl.biax(mouse_data,
               gate = "CD45+",
               layer = "compensated",
               x_channel = "Ly6G",
               y_channel = "SSC-A",
               x_scale = "biex",
               y_scale = "linear",
               figsize = (2, 2),
               show = False)


@image_comparison(baseline_images = ['biax_ax_return'],
                  **IMG_COMP_KWARGS)
def test_biax_return(mouse_data):
    mouse_data = mouse_data.copy()
    _, ax = plt.subplots(ncols = 1,
                         nrows = 1,
                         figsize = (3, 4))
    fp.pl.biax(mouse_data,
               gate = "CD45+",
               layer = "compensated",
               x_channel = "Ly6G",
               y_channel = "SSC-A",
               x_scale = "biex",
               y_scale = "linear",
               ax = ax,
               show = False)
    ax.set_title("test_plot")


@image_comparison(baseline_images = ['biax_ax_return_double'],
                  **IMG_COMP_KWARGS)
def test_biax_ax_return_double(mouse_data):
    mouse_data = mouse_data.copy()
    _, ax = plt.subplots(ncols = 2,
                         nrows = 1,
                         figsize = (4, 2))
    fp.pl.biax(mouse_data,
               gate = "CD45+",
               layer = "compensated",
               x_channel = "Ly6G",
               y_channel = "SSC-A",
               x_scale = "biex",
               y_scale = "linear",
               ax = ax[0],
               show = False)
    ax[0].set_title("left plot")
    fp.pl.biax(mouse_data,
               gate = "CD45+",
               layer = "compensated",
               x_channel = "Ly6G",
               y_channel = "SSC-A",
               x_scale = "biex",
               y_scale = "linear",
               ax = ax[1],
               show = False)
    ax[1].set_title("right plot")


def test_biax_dataframe(mouse_data):
    mouse_data = mouse_data.copy()
    df = fp.pl.biax(mouse_data,
                    gate = "CD45+",
                    layer = "compensated",
                    x_channel = "Ly6G",
                    y_channel = "SSC-A",
                    sample_identifier = "3",
                    x_scale = "biex",
                    y_scale = "linear",
                    return_dataframe = True,
                    show = False)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID"] +
               mouse_data.uns["metadata"].get_factors() +
               mouse_data.var_names.tolist())
    neus = fp.subset_gate(mouse_data, "CD45+", copy = True)
    assert isinstance(neus, AnnData)
    s3: pd.DataFrame = neus[
        neus.obs["sample_ID"] == "3"
    ].to_df(layer = "compensated")
    assert np.array_equal(
        s3.values,
        df[s3.columns].values
    )


def test_biax_dataframe_color(mouse_data):
    mouse_data = mouse_data.copy()
    df = fp.pl.biax(mouse_data,
                    gate = "CD45+",
                    layer = "compensated",
                    x_channel = "Ly6G",
                    y_channel = "SSC-A",
                    sample_identifier = "3",
                    color = "Ly6C",
                    x_scale = "biex",
                    y_scale = "linear",
                    return_dataframe = True,
                    show = False)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID"] +
               mouse_data.uns["metadata"].get_factors() +
               mouse_data.var_names.tolist())
    neus = fp.subset_gate(mouse_data, "CD45+", copy = True)
    assert isinstance(neus, AnnData)
    s3: pd.DataFrame = neus[
        neus.obs["sample_ID"] == "3"
    ].to_df(layer = "compensated")
    assert np.array_equal(
        s3.values,
        df[s3.columns].values
    )


# samplewise DR


@image_comparison(baseline_images = ['pca_samplewise'],
                  **IMG_COMP_KWARGS)
def test_pca_samplewise(mouse_data):
    mouse_data = mouse_data.copy()
    fp.pl.pca_samplewise(mouse_data,
                         gate = "Neutrophils",
                         layer = "compensated",
                         color = "experiment",
                         show = False)


@image_comparison(baseline_images = ['pca_samplewise_figsize'],
                  **IMG_COMP_KWARGS)
def test_pca_samplewise_figsize(mouse_data):
    mouse_data = mouse_data.copy()
    fp.pl.pca_samplewise(mouse_data,
                         gate = "Neutrophils",
                         layer = "compensated",
                         color = "experiment",
                         figsize = (2, 2),
                         show = False)


@image_comparison(baseline_images = ['pca_samplewise_ax_return'],
                  **IMG_COMP_KWARGS)
def test_pca_samplewise_return(mouse_data):
    mouse_data = mouse_data.copy()
    _, ax = plt.subplots(ncols = 1,
                         nrows = 1,
                         figsize = (3, 4))
    fp.pl.pca_samplewise(mouse_data,
                         gate = "Neutrophils",
                         layer = "compensated",
                         color = "sex",
                         ax = ax,
                         show = False)
    ax.set_title("test_plot")


@image_comparison(baseline_images = ['pca_samplewise_ax_return_double'],
                  **IMG_COMP_KWARGS)
def test_pca_samplewise_ax_return_double(mouse_data):
    mouse_data = mouse_data.copy()
    _, ax = plt.subplots(ncols = 2,
                         nrows = 1,
                         figsize = (4, 2))
    fp.pl.pca_samplewise(mouse_data,
                         gate = "Neutrophils",
                         layer = "compensated",
                         color = "sex",
                         ax = ax[0],
                         show = False)
    ax[0].set_title("left plot")
    fp.pl.pca_samplewise(mouse_data,
                         gate = "Neutrophils",
                         layer = "compensated",
                         color = "sex",
                         ax = ax[1],
                         show = False)
    ax[1].set_title("right plot")


def test_pca_samplewise_dataframe(mouse_data):
    mouse_data = mouse_data.copy()
    df = fp.pl.pca_samplewise(mouse_data,
                              gate = "Neutrophils",
                              layer = "compensated",
                              return_dataframe = True,
                              show = False)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID", "gate", "PCA1", "PCA2", "PCA3"] +
               mouse_data.uns["metadata"].get_factors() +
               mouse_data.var_names.tolist())
    assert df["gate"].nunique() == 1
    neus = fp.subset_gate(mouse_data, "Neutrophils", copy = True)
    assert isinstance(neus, AnnData)
    mfi_frame = neus.uns["mfi_sample_ID_compensated"].reset_index()
    pc_coords = mfi_frame.loc[
        (mfi_frame["sample_ID"] == "3") &
        (mfi_frame["gate"] == "root/cells/singlets/live/CD45+/Neutrophils"),
        ["PCA1", "PCA2", "PCA3"]].to_numpy()
    assert np.array_equal(
        df.loc[df["sample_ID"] == "3", ["PCA1", "PCA2", "PCA3"]].to_numpy(),
        pc_coords
    )


def test_pca_samplewise_dataframe_color(mouse_data):
    mouse_data = mouse_data.copy()
    df = fp.pl.pca_samplewise(mouse_data,
                              gate = "Neutrophils",
                              layer = "compensated",
                              color = "sex",
                              return_dataframe = True,
                              show = False)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID", "gate", "PCA1", "PCA2", "PCA3"] +
               mouse_data.uns["metadata"].get_factors() +
               mouse_data.var_names.tolist())
    assert df["gate"].nunique() == 1
    neus = fp.subset_gate(mouse_data, "Neutrophils", copy = True)
    assert isinstance(neus, AnnData)
    mfi_frame = neus.uns["mfi_sample_ID_compensated"].reset_index()
    pc_coords = mfi_frame.loc[
        (mfi_frame["sample_ID"] == "3") &
        (mfi_frame["gate"] == "root/cells/singlets/live/CD45+/Neutrophils"),
        ["PCA1", "PCA2", "PCA3"]].to_numpy()
    assert np.array_equal(
        df.loc[df["sample_ID"] == "3", ["PCA1", "PCA2", "PCA3"]].to_numpy(),
        pc_coords
    )


@image_comparison(baseline_images = ['pca'],
                  **IMG_COMP_KWARGS)
def test_pca(mouse_data):
    mouse_data = mouse_data.copy()
    fp.pl.pca(mouse_data,
              gate = "Neutrophils",
              layer = "compensated",
              color = "experiment",
              show = False)


@image_comparison(baseline_images = ['pca_figsize'],
                  **IMG_COMP_KWARGS)
def test_pca_figsize(mouse_data):
    mouse_data = mouse_data.copy()
    fp.pl.pca(mouse_data,
              gate = "Neutrophils",
              layer = "compensated",
              color = "experiment",
              figsize = (2, 2),
              show = False)


@image_comparison(baseline_images = ['pca_ax_return'],
                  **IMG_COMP_KWARGS)
def test_pca_return(mouse_data):
    mouse_data = mouse_data.copy()
    _, ax = plt.subplots(ncols = 1,
                         nrows = 1,
                         figsize = (3, 4))
    fp.pl.pca(mouse_data,
              gate = "Neutrophils",
              layer = "compensated",
              color = "sex",
              ax = ax,
              show = False)
    ax.set_title("test_plot")


@image_comparison(baseline_images = ['pca_ax_return_double'],
                  **IMG_COMP_KWARGS)
def test_pca_ax_return_double(mouse_data):
    mouse_data = mouse_data.copy()
    _, ax = plt.subplots(ncols = 2,
                         nrows = 1,
                         figsize = (4, 2))
    fp.pl.pca(mouse_data,
              gate = "Neutrophils",
              layer = "compensated",
              color = "sex",
              ax = ax[0],
              show = False)
    ax[0].set_title("left plot")
    fp.pl.pca(mouse_data,
              gate = "Neutrophils",
              layer = "compensated",
              color = "sex",
              ax = ax[1],
              show = False)
    ax[1].set_title("right plot")


def test_pca_dataframe(mouse_data):
    mouse_data = mouse_data.copy()
    df = fp.pl.pca(mouse_data,
                   gate = "Neutrophils",
                   layer = "compensated",
                   return_dataframe = True,
                   show = False)
    assert isinstance(df, pd.DataFrame)
    assert all(k in df.columns
               for k in ["sample_ID", "PCA1", "PCA2"] +
               mouse_data.uns["metadata"].get_factors() +
               mouse_data.var_names.tolist())
    neus = fp.subset_gate(mouse_data, "Neutrophils", copy = True)
    assert isinstance(neus, AnnData)
    df = df.dropna(how = "any")
    pca_coords = neus.obsm["X_pca_Neutrophils_compensated"][:, :2]
    df_coords = df[["PCA1", "PCA2"]].to_numpy()
    assert np.array_equal(
        pca_coords,
        df_coords
    )


# cofactor plots


@image_comparison(baseline_images = ['transformation'],
                  **IMG_COMP_KWARGS)
def test_cofactor_plot(mouse_data):
    mouse_data = mouse_data.copy()
    fp.pl.transformation_plot(mouse_data,
                              gate = "Neutrophils",
                              sample_identifier = "3",
                              marker = "Ly6G",
                              show = False)


@image_comparison(baseline_images = ['transformation_figsize'],
                  **IMG_COMP_KWARGS)
def test_cofactor_plot_figsize(mouse_data):
    mouse_data = mouse_data.copy()
    fp.pl.transformation_plot(mouse_data,
                              gate = "Neutrophils",
                              marker = "Ly6G",
                              sample_identifier = "3",
                              figsize = (3, 1),
                              show = False)
