import pytest

from anndata import AnnData
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import FACSPy as fp
from FACSPy.tools._dr import umap, diffmap, pca, tsne
from FACSPy.tools._utils import _merge_dimred_varm_info_into_adata
from FACSPy.tools._utils import _add_uns_data
from FACSPy.tools._utils import _merge_dimred_coordinates_into_adata
from FACSPy.tools._utils import _merge_symmetrical_csr_matrix
from FACSPy.tools._neighbors import _compute_neighbors
from FACSPy.dataset._supplements import Metadata, Panel
from FACSPy.dataset._workspaces import FlowJoWorkspace
from FACSPy._utils import _fetch_fluo_channels
from sklearn.preprocessing import MinMaxScaler

from FACSPy.tools._utils import (_concat_gate_info_and_obs,
                                 _concat_gate_info_and_obs_and_fluo_data,
                                 _scale_adata,
                                 _preprocess_adata)


WSP_FILE_PATH = "FACSPy/_resources/"
WSP_FILE_NAME = "test_wsp.wsp"

def create_supplement_objects():
    INPUT_DIRECTORY = "FACSPy/_resources/test_suite_dataset"
    panel = Panel(input_directory = INPUT_DIRECTORY,
                  file_name = "panel.txt")
    metadata = Metadata(input_directory = INPUT_DIRECTORY,
                        file_name = "metadata_test_suite.csv")
    workspace = FlowJoWorkspace(input_directory = INPUT_DIRECTORY,
                                file_name = "test_suite.wsp")
    return INPUT_DIRECTORY, panel, metadata, workspace

@pytest.fixture
def mock_dataset() -> AnnData:
    input_directory, panel, metadata, workspace = create_supplement_objects()
    adata = fp.create_dataset(input_directory = input_directory,
                              panel = panel,
                              metadata = metadata,
                              workspace = workspace)
    return adata

def test_concat_gate_info_and_obs(mock_dataset: AnnData):
    df = _concat_gate_info_and_obs(mock_dataset)
    assert isinstance(df, pd.DataFrame)
    assert all([col in df.columns for col in mock_dataset.uns["gating_cols"]])
    assert all([col in df.columns for col in mock_dataset.obs.columns])

def test_concat_gate_info_and_obs(mock_dataset: AnnData):
    df = _concat_gate_info_and_obs_and_fluo_data(mock_dataset,
                                                 "compensated")
    assert isinstance(df, pd.DataFrame)
    assert all([col in df.columns for col in mock_dataset.uns["gating_cols"]])
    assert all([col in df.columns for col in mock_dataset.obs.columns])
    expr_data = mock_dataset.to_df(layer = "compensated")
    assert np.array_equal(expr_data["FSC-A"].values, df["FSC-A"].values)

def test_scale_adata_minmaxscaler(mock_dataset: AnnData):
    df = mock_dataset.to_df(layer = "compensated")
    scaled = MinMaxScaler().fit_transform(df.values)

    adata = _scale_adata(mock_dataset,
                         layer = "compensated",
                         scaling = "MinMaxScaler")
    assert np.array_equal(scaled, adata.X)

def test_scale_adata_robustscaler(mock_dataset: AnnData):
    df = mock_dataset.to_df(layer = "compensated")
    scaled = RobustScaler().fit_transform(df.values)

    adata = _scale_adata(mock_dataset,
                         layer = "compensated",
                         scaling = "RobustScaler")
    assert np.array_equal(scaled, adata.X)

def test_scale_adata_robustscaler(mock_dataset: AnnData):
    df = mock_dataset.to_df(layer = "compensated")
    scaled = StandardScaler().fit_transform(df.values)

    adata = _scale_adata(mock_dataset,
                         layer = "compensated",
                         scaling = "StandardScaler")
    assert np.array_equal(scaled, adata.X)

def test_preprocess_adata_use_only_fluo_true(mock_dataset: AnnData):
    adata = _preprocess_adata(mock_dataset,
                              gate = "live",
                              layer = "compensated",
                              scaling = None,
                              use_only_fluo = True,
                              exclude = None)
    isinstance(adata.X, csr_matrix)
    assert np.array_equal(adata.X.todense(), adata.layers["compensated"])
    # fluo channels should be subset
    assert fp._utils.contains_only_fluo(adata)

    fp.subset_gate(mock_dataset,
                   "live")
    fp.subset_fluo_channels(mock_dataset)
    fluo_data_original = mock_dataset.to_df(layer = "compensated")
    assert np.array_equal(fluo_data_original.values, adata.layers["compensated"])
    assert adata.is_view
    assert adata.shape == mock_dataset.shape
    
    
def test_preprocess_adata_scaling(mock_dataset: AnnData):
    adata = _preprocess_adata(mock_dataset,
                              gate = "live",
                              layer = "compensated",
                              scaling = "MinMaxScaler",
                              use_only_fluo = True,
                              exclude = None)
    isinstance(adata.X, csr_matrix)
    # data are scaled so that should fail
    assert not np.array_equal(adata.X.todense(), adata.layers["compensated"])
    # fluo channels should be subset
    assert fp._utils.contains_only_fluo(adata)

    scaled_data = adata.X
    # no scaling performed
    subset = fp.subset_gate(mock_dataset,
                            "live",
                            as_view = True)
    subset = fp.subset_fluo_channels(subset,
                                     as_view = True)
    fluo_data_original = subset.layers["compensated"]
    assert np.array_equal(MinMaxScaler().fit_transform(fluo_data_original),
                          scaled_data.todense())
    assert adata.is_view
    assert adata.shape == subset.shape

def test_preprocess_adata_exclude(mock_dataset: AnnData):
    adata = _preprocess_adata(mock_dataset,
                              gate = "live",
                              layer = "compensated",
                              scaling = "MinMaxScaler",
                              use_only_fluo = True,
                              exclude = ["CD16", "live_dead"])
    # fluo channels should be subset
    assert fp._utils.contains_only_fluo(adata)

    isinstance(adata.X, csr_matrix)
    # data are scaled so that should fail
    assert not np.array_equal(adata.X.todense(), adata.layers["compensated"])

    scaled_data = adata.X
    # no scaling performed
    subset = fp.subset_gate(mock_dataset,
                            "live",
                            as_view = True)
    subset = fp.subset_fluo_channels(subset,
                                     as_view = True)
    subset = fp.remove_channel(subset,
                               ["CD16", "live_dead"],
                               as_view = True)
    fluo_data_original = subset.layers["compensated"]
    assert np.array_equal(MinMaxScaler().fit_transform(fluo_data_original),
                          scaled_data.todense())
    assert adata.is_view
    assert adata.shape == subset.shape

def test_preprocess_adata_use_only_fluo_false(mock_dataset: AnnData):
    adata = _preprocess_adata(mock_dataset,
                              gate = "live",
                              layer = "compensated",
                              scaling = None,
                              use_only_fluo = False,
                              exclude = None)
    # fluo channels should be subset
    assert not fp._utils.contains_only_fluo(adata)
    assert isinstance(adata.X, csr_matrix)
    assert np.array_equal(adata.X.todense(), adata.layers["compensated"])

    scaled_data = adata.X

    subset = fp.subset_gate(mock_dataset,
                            "live",
                            as_view = True)
    fluo_data_original = subset.layers["compensated"]
    assert np.array_equal(fluo_data_original,
                          scaled_data.todense())
    assert adata.is_view
    assert adata.shape == subset.shape

def test_merge_dimred_varm_info_into_adata(mock_dataset: AnnData):
    adata = mock_dataset
    subset = fp.subset_gate(adata, "live", as_view = True)

    pca_ = PCA(n_components = 10)
    pca_.fit_transform(subset.layers["compensated"])
    varm_info = pca_.components_.T
    adata = _merge_dimred_varm_info_into_adata(adata = adata,
                                               gate_subset = subset,
                                               varm_info = varm_info,
                                               dimred = "pca",
                                               dimred_key = "pca_live_compensated")
    
    assert subset.is_view
    assert "pca_live_compensated" in adata.varm
    assert adata.varm["pca_live_compensated"].shape == (21,10)

def test_merge_dimred_coordinates_into_adata(mock_dataset: AnnData):
    adata = mock_dataset
    subset = fp.subset_gate(adata, "live", as_view = True)

    pca_ = PCA(n_components = 10)
    coords = pca_.fit_transform(subset.layers["compensated"])

    adata = _merge_dimred_coordinates_into_adata(adata = adata,
                                                 gate_subset = subset,
                                                 coordinates = coords,
                                                 dimred = "pca",
                                                 dimred_key = "pca_live_compensated")
    assert subset.is_view
    assert "X_pca_live_compensated" in adata.obsm
    assert adata.obsm["X_pca_live_compensated"].shape == (adata.shape[0],10)

def test_add_uns_data(mock_dataset: AnnData):
    adata = mock_dataset
    subset = fp.subset_gate(adata, "live", as_view = True)

    pca_ = PCA(n_components = 10)
    pca_.fit_transform(subset.layers["compensated"])
    variance = pca_.explained_variance_
    variance_ratio = pca_.explained_variance_ratio_
    settings = {"zero_center": False}
    variance_dict = {"variance": variance,
                     "variance_ratio": variance_ratio}

    uns_dict = {"params": settings, **variance_dict}
    print(uns_dict)
    adata = _add_uns_data(adata = adata,
                          data = uns_dict,
                          key_added = "live_compensated_pca")
    
    assert subset.is_view
    assert "live_compensated_pca" in adata.uns
    uns_dict = adata.uns["live_compensated_pca"]
    assert uns_dict["params"] == {"zero_center": False}
    assert isinstance(uns_dict["variance"], np.ndarray)
    assert isinstance(uns_dict["variance_ratio"], np.ndarray)

def test_merge_symmetrical_csr_matrix_scanpy(mock_dataset: AnnData):
    import scanpy as sc
    adata = mock_dataset
    adata.X = adata.layers["compensated"] # needed for scanpy
    live_subset = fp.subset_gate(adata, "live", copy = True)
    sc.pp.neighbors(live_subset)
    connectivities: csr_matrix = live_subset.obsp["connectivities"]
    adata.obsp["connectivities"] = _merge_symmetrical_csr_matrix(adata,
                                                                 live_subset,
                                                                 connectivities)
    # now anndata does the indexing
    live_subset = fp.subset_gate(adata, "live", copy = True)
    indexed_matrix: csr_matrix = live_subset.obsp["connectivities"]
    assert (connectivities != indexed_matrix).nnz == 0

def test_merge_symmetrical_csr_matrix_facspy(mock_dataset: AnnData):
    adata = mock_dataset
    adata.X = adata.layers["compensated"] # needed for scanpy
    live_subset = fp.subset_gate(adata, "live", copy = True)
    _, connectivities, _ = _compute_neighbors(live_subset)
    adata.obsp["connectivities"] = _merge_symmetrical_csr_matrix(adata,
                                                                 live_subset,
                                                                 connectivities)
    # now anndata does the indexing
    live_subset = fp.subset_gate(adata, "live", copy = True)
    indexed_matrix: csr_matrix = live_subset.obsp["connectivities"]
    assert (connectivities != indexed_matrix).nnz == 0
