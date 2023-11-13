import warnings

import pytest
from anndata import AnnData
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

import FACSPy as fp
from FACSPy.dataset._supplements import Metadata, Panel
from FACSPy.dataset._workspaces import FlowJoWorkspace
from FACSPy.exceptions._exceptions import (AnalysisNotPerformedError,
                                           InvalidScalingError,
                                           InsufficientSampleNumberWarning,
                                           DimredSettingModificationWarning)
from FACSPy._utils import find_gate_path_of_gate, _fetch_fluo_channels
from FACSPy.tools._dr_samplewise import (pca_samplewise,
                                         mds_samplewise,
                                         umap_samplewise,
                                         tsne_samplewise,
                                         _save_samplewise_dr_settings,
                                         _perform_dr)

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
def mock_dataset_without_mfi_calc() -> AnnData:
    input_directory, panel, metadata, workspace = create_supplement_objects()
    adata = fp.create_dataset(input_directory = input_directory,
                              panel = panel,
                              metadata = metadata,
                              workspace = workspace)
    return adata

@pytest.fixture
def mock_dataset() -> AnnData:
    input_directory, panel, metadata, workspace = create_supplement_objects()
    adata = fp.create_dataset(input_directory = input_directory,
                              panel = panel,
                              metadata = metadata,
                              workspace = workspace)
    fp.tl.mfi(adata,
              layer = "compensated")
    return adata

def test_invalid_scaling(mock_dataset):
    with pytest.raises(InvalidScalingError):
        fp.tl.umap_samplewise(mock_dataset,
                              scaling = "MyCustomScaler")
    with pytest.raises(InvalidScalingError):
        fp.tl.tsne_samplewise(mock_dataset,
                              scaling = "MyCustomScaler")
    with pytest.raises(InvalidScalingError):
        fp.tl.pca_samplewise(mock_dataset,
                             scaling = "MyCustomScaler")
    with pytest.raises(InvalidScalingError):
        fp.tl.mds_samplewise(mock_dataset,
                             scaling = "MyCustomScaler")
        
def test_analysis_not_performed(mock_dataset_without_mfi_calc):
    with pytest.raises(AnalysisNotPerformedError):
        fp.tl.umap_samplewise(mock_dataset_without_mfi_calc)
    with pytest.raises(AnalysisNotPerformedError):
        fp.tl.tsne_samplewise(mock_dataset_without_mfi_calc)
    with pytest.raises(AnalysisNotPerformedError):
        fp.tl.pca_samplewise(mock_dataset_without_mfi_calc)
    with pytest.raises(AnalysisNotPerformedError):
        fp.tl.mds_samplewise(mock_dataset_without_mfi_calc)

def test_samplewise_dimreds_work_with_standard_settings(mock_dataset: AnnData):
    fp.tl.pca_samplewise(mock_dataset)
    fp.tl.umap_samplewise(mock_dataset)
    fp.tl.tsne_samplewise(mock_dataset)
    fp.tl.mds_samplewise(mock_dataset)
    assert True

def test_saving(mock_dataset: AnnData):
    _save_samplewise_dr_settings(mock_dataset,
                                 data_group = "sample_ID",
                                 data_metric = "mfi",
                                 layer = "transformed",
                                 use_only_fluo = True,
                                 exclude = ["some", "channels"],
                                 scaling = "MinMaxScaler",
                                 n_components = 15,
                                 reduction = "pca")
    assert "settings" in mock_dataset.uns
    settings_dict = mock_dataset.uns["settings"]
    assert "_pca_samplewise_mfi_transformed" in settings_dict
    settings = settings_dict["_pca_samplewise_mfi_transformed"]
    assert settings["data_group"] == "sample_ID"
    assert settings["data_metric"] == "mfi"
    assert settings["layer"] == "transformed"
    assert settings["use_only_fluo"] == True
    assert settings["exclude"] == ["some", "channels"]
    assert settings["scaling"] == "MinMaxScaler"
    assert settings["n_components"] == 15

def test_saving_with_kwargs(mock_dataset: AnnData):
    _save_samplewise_dr_settings(mock_dataset,
                                 data_group = "sample_ID",
                                 data_metric = "mfi",
                                 layer = "transformed",
                                 use_only_fluo = True,
                                 exclude = ["some", "channels"],
                                 scaling = "MinMaxScaler",
                                 reduction = "pca",
                                 n_components = 15,
                                 keyword1 = "something",
                                 keyword2 = "else")
    assert "settings" in mock_dataset.uns
    settings_dict = mock_dataset.uns["settings"]
    assert "_pca_samplewise_mfi_transformed" in settings_dict
    settings = settings_dict["_pca_samplewise_mfi_transformed"]
    assert settings["data_group"] == "sample_ID"
    assert settings["data_metric"] == "mfi"
    assert settings["layer"] == "transformed"
    assert settings["use_only_fluo"] == True
    assert settings["exclude"] == ["some", "channels"]
    assert settings["scaling"] == "MinMaxScaler"
    assert settings["n_components"] == 15
    assert settings["keyword1"] == "something"
    assert settings["keyword2"] == "else"

def test_values_tsne(mock_dataset: AnnData):
    adata = mock_dataset
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    mfi_values = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"), :].values
    mfi_values = MinMaxScaler().fit_transform(mfi_values)
    coords_manual = TSNE(n_components = 3,
                         perplexity = min(30, mfi_values.shape[0]-1),
                         random_state = 187).fit_transform(mfi_values)
    
    fp.tl.tsne_samplewise(adata,
                          use_only_fluo = False,
                          layer = "compensated")

    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    coords_facspy = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"),
                           ["TSNE1", "TSNE2", "TSNE3"]]
    np.testing.assert_array_equal(coords_manual, coords_facspy.values)

def test_kwarg_passing_tsne(mock_dataset: AnnData):
    adata = mock_dataset
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    mfi_values = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"), :].values
    mfi_values = MinMaxScaler().fit_transform(mfi_values)
    coords_manual = TSNE(n_components = 3,
                         perplexity = min(30, mfi_values.shape[0]-1),
                         random_state = 187,
                         learning_rate = 100).fit_transform(mfi_values)
    
    fp.tl.tsne_samplewise(adata,
                          use_only_fluo = False,
                          layer = "compensated",
                          learning_rate = 100)

    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    coords_facspy = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"),
                           ["TSNE1", "TSNE2", "TSNE3"]]
    np.testing.assert_array_equal(coords_manual, coords_facspy.values)

def test_exclude_channels_tsne(mock_dataset: AnnData):
    adata = mock_dataset
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    mfi_values = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"), :]
    mfi_values = mfi_values.loc[:,~mfi_values.columns.isin(["CD15", "live_dead"])].values
    mfi_values = MinMaxScaler().fit_transform(mfi_values)
    coords_manual = TSNE(n_components = 3,
                         perplexity = min(30, mfi_values.shape[0]-1),
                         random_state = 187).fit_transform(mfi_values)
    
    fp.tl.tsne_samplewise(adata,
                          use_only_fluo = False,
                          layer = "compensated",
                          exclude = ["CD15", "live_dead"])

    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    coords_facspy = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"),
                           ["TSNE1", "TSNE2", "TSNE3"]]
    np.testing.assert_array_equal(coords_manual, coords_facspy.values)

def test_use_only_fluo_channels_tsne(mock_dataset: AnnData):
    adata = mock_dataset
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    mfi_values = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"), :]
    mfi_values = mfi_values[_fetch_fluo_channels(adata)].values
    mfi_values = MinMaxScaler().fit_transform(mfi_values)
    coords_manual = TSNE(n_components = 3,
                         perplexity = min(30, mfi_values.shape[0]-1),
                         random_state = 187).fit_transform(mfi_values)
    
    fp.tl.tsne_samplewise(adata,
                          use_only_fluo = True,
                          layer = "compensated")

    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    coords_facspy = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"),
                           ["TSNE1", "TSNE2", "TSNE3"]]
    np.testing.assert_array_equal(coords_manual, coords_facspy.values)


def test_n_components_parameter_tsne(mock_dataset: AnnData):
    fp.tl.tsne_samplewise(mock_dataset,
                          layer = "compensated",
                          n_components = 4)
    uns_frame = mock_dataset.uns["mfi_sample_ID_compensated"]
    assert "TSNE1" in uns_frame.columns
    assert "TSNE2" in uns_frame.columns
    assert "TSNE3" in uns_frame.columns
    assert "TSNE4" in uns_frame.columns

def test_dr_perform_function_tsne(mock_dataset: AnnData):
    df: pd.DataFrame = mock_dataset.uns["mfi_sample_ID_compensated"]
    df_array = df.values
    output_array = _perform_dr(reduction = "TSNE",
                               data = df_array,
                               n_components = 3)
    assert isinstance(output_array, np.ndarray)
    assert output_array.shape[1] == 3
    assert output_array.shape[0] == df_array.shape[0]

def test_warnings_tsne(mock_dataset: AnnData):
    """both warnings will appear due to low sample size in the test dataset"""
    with pytest.warns(InsufficientSampleNumberWarning):
        tsne_samplewise(mock_dataset)
    with pytest.warns(DimredSettingModificationWarning):
        tsne_samplewise(mock_dataset)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DimredSettingModificationWarning)
        tsne_samplewise(mock_dataset,
                        perplexity = 0.3,
                        method = "exact")

def test_values_mds(mock_dataset: AnnData):
    adata = mock_dataset
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    mfi_values = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"), :].values
    mfi_values = MinMaxScaler().fit_transform(mfi_values)
    coords_manual = MDS(n_components = 3,
                        random_state = 187).fit_transform(mfi_values)
    
    fp.tl.mds_samplewise(adata,
                          use_only_fluo = False,
                          layer = "compensated")

    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    coords_facspy = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"),
                               ["MDS1", "MDS2", "MDS3"]]
    np.testing.assert_array_equal(coords_manual, coords_facspy.values)

def test_kwarg_passing_mds(mock_dataset: AnnData):
    adata = mock_dataset
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    mfi_values = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"), :].values
    mfi_values = MinMaxScaler().fit_transform(mfi_values)
    coords_manual = MDS(n_components = 3,
                        random_state = 187,
                        eps = 1).fit_transform(mfi_values)
    
    fp.tl.mds_samplewise(adata,
                          use_only_fluo = False,
                          layer = "compensated",
                          eps = 1)

    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    coords_facspy = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"),
                           ["MDS1", "MDS2", "MDS3"]]
    np.testing.assert_array_equal(coords_manual, coords_facspy.values)

def test_exclude_channels_mds(mock_dataset: AnnData):
    adata = mock_dataset
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    mfi_values = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"), :]
    mfi_values = mfi_values.loc[:,~mfi_values.columns.isin(["CD15", "live_dead"])].values
    mfi_values = MinMaxScaler().fit_transform(mfi_values)
    coords_manual = MDS(n_components = 3,
                        random_state = 187).fit_transform(mfi_values)
    
    fp.tl.mds_samplewise(adata,
                         use_only_fluo = False,
                         layer = "compensated",
                         exclude = ["CD15", "live_dead"])

    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    coords_facspy = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"),
                           ["MDS1", "MDS2", "MDS3"]]
    np.testing.assert_array_equal(coords_manual, coords_facspy.values)

def test_use_only_fluo_channels_mds(mock_dataset: AnnData):
    adata = mock_dataset
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    mfi_values = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"), :]
    mfi_values = mfi_values[_fetch_fluo_channels(adata)].values
    mfi_values = MinMaxScaler().fit_transform(mfi_values)
    coords_manual = MDS(n_components = 3,
                        random_state = 187).fit_transform(mfi_values)
    
    fp.tl.mds_samplewise(adata,
                          use_only_fluo = True,
                          layer = "compensated")

    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    coords_facspy = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"),
                           ["MDS1", "MDS2", "MDS3"]]
    np.testing.assert_array_equal(coords_manual, coords_facspy.values)


def test_n_components_parameter_mds(mock_dataset: AnnData):
    fp.tl.mds_samplewise(mock_dataset,
                         layer = "compensated",
                         n_components = 4)
    uns_frame = mock_dataset.uns["mfi_sample_ID_compensated"]
    assert "MDS1" in uns_frame.columns
    assert "MDS2" in uns_frame.columns
    assert "MDS3" in uns_frame.columns
    assert "MDS4" in uns_frame.columns

def test_dr_perform_function_mds(mock_dataset: AnnData):
    df: pd.DataFrame = mock_dataset.uns["mfi_sample_ID_compensated"]
    df_array = df.values
    output_array = _perform_dr(reduction = "MDS",
                               data = df_array,
                               n_components = 3)
    assert isinstance(output_array, np.ndarray)
    assert output_array.shape[1] == 3
    assert output_array.shape[0] == df_array.shape[0]

def test_warnings_mds(mock_dataset: AnnData):
    """both warnings will appear due to low sample size in the test dataset"""
    with pytest.warns(InsufficientSampleNumberWarning):
        mds_samplewise(mock_dataset)
    with pytest.warns(DimredSettingModificationWarning):
        mds_samplewise(mock_dataset)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DimredSettingModificationWarning)
        mds_samplewise(mock_dataset,
                       normalized_stress = "auto")

def test_values_umap(mock_dataset: AnnData):
    adata = mock_dataset
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    mfi_values = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"), :].values
    mfi_values = MinMaxScaler().fit_transform(mfi_values)
    coords_manual = UMAP(n_components = 3,
                         random_state = 187).fit_transform(mfi_values)
    
    fp.tl.umap_samplewise(adata,
                          use_only_fluo = False,
                          layer = "compensated")

    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    coords_facspy = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"),
                               ["UMAP1", "UMAP2", "UMAP3"]]
    np.testing.assert_array_equal(coords_manual, coords_facspy.values)

def test_kwarg_passing_umap(mock_dataset: AnnData):
    adata = mock_dataset
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    mfi_values = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"), :].values
    mfi_values = MinMaxScaler().fit_transform(mfi_values)
    coords_manual = UMAP(n_components = 3,
                         random_state = 187,
                         metric = "manhattan").fit_transform(mfi_values)
    
    fp.tl.umap_samplewise(adata,
                          use_only_fluo = False,
                          layer = "compensated",
                          metric = "manhattan")

    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    coords_facspy = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"),
                           ["UMAP1", "UMAP2", "UMAP3"]]
    np.testing.assert_array_equal(coords_manual, coords_facspy.values)

def test_exclude_channels_umap(mock_dataset: AnnData):
    adata = mock_dataset
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    mfi_values = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"), :]
    mfi_values = mfi_values.loc[:,~mfi_values.columns.isin(["CD15", "live_dead"])].values
    mfi_values = MinMaxScaler().fit_transform(mfi_values)
    coords_manual = UMAP(n_components = 3,
                         random_state = 187).fit_transform(mfi_values)
    
    fp.tl.umap_samplewise(adata,
                          use_only_fluo = False,
                          layer = "compensated",
                          exclude = ["CD15", "live_dead"])

    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    coords_facspy = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"),
                           ["UMAP1", "UMAP2", "UMAP3"]]
    np.testing.assert_array_equal(coords_manual, coords_facspy.values)

def test_use_only_fluo_channels_umap(mock_dataset: AnnData):
    adata = mock_dataset
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    mfi_values = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"), :]
    mfi_values = mfi_values[_fetch_fluo_channels(adata)].values
    mfi_values = MinMaxScaler().fit_transform(mfi_values)
    coords_manual = UMAP(n_components = 3,
                         random_state = 187).fit_transform(mfi_values)
    
    fp.tl.umap_samplewise(adata,
                          use_only_fluo = True,
                          layer = "compensated")

    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    coords_facspy = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"),
                           ["UMAP1", "UMAP2", "UMAP3"]]
    np.testing.assert_array_equal(coords_manual, coords_facspy.values)


def test_n_components_parameter_umap(mock_dataset: AnnData):
    fp.tl.umap_samplewise(mock_dataset,
                          layer = "compensated",
                          n_components = 4)
    uns_frame = mock_dataset.uns["mfi_sample_ID_compensated"]
    assert "UMAP1" in uns_frame.columns
    assert "UMAP2" in uns_frame.columns
    assert "UMAP3" in uns_frame.columns
    assert "UMAP4" in uns_frame.columns

def test_dr_perform_function_umap(mock_dataset: AnnData):
    df: pd.DataFrame = mock_dataset.uns["mfi_sample_ID_compensated"]
    df_array = df.values
    output_array = _perform_dr(reduction = "UMAP",
                               data = df_array,
                               n_components = 3)
    assert isinstance(output_array, np.ndarray)
    assert output_array.shape[1] == 3
    assert output_array.shape[0] == df_array.shape[0]

def test_warnings_umap(mock_dataset: AnnData):
    from FACSPy.exceptions._exceptions import InsufficientSampleNumberWarning, DimredSettingModificationWarning
    """both warnings will appear due to low sample size in the test dataset"""
    with pytest.warns(InsufficientSampleNumberWarning):
        umap_samplewise(mock_dataset)
    with pytest.warns(DimredSettingModificationWarning):
        umap_samplewise(mock_dataset)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DimredSettingModificationWarning)
        umap_samplewise(mock_dataset,
                        init = "random")

def test_values_pca(mock_dataset: AnnData):
    adata = mock_dataset
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    mfi_values = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"), :].values
    mfi_values = MinMaxScaler().fit_transform(mfi_values)
    # the kwarg we choose is svd_solver
    pca_coords_manual = PCA(n_components = 3,
                            random_state = 187).fit_transform(mfi_values)
    
    fp.tl.pca_samplewise(adata,
                         use_only_fluo = False,
                         layer = "compensated")

    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    pca_coords_facspy = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"), ["PCA1", "PCA2", "PCA3"]]
    np.testing.assert_array_equal(pca_coords_manual, pca_coords_facspy.values)

def test_kwarg_passing_pca(mock_dataset: AnnData):
    adata = mock_dataset
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    mfi_values = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"), :].values
    mfi_values = MinMaxScaler().fit_transform(mfi_values)
    # the kwarg we choose is svd_solver
    pca_coords_manual = PCA(n_components = 3,
                            random_state = 187,
                            whiten = True).fit_transform(mfi_values)
    
    fp.tl.pca_samplewise(adata,
                         use_only_fluo = False,
                         layer = "compensated",
                         whiten = True)

    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    pca_coords_facspy = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"), ["PCA1", "PCA2", "PCA3"]]
    np.testing.assert_array_equal(pca_coords_manual, pca_coords_facspy.values)

def test_exclude_channels_pca(mock_dataset: AnnData):
    adata = mock_dataset
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    mfi_values = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"), :]
    mfi_values = mfi_values.loc[:,~mfi_values.columns.isin(["CD15", "live_dead"])].values

    mfi_values = MinMaxScaler().fit_transform(mfi_values)
    # the kwarg we choose is svd_solver
    pca_coords_manual = PCA(n_components = 3,
                            random_state = 187).fit_transform(mfi_values)
    
    fp.tl.pca_samplewise(adata,
                         use_only_fluo = False,
                         layer = "compensated",
                         exclude = ["CD15", "live_dead"])

    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    pca_coords_facspy = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"), ["PCA1", "PCA2", "PCA3"]]
    np.testing.assert_array_equal(pca_coords_manual, pca_coords_facspy.values)

def test_use_only_fluo_channels_pca(mock_dataset: AnnData):
    adata = mock_dataset
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    mfi_values = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"), :]
    mfi_values = mfi_values[_fetch_fluo_channels(adata)].values
    mfi_values = MinMaxScaler().fit_transform(mfi_values)
    # the kwarg we choose is svd_solver
    pca_coords_manual = PCA(n_components = 3,
                            random_state = 187).fit_transform(mfi_values)
    
    fp.tl.pca_samplewise(adata,
                         use_only_fluo = True,
                         layer = "compensated")

    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    pca_coords_facspy = df.loc[df.index.get_level_values("gate") == find_gate_path_of_gate(adata, "live"), ["PCA1", "PCA2", "PCA3"]]
    np.testing.assert_array_equal(pca_coords_manual, pca_coords_facspy.values)

def test_n_components_parameter_pca(mock_dataset: AnnData):
    fp.tl.pca_samplewise(mock_dataset,
                         layer = "compensated",
                         n_components = 4)
    uns_frame = mock_dataset.uns["mfi_sample_ID_compensated"]
    assert "PCA1" in uns_frame.columns
    assert "PCA2" in uns_frame.columns
    assert "PCA3" in uns_frame.columns
    assert "PCA4" in uns_frame.columns

def test_dr_perform_function_pca(mock_dataset: AnnData):
    df = mock_dataset.uns["mfi_sample_ID_compensated"]
    df_array = df.values
    output_array = _perform_dr(reduction = "PCA",
                               data = df_array,
                               n_components = 3)
    assert isinstance(output_array, np.ndarray)
    assert output_array.shape[1] == 3
    assert output_array.shape[0] == df_array.shape[0]

def test_warnings_pca(mock_dataset: AnnData):
    """warnings will appear due to low sample size in the test dataset"""
    with pytest.warns(InsufficientSampleNumberWarning):
        pca_samplewise(mock_dataset)
