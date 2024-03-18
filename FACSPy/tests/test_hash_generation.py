import pytest
import os
import scanpy as sc
import pandas as pd
import numpy as np

from anndata import AnnData
import FACSPy as fp
from FACSPy.dataset._supplements import Metadata, Panel
from FACSPy.dataset._workspaces import FlowJoWorkspace

from FACSPy.synchronization._hash_generation import (_generate_dataset_obs_hash,
                                                     _generate_obs_sample_ID_hash,
                                                     _generate_metadata_sample_ID_hash,
                                                     _generate_dataset_var_hash,
                                                     _generate_panel_var_hash,
                                                     _generate_hash_dict)
from FACSPy.synchronization._synchronize import _dataset_has_been_modified, _get_modified_entities

WSP_FILE_PATH = "FACSPy/_resources/"
WSP_FILE_NAME = "test_wsp.wsp"

def create_supplement_objects():
    INPUT_DIRECTORY = "FACSPy/_resources/test_suite_dataset"
    panel = Panel(os.path.join(INPUT_DIRECTORY, "panel.txt"))
    metadata = Metadata(os.path.join(INPUT_DIRECTORY, "metadata_test_suite.csv"))
    workspace = FlowJoWorkspace(os.path.join(INPUT_DIRECTORY, "test_suite.wsp"))
    return INPUT_DIRECTORY, panel, metadata, workspace

@pytest.fixture
def mock_dataset():
    input_directory, panel, metadata, workspace = create_supplement_objects()
    return fp.create_dataset(input_directory = input_directory,
                             panel = panel,
                             metadata = metadata,
                             workspace = workspace,
                             subsample_fcs_to = 100)

def test_generate_hash_dict(mock_dataset: AnnData):
    x = _generate_hash_dict(mock_dataset)
    assert isinstance(x, dict)
    assert x["adata_obs_names"] == _generate_dataset_obs_hash(mock_dataset)
    assert x["adata_var_names"] == _generate_dataset_var_hash(mock_dataset)
    assert x["metadata_sample_ids"] == _generate_metadata_sample_ID_hash(mock_dataset)
    assert x["panel_var_names"] == _generate_panel_var_hash(mock_dataset)
    assert x["adata_sample_ids"] == _generate_obs_sample_ID_hash(mock_dataset)

def test_obs_names_sample_id_hash_generation(mock_dataset: AnnData):
    hash_presubset = _generate_dataset_obs_hash(mock_dataset)
    adata = mock_dataset[0:mock_dataset.shape[0]-10].copy()
    assert _dataset_has_been_modified(adata)
    assert "adata_obs_names" in _get_modified_entities(adata)
    assert not "adata_sample_ids" in _get_modified_entities(adata)
    hash_postsubset = _generate_dataset_obs_hash(adata)
    assert hash_presubset != hash_postsubset

def test_obs_names_hash_function(mock_dataset: AnnData):
    """
    This function tests if the obs_names hash stays the same
    if the cells are shuffled
    """
    hash_preshuffled = _generate_dataset_obs_hash(mock_dataset)
    import scanpy as sc
    shuffled = sc.pp.subsample(mock_dataset, fraction = 1, copy = True)
    assert not _dataset_has_been_modified(mock_dataset)
    assert not _get_modified_entities(mock_dataset)
    hash_postshuffled = _generate_dataset_obs_hash(shuffled)
    assert hash_preshuffled == hash_postshuffled

def test_sample_IDs_obs_hash_function(mock_dataset: AnnData):
    """
    This function tests if the sample_IDs hash stays the same
    if the cells are shuffled
    """
    hash_preshuffled = _generate_obs_sample_ID_hash(mock_dataset)
    shuffled = sc.pp.subsample(mock_dataset, fraction = 1, copy = True)
    assert not _dataset_has_been_modified(mock_dataset)
    assert not _get_modified_entities(mock_dataset)
    hash_postshuffled = _generate_obs_sample_ID_hash(shuffled)
    assert hash_preshuffled == hash_postshuffled

def test_sample_IDs_obs_hash_function_2(mock_dataset: AnnData):
    """
    This function tests if the sample_IDs hash stays does change
    if the sample IDs are removed
    """
    hash_presubset = _generate_obs_sample_ID_hash(mock_dataset)
    adata = mock_dataset[mock_dataset.obs["sample_ID"].isin(["1", "2", "3"])].copy()
    assert _dataset_has_been_modified(adata)
    assert "adata_obs_names" in _get_modified_entities(adata)
    assert "adata_sample_ids" in _get_modified_entities(adata)
    hash_postsubset = _generate_obs_sample_ID_hash(adata)
    assert hash_postsubset != hash_presubset

def test_sample_IDs_metadata_hash_function(mock_dataset: AnnData):
    """
    This function tests if the sample_IDs hash stays the same
    if the cells are shuffled
    """
    hash_preshuffled = _generate_metadata_sample_ID_hash(mock_dataset)
    metadata: pd.DataFrame = mock_dataset.uns["metadata"].dataframe
    metadata = metadata.sample(frac = 1)
    mock_dataset.uns["metadata"] = Metadata(metadata = metadata)
    assert not _dataset_has_been_modified(mock_dataset)
    assert not _get_modified_entities(mock_dataset)
    hash_postshuffled = _generate_metadata_sample_ID_hash(mock_dataset)
    assert hash_preshuffled == hash_postshuffled

def test_sample_IDs_metadata_hash_function_2(mock_dataset: AnnData):
    """
    This function tests if the sample_IDs hash stays does change
    if the sample IDs are removed
    """
    hash_presubset = _generate_metadata_sample_ID_hash(mock_dataset)
    metadata: pd.DataFrame = mock_dataset.uns["metadata"].dataframe
    metadata = metadata[metadata["sample_ID"].isin(["1", "2", "3"])].copy()
    mock_dataset.uns["metadata"] = Metadata(metadata = metadata)
    assert _dataset_has_been_modified(mock_dataset)
    assert "metadata_sample_ids" in _get_modified_entities(mock_dataset)
    hash_postsubset = _generate_metadata_sample_ID_hash(mock_dataset)
    assert hash_postsubset != hash_presubset

def test_var_names_hash(mock_dataset: AnnData):
    """
    This function tests if the var_names hash stays the same
    if the channels are shuffled
    """
    hash_preshuffled = _generate_dataset_var_hash(mock_dataset)
    var_names = mock_dataset.var_names.to_numpy()
    np.random.shuffle(var_names)
    adata = mock_dataset[:, var_names]
    assert not _dataset_has_been_modified(adata)
    assert not _get_modified_entities(adata)
    hash_postshuffled = _generate_dataset_var_hash(adata)
    assert hash_preshuffled == hash_postshuffled

def test_var_names_hash_removed(mock_dataset: AnnData):
    """
    This function tests if the var_names hash stays not the same
    if the channels are removed
    """
    hash_presubset = _generate_dataset_var_hash(mock_dataset)
    var_names = mock_dataset.var_names[:10]
    adata = mock_dataset[:, var_names]
    assert _dataset_has_been_modified(adata)
    assert "adata_var_names" in _get_modified_entities(adata)
    hash_postsubset = _generate_dataset_var_hash(adata)
    assert hash_presubset != hash_postsubset

def test_panel_hash(mock_dataset: AnnData):
    """shuffling should not change the hash"""
    hash_preshuffled = _generate_panel_var_hash(mock_dataset)
    mock_dataset.uns["panel"].dataframe = mock_dataset.uns["panel"].dataframe.sample(frac = 1)
    assert not _dataset_has_been_modified(mock_dataset)
    assert not _get_modified_entities(mock_dataset)
    hash_postshuffled = _generate_panel_var_hash(mock_dataset)
    assert hash_preshuffled == hash_postshuffled

def test_var_names_hash_removed_from_panel(mock_dataset: AnnData):
    """
    This function tests if the var_names hash stays not the same
    if the channels are removed
    """
    hash_presubset = _generate_panel_var_hash(mock_dataset)
    var_names = mock_dataset.var_names[:10]
    mock_dataset.uns["panel"].select_channels(var_names)
    assert _dataset_has_been_modified(mock_dataset)
    assert "panel_var_names" in _get_modified_entities(mock_dataset)
    hash_postsubset = _generate_panel_var_hash(mock_dataset)
    assert hash_presubset != hash_postsubset