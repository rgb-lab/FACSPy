import scanpy as sc
import pandas as pd
import numpy as np

from anndata import AnnData
from FACSPy.dataset._supplements import Metadata

from FACSPy.synchronization._hash_generation import (_generate_dataset_obs_hash,
                                                     _generate_obs_sample_ID_hash,
                                                     _generate_metadata_sample_ID_hash,
                                                     _generate_dataset_var_hash,
                                                     _generate_panel_var_hash,
                                                     _generate_hash_dict)
from FACSPy.synchronization._synchronize import _dataset_has_been_modified, _get_modified_entities, _hash_dataset

def test_generate_hash_dict(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    _hash_dataset(mock_dataset_downsampled)
    x = _generate_hash_dict(mock_dataset_downsampled)
    assert isinstance(x, dict)
    assert x["adata_obs_names"] == _generate_dataset_obs_hash(mock_dataset_downsampled)
    assert x["adata_var_names"] == _generate_dataset_var_hash(mock_dataset_downsampled)
    assert x["metadata_sample_ids"] == _generate_metadata_sample_ID_hash(mock_dataset_downsampled)
    assert x["panel_var_names"] == _generate_panel_var_hash(mock_dataset_downsampled)
    assert x["adata_sample_ids"] == _generate_obs_sample_ID_hash(mock_dataset_downsampled)

def test_obs_names_sample_id_hash_generation(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    _hash_dataset(mock_dataset_downsampled)
    hash_presubset = _generate_dataset_obs_hash(mock_dataset_downsampled)
    adata = mock_dataset_downsampled[0:mock_dataset_downsampled.shape[0]-10].copy()
    assert _dataset_has_been_modified(adata)
    assert "adata_obs_names" in _get_modified_entities(adata)
    assert "adata_sample_ids" not in _get_modified_entities(adata)
    hash_postsubset = _generate_dataset_obs_hash(adata)
    assert hash_presubset != hash_postsubset

def test_obs_names_hash_function(mock_dataset_downsampled: AnnData):
    """
    This function tests if the obs_names hash stays the same
    if the cells are shuffled
    """
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    _hash_dataset(mock_dataset_downsampled)
    hash_preshuffled = _generate_dataset_obs_hash(mock_dataset_downsampled)
    import scanpy as sc
    shuffled = sc.pp.subsample(mock_dataset_downsampled, fraction = 1, copy = True)
    assert not _dataset_has_been_modified(mock_dataset_downsampled)
    assert not _get_modified_entities(mock_dataset_downsampled)
    hash_postshuffled = _generate_dataset_obs_hash(shuffled)
    assert hash_preshuffled == hash_postshuffled

def test_sample_IDs_obs_hash_function(mock_dataset_downsampled: AnnData):
    """
    This function tests if the sample_IDs hash stays the same
    if the cells are shuffled
    """
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    _hash_dataset(mock_dataset_downsampled)
    hash_preshuffled = _generate_obs_sample_ID_hash(mock_dataset_downsampled)
    shuffled = sc.pp.subsample(mock_dataset_downsampled, fraction = 1, copy = True)
    assert not _dataset_has_been_modified(mock_dataset_downsampled)
    assert not _get_modified_entities(mock_dataset_downsampled)
    hash_postshuffled = _generate_obs_sample_ID_hash(shuffled)
    assert hash_preshuffled == hash_postshuffled

def test_sample_IDs_obs_hash_function_2(mock_dataset_downsampled: AnnData):
    """
    This function tests if the sample_IDs hash stays does change
    if the sample IDs are removed
    """
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    _hash_dataset(mock_dataset_downsampled)
    hash_presubset = _generate_obs_sample_ID_hash(mock_dataset_downsampled)
    adata = mock_dataset_downsampled[mock_dataset_downsampled.obs["sample_ID"].isin(["1", "2", "3"])].copy()
    assert _dataset_has_been_modified(adata)
    assert "adata_obs_names" in _get_modified_entities(adata)
    assert "adata_sample_ids" in _get_modified_entities(adata)
    hash_postsubset = _generate_obs_sample_ID_hash(adata)
    assert hash_postsubset != hash_presubset

def test_sample_IDs_metadata_hash_function(mock_dataset_downsampled: AnnData):
    """
    This function tests if the sample_IDs hash stays the same
    if the cells are shuffled
    """
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    _hash_dataset(mock_dataset_downsampled)
    hash_preshuffled = _generate_metadata_sample_ID_hash(mock_dataset_downsampled)
    metadata: pd.DataFrame = mock_dataset_downsampled.uns["metadata"].dataframe
    metadata = metadata.sample(frac = 1)
    mock_dataset_downsampled.uns["metadata"] = Metadata(metadata = metadata)
    assert not _dataset_has_been_modified(mock_dataset_downsampled)
    assert not _get_modified_entities(mock_dataset_downsampled)
    hash_postshuffled = _generate_metadata_sample_ID_hash(mock_dataset_downsampled)
    assert hash_preshuffled == hash_postshuffled

def test_sample_IDs_metadata_hash_function_2(mock_dataset_downsampled: AnnData):
    """
    This function tests if the sample_IDs hash stays does change
    if the sample IDs are removed
    """
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    _hash_dataset(mock_dataset_downsampled)
    hash_presubset = _generate_metadata_sample_ID_hash(mock_dataset_downsampled)
    metadata: pd.DataFrame = mock_dataset_downsampled.uns["metadata"].dataframe
    metadata = metadata[metadata["sample_ID"].isin(["1", "2", "3"])].copy()
    mock_dataset_downsampled.uns["metadata"] = Metadata(metadata = metadata)
    assert _dataset_has_been_modified(mock_dataset_downsampled)
    assert "metadata_sample_ids" in _get_modified_entities(mock_dataset_downsampled)
    hash_postsubset = _generate_metadata_sample_ID_hash(mock_dataset_downsampled)
    assert hash_postsubset != hash_presubset

def test_var_names_hash(mock_dataset_downsampled: AnnData):
    """
    This function tests if the var_names hash stays the same
    if the channels are shuffled
    """
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    _hash_dataset(mock_dataset_downsampled)
    hash_preshuffled = _generate_dataset_var_hash(mock_dataset_downsampled)
    var_names = mock_dataset_downsampled.var_names.to_numpy()
    np.random.shuffle(var_names)
    adata = mock_dataset_downsampled[:, var_names]
    assert not _dataset_has_been_modified(adata)
    assert not _get_modified_entities(adata)
    hash_postshuffled = _generate_dataset_var_hash(adata)
    assert hash_preshuffled == hash_postshuffled

def test_var_names_hash_removed(mock_dataset_downsampled: AnnData):
    """
    This function tests if the var_names hash stays not the same
    if the channels are removed
    """
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    _hash_dataset(mock_dataset_downsampled)
    hash_presubset = _generate_dataset_var_hash(mock_dataset_downsampled)
    var_names = mock_dataset_downsampled.var_names[:10]
    adata = mock_dataset_downsampled[:, var_names]
    assert _dataset_has_been_modified(adata)
    assert "adata_var_names" in _get_modified_entities(adata)
    hash_postsubset = _generate_dataset_var_hash(adata)
    assert hash_presubset != hash_postsubset

def test_panel_hash(mock_dataset_downsampled: AnnData):
    """shuffling should not change the hash"""
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    _hash_dataset(mock_dataset_downsampled)
    hash_preshuffled = _generate_panel_var_hash(mock_dataset_downsampled)
    mock_dataset_downsampled.uns["panel"].dataframe = mock_dataset_downsampled.uns["panel"].dataframe.sample(frac = 1)
    assert not _dataset_has_been_modified(mock_dataset_downsampled)
    assert not _get_modified_entities(mock_dataset_downsampled)
    hash_postshuffled = _generate_panel_var_hash(mock_dataset_downsampled)
    assert hash_preshuffled == hash_postshuffled

def test_var_names_hash_removed_from_panel(mock_dataset_downsampled: AnnData):
    """
    This function tests if the var_names hash stays not the same
    if the channels are removed
    """
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    _hash_dataset(mock_dataset_downsampled)
    hash_presubset = _generate_panel_var_hash(mock_dataset_downsampled)
    var_names = mock_dataset_downsampled.var_names[:10]
    mock_dataset_downsampled.uns["panel"].select_channels(var_names)
    assert _dataset_has_been_modified(mock_dataset_downsampled)
    assert "panel_var_names" in _get_modified_entities(mock_dataset_downsampled)
    hash_postsubset = _generate_panel_var_hash(mock_dataset_downsampled)
    assert hash_presubset != hash_postsubset
