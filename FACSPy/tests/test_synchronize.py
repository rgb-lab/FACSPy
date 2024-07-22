import pytest

from anndata import AnnData
import FACSPy as fp
from FACSPy.synchronization._synchronize import _dataset_has_been_modified, _hash_dataset
from FACSPy.exceptions._exceptions import DataModificationWarning

def test_datamodification_warning(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    _hash_dataset(mock_dataset_downsampled)
    mock_dataset_downsampled = mock_dataset_downsampled[0:mock_dataset_downsampled.shape[0]-10]
    with pytest.warns(DataModificationWarning):
        fp.sync.synchronize_dataset(mock_dataset_downsampled)
    assert not _dataset_has_been_modified(mock_dataset_downsampled)

def test_rehashing(mock_dataset_downsampled: AnnData):
    """datasets should not be modified after syncing"""
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    _hash_dataset(mock_dataset_downsampled)
    mock_dataset_downsampled = mock_dataset_downsampled[0:mock_dataset_downsampled.shape[0]-10]
    fp.sync.synchronize_dataset(mock_dataset_downsampled)
    assert not _dataset_has_been_modified(mock_dataset_downsampled)
