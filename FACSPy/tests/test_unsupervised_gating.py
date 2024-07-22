from anndata import AnnData
import pytest
import FACSPy as fp
import pandas as pd
import numpy as np
from FACSPy.model._annotators import unsupervisedGating


@pytest.fixture
def unsup_gator(mock_dataset_downsampled: AnnData):
    gating_strategy = {"T_cells": ["CD45+", ["CD3+"]]}
    return fp.ml.unsupervisedGating(mock_dataset_downsampled,
                                    gating_strategy = gating_strategy,
                                    layer = "compensated")

def test_class_init(unsup_gator: unsupervisedGating):
    assert unsup_gator.layer == "compensated"
    assert isinstance(unsup_gator.adata, AnnData)
    assert unsup_gator.cluster_key == "clusters"
    assert unsup_gator.sensitivity == 1
    assert unsup_gator.intervals == [0.33, 0.66]

def test_class_init_TypeError(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    gating_strategy = {"T_cells": ["CD45+", ["CD3+"]]}
    with pytest.raises(TypeError):
        _ = fp.ml.unsupervisedGating(mock_dataset_downsampled,
                                     gating_strategy = gating_strategy,
                                     layer = "compensated",
                                     intervals = [1,2,3])

def test_class_init_ValueError(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    gating_strategy = {}
    with pytest.raises(ValueError):
        _ = fp.ml.unsupervisedGating(mock_dataset_downsampled,
                                     layer = "compensated",
                                     gating_strategy = gating_strategy)

def test_clean_marker_names_list_ValueError(unsup_gator: unsupervisedGating):
    clf = unsup_gator
    # ["/", "[", "{", "(", ")", "}", "]", ".", "-"]
    with pytest.raises(ValueError):
        wrong_input_type = np.array(["CD3"])
        clf._clean_marker_names(wrong_input_type)

def test_clean_marker_names_list(unsup_gator: unsupervisedGating):
    clf = unsup_gator
    # ["/", "[", "{", "(", ")", "}", "]", ".", "-"]
    disallowed_characters = clf._disallowed_characters
    for character in disallowed_characters:
        markers = pd.Index([f"CD{character}3"])
        x = clf._clean_marker_names(markers)
        assert isinstance(x, list)
        assert clf._clean_marker_names(markers) == ["CD3"]

def test_clean_marker_names_dict(unsup_gator: unsupervisedGating):
    clf = unsup_gator
    # ["/", "[", "{", "(", ")", "}", "]", ".", "-"]
    disallowed_characters = clf._disallowed_characters
    for character in disallowed_characters:
        markers = {"+": [f"CD{character}3"]}
        x = clf._clean_marker_names(markers)
        assert isinstance(x, dict)
        assert clf._clean_marker_names(markers)["+"] == ["CD3"]
