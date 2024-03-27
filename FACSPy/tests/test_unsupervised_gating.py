from anndata import AnnData
import pytest
import FACSPy as fp
import pandas as pd
import numpy as np
import os
from FACSPy.dataset._supplements import Metadata, Panel
from FACSPy.dataset._workspaces import FlowJoWorkspace
from FACSPy.model._annotators import unsupervisedGating
import scanpy as sc

WSP_FILE_PATH = "FACSPy/_resources/"
WSP_FILE_NAME = "test_wsp.wsp"

def create_supplement_objects():
    INPUT_DIRECTORY = "FACSPy/_resources/test_suite_dataset"
    panel = Panel(os.path.join(INPUT_DIRECTORY, "panel.txt"))
    metadata = Metadata(os.path.join(INPUT_DIRECTORY, "metadata_test_suite.csv"))
    workspace = FlowJoWorkspace(os.path.join(INPUT_DIRECTORY, "test_suite.wsp"))
    return INPUT_DIRECTORY, panel, metadata, workspace

@pytest.fixture
def mock_dataset() -> AnnData:
    input_directory, panel, metadata, workspace = create_supplement_objects()
    adata = fp.create_dataset(input_directory = input_directory,
                              panel = panel,
                              metadata = metadata,
                              workspace = workspace,
                              subsample_fcs_to = 100)
    sc.pp.subsample(adata, n_obs = 200, random_state = 187)
    return adata

@pytest.fixture
def unsup_gator(mock_dataset: AnnData):
    gating_strategy = {"T_cells": ["CD45+", ["CD3+"]]}
    return fp.ml.unsupervisedGating(mock_dataset,
                                    gating_strategy = gating_strategy,
                                    layer = "compensated")

def test_class_init(unsup_gator: unsupervisedGating):
    assert unsup_gator.layer == "compensated"
    assert isinstance(unsup_gator.adata, AnnData)
    assert unsup_gator.cluster_key == "clusters"
    assert unsup_gator.sensitivity == 1
    assert unsup_gator.intervals == [0.33, 0.66]

def test_class_init_TypeError(mock_dataset: AnnData):
    gating_strategy = {"T_cells": ["CD45+", ["CD3+"]]}
    with pytest.raises(TypeError):
        _ = fp.ml.unsupervisedGating(mock_dataset,
                                     gating_strategy = gating_strategy,
                                     layer = "compensated",
                                     intervals = [1,2,3])

def test_class_init_ValueError(mock_dataset: AnnData):
    gating_strategy = {}
    with pytest.raises(ValueError):
        _ = fp.ml.unsupervisedGating(mock_dataset,
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
