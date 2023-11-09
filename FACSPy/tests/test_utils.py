import pytest
import pandas as pd
import numpy as np
from anndata import AnnData
import anndata as ad

from FACSPy._utils import (GATE_SEPARATOR,
                           find_gate_path_of_gate,
                           find_gate_indices,
                           equalize_groups,
                           find_parent_population,
                           find_current_population,
                           find_parent_gate,
                           find_parents_recursively,
                           ifelse,
                           find_grandparent_gate,
                           find_grandparent_population,
                           close_polygon_gate_coordinates,
                           flatten_nested_list,
                           subset_fluo_channels,
                           subset_gate)

from FACSPy.exceptions._utils import (GateNotProvidedError,
                                      ExhaustedHierarchyError)

# QUICKFIND: Gates
@pytest.fixture
def mock_anndata_gating() -> AnnData:
    adata1 = AnnData(
        X = np.zeros((10,10)),
        obs = pd.DataFrame(
            data = {
                "sample_ID": [1] * 10,
                "donor": [1] * 10,
                "something": [1] * 10
            }
        )
    )
    adata2 = AnnData(
        X = np.zeros((100,10)),
        obs = pd.DataFrame(
            data = {
                "sample_ID": [2] * 100,
                "donor": [2] * 100,
                "something": [2] * 100
            }
        )
    )
    adata3 = AnnData(
        X = np.zeros((1000,10)),
        obs = pd.DataFrame(
            data = {
                "sample_ID": [3] * 1000,
                "donor": [3] * 1000,
                "something": [3] * 500 + [4] * 500
            }
        )
    )
    concatenated = ad.concat([adata1, adata2, adata3])
    concatenated.var_names_make_unique()
    concatenated.obs_names_make_unique()
    concatenated.uns = {"gating_cols": pd.Index([f"root{GATE_SEPARATOR}singlets",
                                                 f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}T_cells",
                                                 f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}T_cells{GATE_SEPARATOR}live"])}
    return concatenated


def test_find_gate_path_of_gate(mock_anndata_gating: AnnData):
    assert find_gate_path_of_gate(mock_anndata_gating, "T_cells") == f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}T_cells"
    assert find_gate_path_of_gate(mock_anndata_gating, "live") == f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}T_cells{GATE_SEPARATOR}live"
    assert find_gate_path_of_gate(mock_anndata_gating, "singlets") == f"root{GATE_SEPARATOR}singlets"

def test_find_gate_indices(mock_anndata_gating):
    assert find_gate_indices(mock_anndata_gating, find_gate_path_of_gate(mock_anndata_gating,"T_cells")) == [1]


# QUICKFIND: Group Equalizing
@pytest.fixture
def mock_anndata() -> AnnData:
    adata1 = AnnData(
        X = np.zeros((10,10)),
        obs = pd.DataFrame(
            data = {
                "sample_ID": [1] * 10,
                "donor": [1] * 10,
                "something": [1] * 10
            }
        )
    )
    adata2 = AnnData(
        X = np.zeros((100,10)),
        obs = pd.DataFrame(
            data = {
                "sample_ID": [2] * 100,
                "donor": [2] * 100,
                "something": [2] * 100
            }
        )
    )
    adata3 = AnnData(
        X = np.zeros((1000,10)),
        obs = pd.DataFrame(
            data = {
                "sample_ID": [3] * 1000,
                "donor": [3] * 1000,
                "something": [3] * 500 + [4] * 500
            }
        )
    )
    concatenated = ad.concat([adata1, adata2, adata3])
    concatenated.var_names_make_unique()
    concatenated.obs_names_make_unique()
    return concatenated

def test_equalizing_sample_ID(mock_anndata):
    equalize_groups(mock_anndata, fraction = 0.1)
    assert mock_anndata.shape[0] == 3 ## smallest group has 10

def test_equalizing_something(mock_anndata):
    equalize_groups(mock_anndata, n_obs = 10)
    assert mock_anndata.shape[0] == 30


def test_too_much_to_sample_per_group(mock_anndata):
    with pytest.raises(ValueError):
        equalize_groups(mock_anndata, n_obs = 11)

def test_value_error(mock_anndata):
    with pytest.raises(ValueError):
        equalize_groups(mock_anndata, fraction = 0.1, n_obs = 100)

def test_too_much_to_sample(mock_anndata):
    with pytest.raises(ValueError):
        equalize_groups(mock_anndata, n_obs = 10_000)

# QUICKFIND: Population
def test_find_current_population():
    test_string1 = f"root{GATE_SEPARATOR}singlets"
    test_string2 = "root"
    test_string3 = f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}live"
    test_string4 = ""

    assert find_current_population(test_string1) == "singlets"
    assert find_current_population(test_string2) == "root"
    assert find_current_population(test_string3) == "live"
    with pytest.raises(GateNotProvidedError):
        find_current_population(test_string4)

def test_find_parent_gate():
    test_string1 = f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}T_cells"
    test_string2 = "root"
    test_string3 = f"root{GATE_SEPARATOR}singlets"
    test_string4 = ""

    assert find_parent_gate(test_string1) == f"root{GATE_SEPARATOR}singlets"
    assert find_parent_gate(test_string3) ==  "root"
    with pytest.raises(ExhaustedHierarchyError):
        find_parent_gate(test_string2)
    with pytest.raises(GateNotProvidedError):
        find_parent_gate(test_string4)

def test_find_grandparent_gate():
    test_string1 = f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}T_cells"
    test_string2 = "root"
    test_string3 = f"root{GATE_SEPARATOR}singlets"
    test_string4 = ""

    assert find_grandparent_gate(test_string1) == "root"
    with pytest.raises(ExhaustedHierarchyError):
        find_grandparent_gate(test_string2)
    with pytest.raises(ExhaustedHierarchyError):
        find_grandparent_gate(test_string3)
    with pytest.raises(GateNotProvidedError):
        find_grandparent_gate(test_string4)

def test_find_parent_population():
    test_string1 = f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}T_cells"
    test_string2 = "root"
    test_string3 = f"root{GATE_SEPARATOR}singlets"
    test_string4 = ""

    assert find_parent_population(test_string1) == "singlets"
    assert find_parent_population(test_string3) == "root"
    with pytest.raises(GateNotProvidedError):
        find_parent_population(test_string4)
    with pytest.raises(ExhaustedHierarchyError):
        find_parent_population(test_string2)

def test_find_grandparent_population():
    test_string1 = f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}T_cells"
    test_string2 = "root"
    test_string3 = f"root{GATE_SEPARATOR}singlets"
    test_string4 = ""
    
    assert find_grandparent_population(test_string1) == "root"
    with pytest.raises(ExhaustedHierarchyError):
        find_grandparent_population(test_string2)
    with pytest.raises(ExhaustedHierarchyError):
        find_grandparent_population(test_string3)
    with pytest.raises(GateNotProvidedError):
        find_grandparent_population(test_string4)

def test_find_parents_recursively():
    test_string1 = f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}T_cells"
    test_string2 = "root"
    test_string3 = f"root{GATE_SEPARATOR}singlets"
    test_string4 = ""
    
    parent_list = find_parents_recursively(test_string1)
    assert f"root{GATE_SEPARATOR}singlets" in parent_list
    assert "root" in parent_list
    with pytest.raises(ExhaustedHierarchyError):
        find_parents_recursively(test_string2)
    parent_list = find_parents_recursively(test_string3)
    assert parent_list ==  ["root"]
    with pytest.raises(GateNotProvidedError):
        find_parents_recursively(test_string4)

def test_close_polygon_gate_coordinates():
    coordinate_array = np.array([[1,2],[3,4]])
    assert np.array_equal(close_polygon_gate_coordinates(coordinate_array),
                          np.array([[1,2],[3,4],[1,2]]))

def test_ifelse():
    assert ifelse(True, "right", "wrong") == "right"
    assert ifelse(False, "right", "wrong") == "wrong"
    # sourcery skip: use-isna
    assert ifelse(np.nan == np.nan, "right", "wrong") == "wrong"

def test_flatten_nested_list():
    test_list = [["some", "strings", 2], ["some", "other", "ints"]]
    assert flatten_nested_list(test_list) == ["some", "strings", 2, "some", "other", "ints"]



def test_subset_stained_samples(): pass
def test_subset_unstained_samples(): pass
def test_fetch_fluo_channels(): pass
def test_subset_channels(): pass
def test_subset_gate(): pass
def test_annotate_metadata_samplewise(): pass
def test_contains_only_fluo(): pass
def test_get_idx_loc(): pass
def test_remove_unnamed_channels(): pass
def test_get_filename(): pass
def test_create_comparisons(): pass
def test_convert_gate_to_obs(): pass
def test_convert_gates_to_obs(): pass


# QUICKFIND: Subset fluo channels
@pytest.fixture
def mock_anndata_subset():
    return AnnData(
        X = np.zeros((7,7)),
        var = pd.DataFrame(
            data = {
                "pns": ["FSC-A", "FSC-H", "SSC-A", "SSC-H", "CD3", "time", "CD8"],
                "type": ["scatter", "scatter", "scatter", "scatter", "fluo", "time", "fluo"]
            },
            index = ["FSC-A", "FSC-H", "SSC-A", "SSC-H", "BUV 395-A", "Time", "APC-Cy7-A"]
        )
    )

def test_fluo_channel_copy_function(mock_anndata_subset):
    subset_fluo_channels(mock_anndata_subset,
                         copy = False)
    assert mock_anndata_subset.shape[1] == 2

def test_fluo_channel_subset(mock_anndata_subset):
    dataset = subset_fluo_channels(mock_anndata_subset,
                                   copy = True)
    assert dataset.shape[1] == 2

def test_fluo_channel_subset_2(mock_anndata_subset):
    dataset = subset_fluo_channels(mock_anndata_subset, copy = True)
    assert "BUV 395-A" in dataset.var.index
    assert "APC-Cy7-A" in dataset.var.index
    assert "Time" not in dataset.var.index


@pytest.fixture
def mock_anndata_gate_subset() -> AnnData:
    return AnnData(
        X = np.zeros((7,7), dtype = np.float64),
        var = pd.DataFrame(
            data = {
                "pns": ["FSC-A", "FSC-H", "SSC-A", "SSC-H", "CD3", "time", "CD8"],
                "type": ["scatter", "scatter", "scatter", "scatter", "fluo", "time", "fluo"]
            },
            index = ["FSC-A", "FSC-H", "SSC-A", "SSC-H", "BUV 395-A", "Time", "APC-Cy7-A"]
        ),
        uns = {"gating_cols": pd.Index(["root/singlets", "root/singlets/T_cells"])},
        obsm = {"gating": np.array([[1,1,1,1,1,0,0], [1,1,1,0,0,0,0]], dtype = np.float64).T},
        dtype = np.int32
    )

def test_gate_subset_copy_function(mock_anndata_gate_subset):
    subset_gate(mock_anndata_gate_subset,
                gate = "T_cells",
                copy = False)
    assert mock_anndata_gate_subset.shape[0] == 3

def test_gate_subset_copy_function_2(mock_anndata_gate_subset):
    subset_gate(mock_anndata_gate_subset,
                gate = "singlets",
                copy = False)
    assert mock_anndata_gate_subset.shape[0] == 5

def test_gate_subset_return(mock_anndata_gate_subset):
    dataset = subset_gate(mock_anndata_gate_subset,
                          gate = "T_cells",
                          copy = True)
    assert dataset.shape[0] == 3

def test_gate_subset_return(mock_anndata_gate_subset):
    dataset = subset_gate(mock_anndata_gate_subset,
                          gate = "singlets",
                          copy = True)
    assert dataset.shape[0] == 5

def test_gate_subset_gate_path(mock_anndata_gate_subset):
    dataset = subset_gate(mock_anndata_gate_subset,
                          gate_path = "root/singlets/T_cells",
                          copy = True)
    assert dataset.shape[0] == 3

def test_gate_subset_gate_path(mock_anndata_gate_subset):
    dataset = subset_gate(mock_anndata_gate_subset,
                          gate_path = "root/singlets",
                          copy = True)
    assert dataset.shape[0] == 5

def test_gate_subset_gate_path_as_gate(mock_anndata_gate_subset):
    dataset = subset_gate(mock_anndata_gate_subset,
                          gate = "root/singlets/T_cells",
                          copy = True)
    assert dataset.shape[0] == 3

def test_gate_subset_gate_path_as_gate(mock_anndata_gate_subset):
    dataset = subset_gate(mock_anndata_gate_subset,
                          gate = "root/singlets",
                          copy = True)
    assert dataset.shape[0] == 5

def test_gate_subset_wrong_inputs(mock_anndata_gate_subset):
    with pytest.raises(TypeError):
        subset_gate(mock_anndata_gate_subset)
