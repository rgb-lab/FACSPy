import pytest
import pandas as pd
import numpy as np
from anndata import AnnData
import anndata as ad
import FACSPy as fp
from FACSPy._utils import (GATE_SEPARATOR,
                           _check_gate_name,
                           _check_gate_path,
                           _is_parent,
                           _find_gate_path_of_gate,
                           _extract_partial_gate_path_end,
                           _extract_partial_gate_path_start,
                           _find_gate_indices,
                           equalize_groups,
                           _find_parent_population,
                           _find_current_population,
                           _find_parent_gate,
                           _find_parents_recursively,
                           _find_grandparent_gate,
                           _find_grandparent_population,
                           _close_polygon_gate_coordinates,
                           _flatten_nested_list,
                           subset_fluo_channels,
                           subset_gate,
                           _default_layer,
                           _default_gate,
                           _default_gate_and_default_layer)

from FACSPy.exceptions._utils import (GateNotProvidedError,
                                      ExhaustedHierarchyError)
from FACSPy.exceptions._exceptions import (GateAmbiguityError,
                                           GateNotFoundError,
                                           PopulationAsGateError,
                                           ExhaustedGatePathError,
                                           GateNameError)

from FACSPy.dataset._supplements import Panel, Metadata
from FACSPy.dataset._workspaces import FlowJoWorkspace
import os
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
    adata.layers["transformed"] = adata.layers["compensated"].copy()
    fp.tl.mfi(adata)
    return adata

@pytest.fixture
def mock_dataset_incomplete_panel() -> AnnData:
    input_directory, panel, metadata, workspace = create_supplement_objects()
    panel.select_channels([ch for ch in panel.get_antigens() if ch not in ["CXCR2", "TNFR2"]])
    adata = fp.create_dataset(input_directory = input_directory,
                              panel = panel,
                              metadata = metadata,
                              workspace = workspace,
                              subsample_fcs_to = 100)
    adata.layers["transformed"] = adata.layers["compensated"].copy()
    fp.tl.mfi(adata)
    return adata

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
    concatenated.uns = {"gating_cols": pd.Index([
        f"root{GATE_SEPARATOR}singlets",
        f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}T_cells",
        f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}T_cells{GATE_SEPARATOR}live",
        f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}T_cells{GATE_SEPARATOR}live{GATE_SEPARATOR}subset",
        f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}T_cells{GATE_SEPARATOR}subset"
    ])}
    return concatenated

def test_check_gate_name_valid():
    # Test valid gate names
    assert _check_gate_name("valid_gate_name") is None
    assert _check_gate_name("another_valid_gate") is None
    assert _check_gate_name("root/singlets") is None

def test_check_gate_name_invalid_start():
    # Test gate names starting with GATE_SEPARATOR
    with pytest.raises(GateNameError):
        _check_gate_name("/invalid_start")

def test_check_gate_name_invalid_end():
    # Test gate names ending with GATE_SEPARATOR
    with pytest.raises(GateNameError):
        _check_gate_name("invalid_end/")

def test_check_gate_name_empty():
    # Test empty gate name
    with pytest.raises(GateNotProvidedError):
        _check_gate_name("")

def test_check_gate_name_whitespace():
    # Test gate name with only whitespace
    assert _check_gate_name("   ") is None

def test_check_gate_name_whitespace_start():
    # Test gate name starting with whitespace
    assert _check_gate_name("   gate") is None

def test_check_gate_name_whitespace_end():
    # Test gate name ending with whitespace
    assert _check_gate_name("gate   ") is None

def test_check_gate_name_whitespace_both_ends():
    # Test gate name with whitespace both ends
    assert _check_gate_name("   gate   ") is None

def test_check_gate_path_valid():
    # Test valid gate paths
    assert _check_gate_path("root/singlets/T_cells") is None
    assert _check_gate_path("root/singlets") is None

def test_check_gate_path_invalid_name():
    # Test invalid gate name
    with pytest.raises(GateNameError):
        _check_gate_path("/invalid_name")

def test_check_gate_path_population_as_gate():
    # Test gate path with population as gate
    with pytest.raises(PopulationAsGateError):
        _check_gate_path("population_as_gate")

def test_check_gate_path_empty():
    # Test empty gate path
    with pytest.raises(GateNotProvidedError):
        _check_gate_path("")

def test_check_gate_path_whitespace():
    # Test gate path with only whitespace
    with pytest.raises(PopulationAsGateError):
        _check_gate_path("   ")

def test_check_gate_path_whitespace_start():
    # Test gate path starting with whitespace
    with pytest.raises(PopulationAsGateError):
        _check_gate_path("   gate")

def test_check_gate_path_whitespace_end():
    # Test gate path ending with whitespace
    with pytest.raises(PopulationAsGateError):
        _check_gate_path("gate   ")

def test_check_gate_path_whitespace_both_ends():
    # Test gate path with whitespace both ends
    with pytest.raises(PopulationAsGateError):
       _check_gate_path("   gate   ")

def test_is_parent_true():
    # Test when gate is parent
    adata = AnnData(uns={"gating_cols": pd.Index(["root/singlets", "root/singlets/T_cells"])})
    assert _is_parent(adata, "root/singlets/T_cells", "root/singlets")
    assert _is_parent(adata, "T_cells", "singlets")

def test_is_parent_false():
    # Test when gate is not parent
    adata = AnnData(uns={"gating_cols": pd.Index(["root/singlets", "root/singlets/T_cells"])})
    assert not _is_parent(adata, "root/singlets", "root/singlets/T_cells")
    assert not _is_parent(adata, "singlets", "T_cells")
    assert not _is_parent(adata, "T_cells", "T_cells")

def test_extract_partial_gate_path_start():
    test_string1 = f"root{GATE_SEPARATOR}singlets"
    test_string2 = "root"
    test_string3 = f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}live"
    test_string4 = ""
    test_string0 = "/some_gate"
    test_string01 = "some_gate/"

    with pytest.raises(GateNameError):
        _extract_partial_gate_path_start(test_string0, 1)
    with pytest.raises(GateNameError):
        _extract_partial_gate_path_start(test_string01, 1)

    assert _extract_partial_gate_path_start(test_string1, 1) == "root"
    with pytest.raises(ExhaustedGatePathError):
        _extract_partial_gate_path_start(test_string1, 4)
    with pytest.raises(PopulationAsGateError):
        _extract_partial_gate_path_start(test_string2, 1)
    assert _extract_partial_gate_path_start(test_string3, 2) == "root/singlets"
    assert _extract_partial_gate_path_start(test_string3, 3) == test_string3
    with pytest.raises(GateNotProvidedError):
        _extract_partial_gate_path_start(test_string4, 3)

def test_extract_partial_gate_path_end():
    test_string1 = f"root{GATE_SEPARATOR}singlets"
    test_string2 = "root"
    test_string3 = f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}live"
    test_string4 = ""
    test_string0 = "/some_gate"
    test_string01 = "some_gate/"

    with pytest.raises(GateNameError):
        _extract_partial_gate_path_end(test_string0, 1)
    with pytest.raises(GateNameError):
        _extract_partial_gate_path_end(test_string01, 1)

    assert _extract_partial_gate_path_end(test_string1, 1) == "singlets"
    with pytest.raises(ExhaustedGatePathError):
        _extract_partial_gate_path_end(test_string1, 4)
    with pytest.raises(PopulationAsGateError):
        _extract_partial_gate_path_end(test_string2, 1)
    assert _extract_partial_gate_path_end(test_string3, 2) == "singlets/live"
    assert _extract_partial_gate_path_end(test_string3, 3) == test_string3
    with pytest.raises(GateNotProvidedError):
        _extract_partial_gate_path_end(test_string4, 3)


def test_find_gate_path_of_gate(mock_anndata_gating: AnnData):
    assert _find_gate_path_of_gate(mock_anndata_gating, "T_cells") == f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}T_cells"
    assert _find_gate_path_of_gate(mock_anndata_gating, "live") == f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}T_cells{GATE_SEPARATOR}live"
    assert _find_gate_path_of_gate(mock_anndata_gating, "singlets") == f"root{GATE_SEPARATOR}singlets"
    with pytest.raises(GateNotFoundError):
        _find_gate_path_of_gate(mock_anndata_gating, "some_gate")
    with pytest.raises(GateAmbiguityError):
        _find_gate_path_of_gate(mock_anndata_gating, "subset")

def test_find_gate_indices(mock_anndata_gating):
    assert _find_gate_indices(mock_anndata_gating, _find_gate_path_of_gate(mock_anndata_gating,"T_cells")) == [1]


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

def test_equaling_sample_ID_real_dataset_n_obs_smaller_than_smallest_group(mock_dataset: AnnData):
    # smallest group is 200
    adata = mock_dataset
    equalize_groups(adata, n_obs = 100, on = "condition2")
    assert all(adata.obs.groupby("condition2").size() == 100)

def test_equaling_sample_ID_real_dataset_n_obs_equal_to_smallest_group(mock_dataset: AnnData):
    # smallest group is 200
    adata = mock_dataset
    equalize_groups(adata, n_obs = 200, on = "condition2")
    assert all(adata.obs.groupby("condition2").size() == 200)

def test_equaling_sample_ID_real_dataset_n_obs_larger_than_smallest_group(mock_dataset: AnnData):
    # smallest group is 200
    adata = mock_dataset
    with pytest.warns(UserWarning):
        equalize_groups(adata, n_obs = 300, on = "condition2")
    res = adata.obs.groupby("condition2").size()
    assert res.loc[res.index == "x"].iloc[0] == 300
    assert res.loc[res.index == "y"].iloc[0] == 200

def test_equalizing_groups_as_view(mock_dataset: AnnData):
    adata = mock_dataset
    equalized = equalize_groups(adata, n_obs = 200, on = "condition2", as_view = True)
    assert equalized.is_view

def test_equalizing_groups_copy(mock_dataset: AnnData):
    adata = mock_dataset
    equalized = equalize_groups(adata, n_obs = 200, on = "condition2", copy = True)
    assert not equalized.is_view
    assert isinstance(equalized, AnnData)

def test_equalizing_groups_copy_false(mock_dataset: AnnData):
    adata = mock_dataset
    equalized = equalize_groups(adata, n_obs = 200, on = "condition2", copy = False)
    assert equalized is None



def test_equalizing_sample_ID(mock_anndata):
    equalize_groups(mock_anndata, fraction = 0.1)
    assert mock_anndata.shape[0] == 3 ## smallest group has 10

def test_equalizing_something(mock_anndata):
    equalize_groups(mock_anndata, n_obs = 10)
    assert mock_anndata.shape[0] == 30

def test_too_much_to_sample_per_group(mock_anndata):
    with pytest.warns(UserWarning):
        equalize_groups(mock_anndata, n_obs = 11)

def test_value_error(mock_anndata):
    with pytest.raises(ValueError):
        equalize_groups(mock_anndata, fraction = 0.1, n_obs = 100)

def test_too_much_to_sample(mock_anndata):
    with pytest.warns(UserWarning):
        equalize_groups(mock_anndata, n_obs = 10_000)

# QUICKFIND: Population
def test_find_current_population():
    test_string1 = f"root{GATE_SEPARATOR}singlets"
    test_string2 = "root"
    test_string3 = f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}live"
    test_string4 = ""

    assert _find_current_population(test_string1) == "singlets"
    assert _find_current_population(test_string2) == "root"
    assert _find_current_population(test_string3) == "live"
    with pytest.raises(GateNotProvidedError):
        _find_current_population(test_string4)

def test_find_parent_gate():
    test_string1 = f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}T_cells"
    test_string2 = "root"
    test_string3 = f"root{GATE_SEPARATOR}singlets"
    test_string4 = ""

    assert _find_parent_gate(test_string1) == f"root{GATE_SEPARATOR}singlets"
    assert _find_parent_gate(test_string3) ==  "root"
    with pytest.raises(ExhaustedHierarchyError):
        _find_parent_gate(test_string2)
    with pytest.raises(GateNotProvidedError):
        _find_parent_gate(test_string4)

def test_find_grandparent_gate():
    test_string1 = f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}T_cells"
    test_string2 = "root"
    test_string3 = f"root{GATE_SEPARATOR}singlets"
    test_string4 = ""

    assert _find_grandparent_gate(test_string1) == "root"
    with pytest.raises(ExhaustedHierarchyError):
        _find_grandparent_gate(test_string2)
    with pytest.raises(ExhaustedHierarchyError):
        _find_grandparent_gate(test_string3)
    with pytest.raises(GateNotProvidedError):
        _find_grandparent_gate(test_string4)

def test_find_parent_population():
    test_string1 = f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}T_cells"
    test_string2 = "root"
    test_string3 = f"root{GATE_SEPARATOR}singlets"
    test_string4 = ""

    assert _find_parent_population(test_string1) == "singlets"
    assert _find_parent_population(test_string3) == "root"
    with pytest.raises(GateNotProvidedError):
        _find_parent_population(test_string4)
    with pytest.raises(ExhaustedHierarchyError):
        _find_parent_population(test_string2)

def test_find_grandparent_population():
    test_string1 = f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}T_cells"
    test_string2 = "root"
    test_string3 = f"root{GATE_SEPARATOR}singlets"
    test_string4 = ""
    
    assert _find_grandparent_population(test_string1) == "root"
    with pytest.raises(ExhaustedHierarchyError):
        _find_grandparent_population(test_string2)
    with pytest.raises(ExhaustedHierarchyError):
        _find_grandparent_population(test_string3)
    with pytest.raises(GateNotProvidedError):
        _find_grandparent_population(test_string4)

def test_find_parents_recursively():
    test_string1 = f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}T_cells"
    test_string2 = "root"
    test_string3 = f"root{GATE_SEPARATOR}singlets"
    test_string4 = ""
    
    parent_list = _find_parents_recursively(test_string1)
    assert f"root{GATE_SEPARATOR}singlets" in parent_list
    assert "root" in parent_list
    with pytest.raises(ExhaustedHierarchyError):
        _find_parents_recursively(test_string2)
    parent_list = _find_parents_recursively(test_string3)
    assert parent_list ==  ["root"]
    with pytest.raises(GateNotProvidedError):
        _find_parents_recursively(test_string4)

def test_close_polygon_gate_coordinates():
    coordinate_array = np.array([[1,2],[3,4]])
    assert np.array_equal(_close_polygon_gate_coordinates(coordinate_array),
                          np.array([[1,2],[3,4],[1,2]]))

def test_flatten_nested_list():
    test_list = [["some", "strings", 2], ["some", "other", "ints"]]
    assert _flatten_nested_list(test_list) == ["some", "strings", 2, "some", "other", "ints"]


def test_subset_gate_by_population(mock_dataset: AnnData):
    adata = mock_dataset
    gate = "live"
    gates: list[str] = adata.uns["gating_cols"].tolist()
    gate_path = _find_gate_path_of_gate(adata, gate)
    parents = _find_parents_recursively(gate_path)
    gate_indices = []
    gate_matrix = adata.obsm["gating"].toarray()
    target_shape = gate_matrix[:, gates.index(gate_path)].sum()
    for _gate in parents + [gate_path]:
        if _gate == "root":
            continue
        gate_indices.append(gates.index(_gate))
    fp.subset_gate(adata, gate)
    assert adata.shape[0] == target_shape
    gates = adata.obsm["gating"].toarray()
    # all parents and the gate we subset to have to be all positive.
    assert all(np.sum(gates, axis = 0)[gate_indices] == adata.shape[0])

def test_subset_gate_by_full_gate_path(mock_dataset: AnnData):
    adata = mock_dataset
    gate = 'root/FSC_SSC/FSC_singlets/SSC_singlets/live'
    gates: list[str] = adata.uns["gating_cols"].tolist()
    gate_path = _find_gate_path_of_gate(adata, gate)
    parents = _find_parents_recursively(gate_path)
    gate_indices = []
    gate_matrix = adata.obsm["gating"].toarray()
    target_shape = gate_matrix[:, gates.index(gate_path)].sum()
    for _gate in parents + [gate_path]:
        if _gate == "root":
            continue
        gate_indices.append(gates.index(_gate))
    fp.subset_gate(adata, gate)
    assert adata.shape[0] == target_shape
    gates = adata.obsm["gating"].toarray()
    # all parents and the gate we subset to have to be all positive.
    assert all(np.sum(gates, axis = 0)[gate_indices] == adata.shape[0])

def test_subset_gate_by_partial_gate_path(mock_dataset: AnnData):
    adata = mock_dataset
    gate = 'FSC_singlets/SSC_singlets/live'
    gates: list[str] = adata.uns["gating_cols"].tolist()
    gate_path = _find_gate_path_of_gate(adata, gate)
    parents = _find_parents_recursively(gate_path)
    gate_indices = []
    gate_matrix = adata.obsm["gating"].toarray()
    target_shape = gate_matrix[:, gates.index(gate_path)].sum()
    for _gate in parents + [gate_path]:
        if _gate == "root":
            continue
        gate_indices.append(gates.index(_gate))
    fp.subset_gate(adata, gate)
    assert adata.shape[0] == target_shape
    gates = adata.obsm["gating"].toarray()
    # all parents and the gate we subset to have to be all positive.
    assert all(np.sum(gates, axis = 0)[gate_indices] == adata.shape[0])

def test_gate_subset_as_view(mock_dataset: AnnData):
    adata = mock_dataset
    subset_adata = fp.subset_gate(adata, "live", as_view = True)
    assert subset_adata.is_view

def test_gate_subset_copy(mock_dataset: AnnData):
    adata = mock_dataset
    subset_adata = fp.subset_gate(adata, "live", copy = True)
    assert subset_adata is not None
    subset_adata = fp.subset_gate(adata, "live", copy = False)
    assert subset_adata is None

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

def test_gate_subset_return_2(mock_anndata_gate_subset):
    dataset = subset_gate(mock_anndata_gate_subset,
                          gate = "singlets",
                          copy = True)
    assert dataset.shape[0] == 5

def test_gate_subset_gate_path(mock_anndata_gate_subset):
    dataset = subset_gate(mock_anndata_gate_subset,
                          gate = "root/singlets/T_cells",
                          copy = True)
    assert dataset.shape[0] == 3

def test_gate_subset_gate_path_2(mock_anndata_gate_subset):
    dataset = subset_gate(mock_anndata_gate_subset,
                          gate = "root/singlets",
                          copy = True)
    assert dataset.shape[0] == 5

def test_gate_subset_gate_path_as_gate(mock_anndata_gate_subset):
    dataset = subset_gate(mock_anndata_gate_subset,
                          gate = "root/singlets/T_cells",
                          copy = True)
    assert dataset.shape[0] == 3

def test_gate_subset_gate_path_as_gate_2(mock_anndata_gate_subset):
    dataset = subset_gate(mock_anndata_gate_subset,
                          gate = "root/singlets",
                          copy = True)
    assert dataset.shape[0] == 5

def test_gate_subset_gate_path_as_partial_gate(mock_anndata_gate_subset):
    dataset = subset_gate(mock_anndata_gate_subset,
                          gate = "singlets/T_cells",
                          copy = True)
    assert dataset.shape[0] == 3

def test_gate_subset_wrong_inputs(mock_anndata_gate_subset):
    with pytest.raises(TypeError):
        subset_gate(mock_anndata_gate_subset)

def test_subset_fluo_channels(mock_dataset: AnnData):
    adata = mock_dataset
    n_fluo_channels = adata.var[adata.var["type"] == "fluo"].shape[0]
    fp.subset_fluo_channels(adata)
    assert adata.var["type"].unique().tolist() == ["fluo"]
    assert adata.shape[1] == n_fluo_channels

def test_subset_fluo_channels_as_view(mock_dataset: AnnData):
    adata = mock_dataset
    n_fluo_channels = adata.var[adata.var["type"] == "fluo"].shape[0]
    subset_adata = fp.subset_fluo_channels(adata, as_view = True)
    assert subset_adata.is_view
    assert subset_adata.var["type"].unique().tolist() == ["fluo"]
    assert subset_adata.shape[1] == n_fluo_channels

def test_subset_fluo_channels_copy(mock_dataset: AnnData):
    adata = mock_dataset
    n_fluo_channels = adata.var[adata.var["type"] == "fluo"].shape[0]
    subset_adata = fp.subset_fluo_channels(adata, copy = True)
    assert not subset_adata.is_view
    assert subset_adata.var["type"].unique().tolist() == ["fluo"]
    assert subset_adata.shape[1] == n_fluo_channels

def test_subset_fluo_channels_copy_false(mock_dataset: AnnData):
    adata = mock_dataset
    n_fluo_channels = adata.var[adata.var["type"] == "fluo"].shape[0]
    subset_adata = fp.subset_fluo_channels(adata, copy = False)
    assert subset_adata is None
    assert adata.var["type"].unique().tolist() == ["fluo"]
    assert adata.shape[1] == n_fluo_channels

def test_remove_unnamed_channels(mock_dataset_incomplete_panel: AnnData):
    # we artificially create unnamed channels by removing them from the panel
    # We excluded TNFR2 and CXCR2
    adata = mock_dataset_incomplete_panel
    adata_shape = adata.shape
    fp.remove_unnamed_channels(adata)
    assert adata.shape[1] == adata_shape[1] - 2
    for antigen, channel, type in zip(adata.var["pns"].tolist(), adata.var["pnn"].tolist(), adata.var["type"].tolist()):
        if type == "fluo":
            assert antigen != channel
        else:
            assert antigen == channel

def test_remove_unnamed_channels_as_view(mock_dataset_incomplete_panel: AnnData):
    # we artificially create unnamed channels by removing them from the panel
    # We excluded TNFR2 and CXCR2
    adata = mock_dataset_incomplete_panel
    adata_shape = adata.shape
    subset_adata = fp.remove_unnamed_channels(adata, as_view = True)
    assert subset_adata.is_view
    assert subset_adata.shape[1] == adata_shape[1] - 2
    for antigen, channel, type in zip(subset_adata.var["pns"].tolist(), subset_adata.var["pnn"].tolist(), subset_adata.var["type"].tolist()):
        if type == "fluo":
            assert antigen != channel
        else:
            assert antigen == channel

def test_remove_unnamed_channels_copy(mock_dataset_incomplete_panel: AnnData):
    # we artificially create unnamed channels by removing them from the panel
    # We excluded TNFR2 and CXCR2
    adata = mock_dataset_incomplete_panel
    adata_shape = adata.shape
    subset_adata = fp.remove_unnamed_channels(adata, copy = True)
    assert not subset_adata.is_view
    assert subset_adata.shape[1] == adata_shape[1] - 2
    for antigen, channel, type in zip(subset_adata.var["pns"].tolist(), subset_adata.var["pnn"].tolist(), subset_adata.var["type"].tolist()):
        if type == "fluo":
            assert antigen != channel
        else:
            assert antigen == channel

def test_remove_unnamed_channels_copy_false(mock_dataset_incomplete_panel: AnnData):
    # we artificially create unnamed channels by removing them from the panel
    # We excluded TNFR2 and CXCR2
    adata = mock_dataset_incomplete_panel
    adata_shape = adata.shape
    subset_adata = fp.remove_unnamed_channels(adata, copy = False)
    assert subset_adata is None
    assert adata.shape[1] == adata_shape[1] - 2
    for antigen, channel, type in zip(adata.var["pns"].tolist(), adata.var["pnn"].tolist(), adata.var["type"].tolist()):
        if type == "fluo":
            assert antigen != channel
        else:
            assert antigen == channel

def test_remove_channel(mock_dataset: AnnData):
    adata = mock_dataset
    fp.remove_channel(adata, "CXCR2")
    assert "CXCR2" not in adata.var_names
    assert "CXCR2" not in adata.var["pns"].tolist()

def test_remove_channel_valueerror(mock_dataset: AnnData):
    adata = mock_dataset
    with pytest.raises(ValueError):
        fp.remove_channel(adata, "whatever")

def test_remove_channel_multiple(mock_dataset: AnnData):
    adata = mock_dataset
    fp.remove_channel(adata, channel = ["CXCR2", "TNFR2"])
    assert "CXCR2" not in adata.var_names
    assert "TNFR2" not in adata.var_names
    assert "CXCR2" not in adata.var["pns"].tolist()
    assert "TNFR2" not in adata.var["pns"].tolist()

def test_remove_channel_as_view(mock_dataset: AnnData):
    adata = mock_dataset
    subset_adata = fp.remove_channel(adata, "CXCR2", as_view = True)
    assert subset_adata.is_view
    assert "CXCR2" not in subset_adata.var_names
    assert "CXCR2" not in subset_adata.var["pns"].tolist()

def test_remove_channel_copy(mock_dataset: AnnData):
    adata = mock_dataset
    subset_adata = fp.remove_channel(adata, "CXCR2", copy = True)
    assert not subset_adata.is_view
    assert "CXCR2" not in subset_adata.var_names
    assert "CXCR2" not in subset_adata.var["pns"].tolist()

def test_remove_channel_copy_false(mock_dataset: AnnData):
    adata = mock_dataset
    subset_adata = fp.remove_channel(adata, "CXCR2", copy = False)
    assert subset_adata is None
    assert "CXCR2" not in adata.var_names
    assert "CXCR2" not in adata.var["pns"].tolist()

def test_convert_gate_to_obs(mock_dataset: AnnData):
    adata = mock_dataset
    fp.convert_gate_to_obs(adata, "live")
    assert "live" in adata.obs.columns
    assert all(k in adata.obs["live"].unique() for k in ["live", "other"])

def test_convert_gate_to_obs_full_gate_path(mock_dataset: AnnData):
    adata = mock_dataset
    gate = 'root/FSC_SSC/FSC_singlets/SSC_singlets/live'
    fp.convert_gate_to_obs(adata, gate)
    assert gate in adata.obs.columns
    assert all(k in adata.obs[gate].unique() for k in [gate, "other"])

def test_convert_gate_to_obs_gate_alias(mock_dataset: AnnData):
    adata = mock_dataset
    fp.settings.add_new_alias('root/FSC_SSC/FSC_singlets/SSC_singlets/live', "favorites")
    gate = "favorites"
    fp.convert_gate_to_obs(adata, gate = gate, key_added = "favorites")
    assert gate in adata.obs.columns
    assert all(k in adata.obs[gate].unique() for k in [gate, "other"])

def test_convert_gate_to_obs_partial_gate_path(mock_dataset: AnnData):
    adata = mock_dataset
    gate = 'FSC_singlets/SSC_singlets/live'
    fp.convert_gate_to_obs(adata, gate)
    assert gate in adata.obs.columns
    assert all(k in adata.obs[gate].unique() for k in [gate, "other"])

def test_convert_gate_to_obs_key_added(mock_dataset: AnnData):
    adata = mock_dataset
    fp.convert_gate_to_obs(adata, "live", key_added = "live_cells")
    assert "live_cells" in adata.obs.columns
    assert all(k in adata.obs["live_cells"].unique() for k in ["live_cells", "other"])









def test_fetch_fluo_channels(): pass
def test_subset_channels(): pass
def test_annotate_metadata_samplewise(): pass
def test_contains_only_fluo(): pass
def test_get_idx_loc(): pass
def test_get_filename(): pass
def test_create_comparisons(): pass
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
    mock_anndata_subset = subset_fluo_channels(mock_anndata_subset,
                                               as_view = True,
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

def test_rename_channel(mock_dataset: AnnData):
    adata = mock_dataset
    fp.rename_channel(adata, "CXCR4", "something")
    print(adata.var_names.tolist())
    for entry in adata.var_names.tolist():
        if entry == "something":
            print("got it.")
    assert "something" in adata.var_names.tolist()
    assert "CXCR4" not in adata.var_names.tolist()
    assert "something" in adata.var["pns"].tolist()
    assert "CXCR4" not in adata.var["pns"].tolist()
    # check if the panel has been adjusted
    assert "something" in adata.uns["panel"].dataframe["antigens"].tolist()
    assert "CXCR4" not in adata.uns["panel"].dataframe["antigens"].tolist()

def test_convert_cluster_to_gate(mock_dataset: AnnData):
    adata = mock_dataset
    adata.obs["cluster_col"] = np.random.randint(0, 5, adata.shape[0])
    cluster_idxs = adata.obs[adata.obs["cluster_col"] == 1].index.to_numpy()
    cluster_bool = adata.obs["cluster_col"] == 1
    cluster_bool = cluster_bool.to_numpy().flatten()
    fp.convert_cluster_to_gate(adata, "cluster_col", 1, "my_pop", "live")
    gate_matrix = adata.obsm["gating"]
    gates: list[str] = adata.uns["gating_cols"].tolist()
    full_gate_path = fp._utils._find_gate_path_of_gate(adata, "live") + "/my_pop"
    assert full_gate_path in gates
    gate_idx = gates.index(full_gate_path)
    assert all(gate_matrix[:, gate_idx].toarray().flatten() == cluster_bool)
    assert np.sum(gate_matrix[:, gate_idx].toarray()) == len(cluster_idxs)

def test_default_layer_decorator(mock_anndata):

    @_default_layer
    def my_func_decorated(adata: AnnData,
                          layer: str = None,
                          some: str = "some",
                          other: str = "other",
                          keyword_arg: str = "keyword_arg"):
        return layer, some, other, keyword_arg
    
    def my_func(adata: AnnData,
                layer: str = None,
                some: str = "some",
                other: str = "other",
                keyword_arg: str = "keyword_arg"):
        return layer, some, other, keyword_arg
    
    layer, some, other, keyword_arg = my_func_decorated(adata = mock_anndata,
                                                        layer = "my_layer")
    assert layer == "my_layer"
    assert some == "some"
    assert other == "other"
    assert keyword_arg == "keyword_arg"

    fp.settings.default_layer = "transformed"
    layer, some, other, keyword_arg = my_func_decorated(adata = mock_anndata)
    assert layer == "transformed"
    assert some == "some"
    assert other == "other"
    assert keyword_arg == "keyword_arg"

    layer, some, other, keyword_arg = my_func_decorated(adata = mock_anndata,
                                                        some = "some_other",
                                                        other = "actually_same")
    assert layer == "transformed"
    assert some == "some_other"
    assert other == "actually_same"
    assert keyword_arg == "keyword_arg"
     
    layer, some, other, keyword_arg = my_func(adata = mock_anndata,
                                              layer = "my_layer")
    assert layer == "my_layer"
    assert some == "some"
    assert other == "other"
    assert keyword_arg == "keyword_arg"

    layer, some, other, keyword_arg = my_func(adata = mock_anndata,
                                              some = "some_other",
                                              other = "actually_same")
    assert layer is None
    assert some == "some_other"
    assert other == "actually_same"
    assert keyword_arg == "keyword_arg"

def test_default_gate_decorator(mock_anndata):

    @_default_gate
    def my_func_decorated(adata: AnnData,
                          gate: str = None,
                          some: str = "some",
                          other: str = "other",
                          keyword_arg: str = "keyword_arg"):
        return gate, some, other, keyword_arg

    def my_func(adata: AnnData,
                gate: str = None,
                some: str = "some",
                other: str = "other",
                keyword_arg: str = "keyword_arg"):
        return gate, some, other, keyword_arg
    
    gate, some, other, keyword_arg = my_func_decorated(adata = mock_anndata,
                                                        gate = "my_gate")
    assert gate == "my_gate"
    assert some == "some"
    assert other == "other"
    assert keyword_arg == "keyword_arg"

    fp.settings.default_gate = "t_cells"
    gate, some, other, keyword_arg = my_func_decorated(adata = mock_anndata)
    assert gate == "t_cells"
    assert some == "some"
    assert other == "other"
    assert keyword_arg == "keyword_arg"

    gate, some, other, keyword_arg = my_func_decorated(adata = mock_anndata,
                                                       some = "some_other",
                                                       other = "actually_same")
    assert gate == "t_cells"
    assert some == "some_other"
    assert other == "actually_same"
    assert keyword_arg == "keyword_arg"
     
    gate, some, other, keyword_arg = my_func(adata = mock_anndata,
                                             gate = "my_gate")
    assert gate == "my_gate"
    assert some == "some"
    assert other == "other"
    assert keyword_arg == "keyword_arg"

    gate, some, other, keyword_arg = my_func(adata = mock_anndata,
                                             some = "some_other",
                                             other = "actually_same")
    assert gate is None
    assert some == "some_other"
    assert other == "actually_same"
    assert keyword_arg == "keyword_arg"

def test_default_gate_and_layerdecorator(mock_anndata):

    @_default_gate_and_default_layer
    def my_func_decorated(adata: AnnData,
                          layer: str = None,
                          gate: str = None,
                          some: str = "some",
                          other: str = "other",
                          keyword_arg: str = "keyword_arg"):
        return layer, gate, some, other, keyword_arg

    def my_func(adata: AnnData,
                layer: str = None,
                gate: str = None,
                some: str = "some",
                other: str = "other",
                keyword_arg: str = "keyword_arg"):
        return layer, gate, some, other, keyword_arg
    
    layer, gate, some, other, keyword_arg = my_func_decorated(adata = mock_anndata,
                                                              layer = "my_layer",
                                                              gate = "my_gate")
    assert layer == "my_layer"
    assert gate == "my_gate"
    assert some == "some"
    assert other == "other"
    assert keyword_arg == "keyword_arg"

    fp.settings.default_gate = "t_cells"
    fp.settings.default_layer = "transformed"
    layer, gate, some, other, keyword_arg = my_func_decorated(adata = mock_anndata)
    assert layer == "transformed"
    assert gate == "t_cells"
    assert some == "some"
    assert other == "other"
    assert keyword_arg == "keyword_arg"

    layer, gate, some, other, keyword_arg = my_func_decorated(adata = mock_anndata,
                                                              some = "some_other",
                                                              other = "actually_same")
    assert layer == "transformed"
    assert gate == "t_cells"
    assert some == "some_other"
    assert other == "actually_same"
    assert keyword_arg == "keyword_arg"
     
    layer, gate, some, other, keyword_arg = my_func(adata = mock_anndata,
                                                    layer = "my_layer",
                                                    gate = "my_gate")
    assert layer == "my_layer"
    assert gate == "my_gate"
    assert some == "some"
    assert other == "other"
    assert keyword_arg == "keyword_arg"

    layer, gate, some, other, keyword_arg = my_func(adata = mock_anndata,
                                                    some = "some_other",
                                                    other = "actually_same")
    assert layer is None
    assert gate is None
    assert some == "some_other"
    assert other == "actually_same"
    assert keyword_arg == "keyword_arg"
