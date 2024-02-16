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
