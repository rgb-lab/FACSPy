import pytest
from ...utils import GATE_SEPARATOR
from ...utils import *
from ...exceptions.utils import GateNotProvidedError, ExhaustedHierarchyError

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

from sklearn.preprocessing import MinMaxScaler