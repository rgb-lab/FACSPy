import pytest

from ..utils import find_parent, GATE_SEPARATOR



def test_parent_function():
    x = find_parent("root/singlets/T_cells")
    assert x = "root/singlets"