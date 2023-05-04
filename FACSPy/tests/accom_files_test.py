import pytest
import pandas as pd
from ..data._accom_files import Panel, MetaData
from ..exceptions.exceptions import PanelDataTypeError, PanelFileTypeError, PanelFileNotFoundError, PanelNoInputDirectoryError, PanelCreationError
import os

# TODO: create small dummy dataset for full testing abilities

@pytest.fixture
def mock_panel():
    return pd.DataFrame(
        {
            "fcs_colname": [
                "FSC-A",
                "SSC-A",
                "APC-H7-A",
                "FJComp-PE-CF594-A",
                "Comp-PE-A",
            ],
            "antigens": ["FSC-A", "SSC-A", "CD16", "Live/Dead", "mitoTracker"],
        }
    )

@pytest.fixture
def mock_panel_nparray():
    df = pd.DataFrame({"fcs_colname": ["FSC-A", "SSC-A", "APC-H7-A", "FJComp-PE-CF594-A", "Comp-PE-A"],
                        "antigens": ["FSC-A", "SSC-A", "CD16", "Live/Dead", "mitoTracker"]})
    return df.to_numpy()

@pytest.fixture
def mock_panel_wrong_colnames():
    return pd.DataFrame(
        {
            "names": [
                "FSC-A",
                "SSC-A",
                "APC-H7-A",
                "FJComp-PE-CF594-A",
                "Comp-PE-A",
            ],
            "antigens": ["FSC-A", "SSC-A", "CD16", "Live/Dead", "mitoTracker"],
        }
    )

# def test_panel(mock_panel_wrong_colnames):
#     panel = Panel(panel = mock_panel_wrong_colnames)

def test_panel_input_parameters(mock_panel_nparray):
    with pytest.raises(PanelDataTypeError):
        _ = Panel(panel = mock_panel_nparray)

def test_panel_creation_errors():
    with pytest.raises(PanelCreationError):
        _ = Panel()
    with pytest.raises(PanelNoInputDirectoryError):
        _ = Panel(panel_from_file = "panel.txt")
    with pytest.raises(PanelFileNotFoundError):
        _ = Panel(panel_from_file = "panel.txt", inputDir = os.getcwd())
    
# def test_panel_creation(mock_panel):
#     a = Panel(panel = mock_panel)
#     assert a.source == "user_provided_data"




