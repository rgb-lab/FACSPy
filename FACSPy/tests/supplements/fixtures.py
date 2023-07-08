import pytest
import pandas as pd

@pytest.fixture
def mock_panel_with_prefixes():
    return pd.DataFrame(
        {
            "fcs_colname": [
                "FSC-A",
                "FSC-H",
                "APC-H7-A",
                "FJComp-PE-CF594-A",
                "Comp-PE-A"
            ],
            "antigens": [
                "FSC-A",
                "SSC-A",
                "CD16",
                "Live/Dead",
                "mitoTracker"
            ]

        }
    )

@pytest.fixture
def mock_cofactors_with_prefixes():
    return pd.DataFrame(
        {
            "fcs_colname": [
                "FSC-A",
                "SSC-A",
                "APC-H7-A",
                "FJComp-PE-CF594-A",
                "Comp-PE-A",
            ],
            "cofactors": list(range(5)),
        }
    )

@pytest.fixture
def mock_metadata_with_factors():
    return pd.DataFrame(
        {
            "sample_ID": list(range(5)),
            "file_name": [f"file{i}" for i in range(5)],
            "some": ["some" for _ in range(5)],
            "metadata": ["metadata" for _ in range(5)],
            "factors": ["factors" for _ in range(5)]
        }
    )

@pytest.fixture
def mock_cofactors_correct():
    return pd.DataFrame(
        {
            "fcs_colname": [
                "FSC-A",
                "SSC-A",
                "APC-H7-A",
                "PE-CF594-A",
                "PE-A",
            ],
            "cofactors": list(range(5)),
        }
    )

@pytest.fixture
def mock_cofactors_np_array_instead_of_dataframe():
    df = pd.DataFrame(
        {
            "fcs_colname": [
                "FSC-A",
                "SSC-A",
                "APC-H7-A",
                "PE-CF594-A",
                "PE-A",
            ],
            "cofactors": list(range(5)),
        }
    )
    return df.to_numpy()

@pytest.fixture
def mock_cofactors_wrong_colname():
    return pd.DataFrame(
        {
            "fcs_colnamez": [
                "FSC-A",
                "SSC-A",
                "APC-H7-A",
                "FJComp-PE-CF594-A",
                "Comp-PE-A",
            ],
            "cofactor": list(range(5)),
        }
    )

@pytest.fixture
def mock_metadata_correct():
    return pd.DataFrame(
        {
            "sample_ID": list(range(2)),
            "file_name": ["some_file.fcs", "some_other_file.fcs"]
        }
    )

@pytest.fixture
def mock_metadata_wrong_colnames():
    return pd.DataFrame(
        {
            "sample_ideas": list(range(2)),
            "file_namez": ["some_file.fcs", "some_other_file.fcs"]
        }
    )
    
@pytest.fixture
def mock_metadata_np_array_instead_of_dataframe():
    df = pd.DataFrame(
        {
            "sample_id": list(range(2)),
            "file_name": ["some_file.fcs", "some_other_file.fcs"]
        }
    )
    return df.to_numpy()

@pytest.fixture
def mock_panel_correct():
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
def mock_panel_nparray_instead_of_dataframe():
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
            "antigenitos": ["FSC-A", "SSC-A", "CD16", "Live/Dead", "mitoTracker"],
        }
    )