import pytest
import os

from anndata import AnnData
import pandas as pd
import FACSPy as fp
from FACSPy.synchronization._var_sync import _sync_panel_from_var, _sync_var_from_panel
from FACSPy.dataset._supplements import Metadata, Panel
from FACSPy.dataset._workspaces import FlowJoWorkspace

WSP_FILE_PATH = "FACSPy/_resources/"
WSP_FILE_NAME = "test_wsp.wsp"

def create_supplement_objects():
    INPUT_DIRECTORY = "FACSPy/_resources/test_suite_dataset"
    panel = Panel(os.path.join(INPUT_DIRECTORY, "panel.txt"))
    metadata = Metadata(os.path.join(INPUT_DIRECTORY, "metadata_test_suite.csv"))
    workspace = FlowJoWorkspace(os.path.join(INPUT_DIRECTORY, "test_suite.wsp"))
    return INPUT_DIRECTORY, panel, metadata, workspace

@pytest.fixture
def mock_dataset():
    input_directory, panel, metadata, workspace = create_supplement_objects()
    adata = fp.create_dataset(input_directory = input_directory,
                              panel = panel,
                              metadata = metadata,
                              workspace = workspace,
                              subsample_fcs_to = 100)
    adata.layers["transformed"] = adata.layers["compensated"].copy()
    fp.settings.default_layer = "compensated"
    fp.tl.mfi(adata,
              use_only_fluo = False)
    fp.tl.mds_samplewise(adata)
    return adata

def test_sync_var_names_from_var_appended_scatters(mock_dataset: AnnData):
    """the panel does not contain scatter channels, these should be appended"""
    var_names = mock_dataset.var_names
    subset = mock_dataset[:, var_names]
    _sync_panel_from_var(subset)
    panel_frame: pd.DataFrame = subset.uns["panel"].dataframe
    fcs_colnames: pd.Series = panel_frame["fcs_colname"].tolist()
    antigens = panel_frame["antigens"].tolist()
    assert all(k in fcs_colnames for k in subset.var["pnn"].tolist())
    assert all(k in antigens for k in subset.var["pns"].tolist())
    assert all(k in subset.var["pnn"].tolist() for k in fcs_colnames)
    assert all(k in subset.var["pns"].tolist() for k in antigens)
    # specific check for Scatter and time channel
    assert all(k in fcs_colnames for k in ["FSC-A", "FSC-H", "FSC-W",
                                           "SSC-A", "SSC-H", "SSC-W",
                                           "Time"])

def test_sync_var_names_from_var_removed_channels(mock_dataset: AnnData):
    """we test simultaneously if scatter channels are appended and the subset
    channels are excluded"""
    var_names = mock_dataset.var_names
    selected_channels = var_names[:10]
    excluded_channels = var_names[10:-1]
    assert not any(k in selected_channels for k in excluded_channels)
    subset = mock_dataset[:, selected_channels]
    var_names = subset.var_names
    _sync_panel_from_var(subset)
    panel_frame: pd.DataFrame = subset.uns["panel"].dataframe
    fcs_colnames = panel_frame["fcs_colname"].tolist()
    antigens = panel_frame["antigens"].tolist()

    assert all(k in fcs_colnames for k in subset.var["pnn"].tolist())
    assert all(k in antigens for k in subset.var["pns"].tolist())
    assert all(k in subset.var["pnn"].tolist() for k in fcs_colnames)
    assert all(k in subset.var["pns"].tolist() for k in antigens)
    # specific check for Scatter and time channel
    # Time channel was the last var_name and therefore gets excluded
    assert all(k in fcs_colnames for k in ["FSC-A", "FSC-H", "FSC-W",
                                           "SSC-A", "SSC-H", "SSC-W"])

def test_sync_var_names_from_panel(mock_dataset: AnnData):
    ## the panel does not contain scatter and time
    panel_frame = mock_dataset.uns["panel"].dataframe
    channels = panel_frame["antigens"].tolist()
    subset_frame = panel_frame.loc[panel_frame["antigens"].isin(channels[:5]),:]
    mock_dataset.uns["panel"] = Panel(panel = subset_frame)
    _sync_var_from_panel(mock_dataset)
    var_names = mock_dataset.var_names.tolist()
    assert all(k in var_names for k in ["FSC-A", "FSC-H", "FSC-W",
                                        "SSC-A", "SSC-H", "SSC-W",
                                        "Time"])
    assert mock_dataset.var[mock_dataset.var["type"] == "fluo"].shape[0] == 5
