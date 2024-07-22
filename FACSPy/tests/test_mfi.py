import pytest
from anndata import AnnData
import pandas as pd

import FACSPy as fp
from FACSPy.tools._mfi import mfi
from FACSPy._utils import _find_gate_path_of_gate


def test_aliasing(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.tl.mfi(mock_dataset_downsampled)
    assert "mfi_sample_ID_compensated" in mock_dataset_downsampled.uns

def test_copying(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    x = fp.tl.mfi(mock_dataset_downsampled,
                  copy = True)
    assert isinstance(x, AnnData)

    x = fp.tl.mfi(mock_dataset_downsampled,
                  copy = False)
    assert x is None

def test_mfi_function(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    mfi(mock_dataset_downsampled,
        groupby = "sample_ID",
        layer = ["compensated", "transformed"])
    assert "mfi_sample_ID_compensated" in mock_dataset_downsampled.uns
    assert "mfi_sample_ID_transformed" in mock_dataset_downsampled.uns

def test_only_fluo(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    mfi(mock_dataset_downsampled,
        use_only_fluo = False)
    mfi_frame = mock_dataset_downsampled.uns["mfi_sample_ID_compensated"]
    assert "FSC-A" in mfi_frame.columns

def test_only_fluo_2(mock_dataset_downsampled: AnnData):
    adata = mock_dataset_downsampled.copy()
    assert "mfi_sample_ID_compensated" not in adata.uns
    mfi(adata,
        use_only_fluo = True)
    mfi_frame = adata.uns["mfi_sample_ID_compensated"]
    assert "FSC-A" not in mfi_frame.columns

def test_dataframe_contents(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    mfi(mock_dataset_downsampled,
        use_only_fluo = False)
    mfi_frame: pd.DataFrame = mock_dataset_downsampled.uns["mfi_sample_ID_compensated"]
    assert list(mfi_frame.index.names) == ["sample_ID", "gate"]
    assert all([name in mfi_frame.columns for name in mock_dataset_downsampled.var_names.tolist()])

def test_groupby_parameter(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    mfi(mock_dataset_downsampled,
        groupby = "condition1",
        use_only_fluo = False)
    assert "mfi_condition1_compensated" in mock_dataset_downsampled.uns
    mfi_frame: pd.DataFrame = mock_dataset_downsampled.uns["mfi_condition1_compensated"]
    for index_entry in ["sample_ID", "condition1", "gate"]:
        assert index_entry in list(mfi_frame.index.names)
    
def test_error_for_wrong_metric(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    with pytest.raises(NotImplementedError):
        mfi(mock_dataset_downsampled,
            method = "random_method")

def test_layer_setting(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    mock_dataset_downsampled.layers["test_layer"] = mock_dataset_downsampled.layers["compensated"].copy()
    mfi(mock_dataset_downsampled,
        layer = "test_layer")
    assert "mfi_sample_ID_test_layer" in mock_dataset_downsampled.uns

def test_correct_mfi_mean(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    specific_gate_adata = fp.subset_gate(mock_dataset_downsampled, "live", copy = True)
    full_gate_path = _find_gate_path_of_gate(mock_dataset_downsampled, "live")
    df = specific_gate_adata.to_df(layer = "compensated")
    df["sample_ID"] = specific_gate_adata.obs["sample_ID"]
    mean_df = df.groupby("sample_ID").mean()
    mean_df["gate"] = full_gate_path
    mean_df = mean_df.set_index("gate", append = True)

    mfi(mock_dataset_downsampled,
        use_only_fluo = False,
        method = "mean")
    
    facspy_calculated: pd.DataFrame = mock_dataset_downsampled.uns["mfi_sample_ID_compensated"]
    gate_specific_data: pd.DataFrame = facspy_calculated.loc[facspy_calculated.index.get_level_values("gate") == full_gate_path,:]

    assert gate_specific_data.equals(mean_df)

def test_correct_mfi_median(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    specific_gate_adata = fp.subset_gate(mock_dataset_downsampled, "live", copy = True)
    full_gate_path = _find_gate_path_of_gate(mock_dataset_downsampled, "live")
    df = specific_gate_adata.to_df(layer = "compensated")
    df["sample_ID"] = specific_gate_adata.obs["sample_ID"]
    mean_df = df.groupby("sample_ID").median()
    mean_df["gate"] = full_gate_path
    mean_df = mean_df.set_index("gate", append = True)

    mfi(mock_dataset_downsampled,
        use_only_fluo = False,
        method = "median")
    
    facspy_calculated: pd.DataFrame = mock_dataset_downsampled.uns["mfi_sample_ID_compensated"]
    gate_specific_data: pd.DataFrame = facspy_calculated.loc[facspy_calculated.index.get_level_values("gate") == full_gate_path,:]

    assert gate_specific_data.equals(mean_df)

def test_settings_save(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    mfi(mock_dataset_downsampled,
        use_only_fluo = False,
        method = "median")
    assert "settings" in mock_dataset_downsampled.uns
    settings = mock_dataset_downsampled.uns["settings"]
    assert "_mfi_sample_ID_compensated" in settings
    assert settings["_mfi_sample_ID_compensated"]["groupby"] == "sample_ID"
    assert settings["_mfi_sample_ID_compensated"]["use_only_fluo"] == False
    assert settings["_mfi_sample_ID_compensated"]["method"] == "median"
    assert settings["_mfi_sample_ID_compensated"]["layer"] == "compensated"






