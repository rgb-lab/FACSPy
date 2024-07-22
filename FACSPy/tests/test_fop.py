from anndata import AnnData
import pandas as pd

import FACSPy as fp
from FACSPy.tools._fop import fop

def test_aliasing(mock_dataset_downsampled_with_cofactors: AnnData):
    mock_dataset_downsampled_with_cofactors = mock_dataset_downsampled_with_cofactors.copy()
    fp.tl.fop(mock_dataset_downsampled_with_cofactors)
    assert "fop_sample_ID_compensated" in mock_dataset_downsampled_with_cofactors.uns

def test_copying(mock_dataset_downsampled_with_cofactors: AnnData):
    mock_dataset_downsampled_with_cofactors = mock_dataset_downsampled_with_cofactors.copy()
    x = fp.tl.fop(mock_dataset_downsampled_with_cofactors,
                  copy = True)
    assert isinstance(x, AnnData)

    x = fp.tl.fop(mock_dataset_downsampled_with_cofactors,
                  copy = False)
    assert x is None

def test_fop_function(mock_dataset_downsampled_with_cofactors: AnnData):
    mock_dataset_downsampled_with_cofactors = mock_dataset_downsampled_with_cofactors.copy()
    fop(mock_dataset_downsampled_with_cofactors,
        groupby = "sample_ID",
        layer = ["compensated", "transformed"])
    assert "fop_sample_ID_compensated" in mock_dataset_downsampled_with_cofactors.uns
    assert "fop_sample_ID_transformed" in mock_dataset_downsampled_with_cofactors.uns

def test_only_fluo(mock_dataset_downsampled_with_cofactors: AnnData):
    mock_dataset_downsampled_with_cofactors = mock_dataset_downsampled_with_cofactors.copy()
    fop(mock_dataset_downsampled_with_cofactors,
        use_only_fluo = False)
    fop_frame = mock_dataset_downsampled_with_cofactors.uns["fop_sample_ID_compensated"]
    assert "FSC-A" in fop_frame.columns

def test_only_fluo_2(mock_dataset_downsampled_with_cofactors: AnnData):
    mock_dataset_downsampled_with_cofactors = mock_dataset_downsampled_with_cofactors.copy()
    fop(mock_dataset_downsampled_with_cofactors,
        use_only_fluo = True)
    fop_frame: pd.DataFrame = mock_dataset_downsampled_with_cofactors.uns["fop_sample_ID_compensated"]
    assert "FSC-A" not in fop_frame.columns

def test_dataframe_contents(mock_dataset_downsampled_with_cofactors: AnnData):
    mock_dataset_downsampled_with_cofactors = mock_dataset_downsampled_with_cofactors.copy()
    fop(mock_dataset_downsampled_with_cofactors,
        use_only_fluo = False)
    fop_frame: pd.DataFrame = mock_dataset_downsampled_with_cofactors.uns["fop_sample_ID_compensated"]
    assert list(fop_frame.index.names) == ["sample_ID", "gate"]
    assert all([name in fop_frame.columns for name in mock_dataset_downsampled_with_cofactors.var_names.tolist()])

def test_groupby_parameter(mock_dataset_downsampled_with_cofactors: AnnData):
    mock_dataset_downsampled_with_cofactors = mock_dataset_downsampled_with_cofactors.copy()
    fop(mock_dataset_downsampled_with_cofactors,
        groupby = "condition1",
        use_only_fluo = False)
    assert "fop_condition1_compensated" in mock_dataset_downsampled_with_cofactors.uns
    fop_frame: pd.DataFrame = mock_dataset_downsampled_with_cofactors.uns["fop_condition1_compensated"]
    for index_entry in ["sample_ID", "condition1", "gate"]:
        assert index_entry in list(fop_frame.index.names)
    
def test_layer_setting(mock_dataset_downsampled_with_cofactors: AnnData):
    mock_dataset_downsampled_with_cofactors = mock_dataset_downsampled_with_cofactors.copy()
    mock_dataset_downsampled_with_cofactors.layers["test_layer"] = mock_dataset_downsampled_with_cofactors.layers["compensated"].copy()
    fop(mock_dataset_downsampled_with_cofactors,
        layer = "test_layer")
    assert "fop_sample_ID_test_layer" in mock_dataset_downsampled_with_cofactors.uns

def test_cuttoff(mock_dataset_downsampled_with_cofactors: AnnData):
    mock_dataset_downsampled_with_cofactors = mock_dataset_downsampled_with_cofactors.copy()
    fop(mock_dataset_downsampled_with_cofactors,
        cutoff = 1000)
    assert "fop_sample_ID_compensated" in mock_dataset_downsampled_with_cofactors.uns

def test_cuttoff_2(mock_dataset_downsampled_with_cofactors: AnnData):
    mock_dataset_downsampled_with_cofactors = mock_dataset_downsampled_with_cofactors.copy()
    fop(mock_dataset_downsampled_with_cofactors,
        cutoff = [90+i for i in range(len(mock_dataset_downsampled_with_cofactors.var_names))])
    assert "fop_sample_ID_compensated" in mock_dataset_downsampled_with_cofactors.uns

def test_settings_save(mock_dataset_downsampled_with_cofactors: AnnData):
    mock_dataset_downsampled_with_cofactors = mock_dataset_downsampled_with_cofactors.copy()
    fop(mock_dataset_downsampled_with_cofactors,
        use_only_fluo = False)
        
    assert "settings" in mock_dataset_downsampled_with_cofactors.uns
    settings = mock_dataset_downsampled_with_cofactors.uns["settings"]
    assert "_fop_sample_ID_compensated" in settings
    assert settings["_fop_sample_ID_compensated"]["groupby"] == "sample_ID"
    assert settings["_fop_sample_ID_compensated"]["use_only_fluo"] == False
    assert all(settings["_fop_sample_ID_compensated"]["cutoff"] == mock_dataset_downsampled_with_cofactors.var["cofactors"].to_numpy())
    assert settings["_fop_sample_ID_compensated"]["layer"] == "compensated"





