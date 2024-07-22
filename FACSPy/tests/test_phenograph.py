import pytest
from anndata import AnnData
import FACSPy as fp
import phenograph as _phenograph
from FACSPy.exceptions._exceptions import InvalidScalingError


def test_invalid_scaling_error(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    with pytest.raises(InvalidScalingError):
        fp.tl.phenograph(mock_dataset_downsampled,
                         gate = "live",
                         scaling = "CustomScaler")

def test_save_settings_from_phenograph(mock_dataset_downsampled: AnnData):
    adata = mock_dataset_downsampled.copy()
    fp.tl.phenograph(adata,
                     gate = "live",
                     layer = "compensated",
                     seed = 187,
                     use_only_fluo = False,
                     scaling = None,
                     exclude = None)
    assert "settings" in adata.uns
    assert "_phenograph_live_compensated" in adata.uns["settings"]
    settings = adata.uns["settings"]["_phenograph_live_compensated"]
    assert settings["gate"] == "live"
    assert settings["layer"] == "compensated"
    assert settings["clustering_algo"] == "leiden"
    assert settings["use_only_fluo"] == False

def test_phenograph_works_as_phenograph(mock_dataset_downsampled: AnnData):
    adata = mock_dataset_downsampled.copy()
    fp.subset_gate(adata, "live")
    phenograph_data = adata.copy()
    phenograph_data.X = phenograph_data.layers["compensated"]
    comms, graph, Q = _phenograph.cluster(phenograph_data.X,
                                          clustering_algo = "leiden",
                                          seed = 187)

    facspy_adata = adata.copy()
    fp.tl.phenograph(facspy_adata,
                     gate = "live",
                     layer = "compensated",
                     use_only_fluo = False,
                     exclude = None,
                     scaling = None,
                     seed = 187,
                     clustering_algo = "leiden")
    assert "live_compensated_phenograph" in facspy_adata.obs
    assert comms.tolist() == facspy_adata.obs["live_compensated_phenograph"].tolist()
    assert (graph != facspy_adata.uns["live_compensated_phenograph_graph"]).nnz == 0
    assert Q == facspy_adata.uns["live_compensated_phenograph_Q"]


def test_decorator_default_gate_and_default_layer(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"

    fp.tl.phenograph(mock_dataset_downsampled)
    assert "live_compensated_phenograph" in mock_dataset_downsampled.obs.columns

def test_decorator_default_gate_and_default_layer_only_gate_provided(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_layer = "compensated"

    fp.tl.phenograph(mock_dataset_downsampled, gate = "live")
    assert "live_compensated_phenograph" in mock_dataset_downsampled.obs.columns

def test_decorator_default_gate_and_default_layer_only_layer_provided(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"

    fp.tl.phenograph(mock_dataset_downsampled, layer = "compensated")
    assert "live_compensated_phenograph" in mock_dataset_downsampled.obs.columns

def test_decorator_default_gate_and_default_layer_and_gate_alias(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"
    fp.settings.add_new_alias("live", "my_personal_gate")

    fp.tl.phenograph(mock_dataset_downsampled)
    assert "live_compensated_phenograph" in mock_dataset_downsampled.obs.columns

def test_decorator_default_gate_and_default_layer_and_gate_alias_use_alias_as_arg(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"
    fp.settings.add_new_alias("live", "my_personal_gate")

    fp.tl.phenograph(mock_dataset_downsampled, "my_personal_gate")
    assert "live_compensated_phenograph" in mock_dataset_downsampled.obs.columns

def test_decorator_default_gate_and_default_layer_and_gate_alias_use_alias_as_kwarg(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"
    fp.settings.add_new_alias("live", "my_personal_gate")

    fp.tl.phenograph(mock_dataset_downsampled, gate = "my_personal_gate")
    assert "live_compensated_phenograph" in mock_dataset_downsampled.obs.columns


