import pytest
from anndata import AnnData
import scanpy as sc
import FACSPy as fp
from FACSPy.exceptions._exceptions import InvalidScalingError

def test_invalid_scaling_error(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    with pytest.raises(InvalidScalingError):
        fp.tl.leiden(mock_dataset_downsampled,
                     gate = "live",
                     scaling = "CustomScaler")

def test_save_settings_from_leiden(mock_dataset_downsampled: AnnData):
    adata = mock_dataset_downsampled.copy()
    fp.tl.leiden(adata,
                 gate = "live",
                 layer = "compensated",
                 random_state = 187,
                 use_only_fluo = False,
                 resolution = 5,
                 use_rep = "X",
                 scaling = None,
                 exclude = None)
    assert "settings" in adata.uns
    assert "_leiden_live_compensated" in adata.uns["settings"]
    settings = adata.uns["settings"]["_leiden_live_compensated"]
    assert settings["gate"] == "live"
    assert settings["layer"] == "compensated"
    assert settings["resolution"] == 5
    assert settings["use_only_fluo"] is False

def test_leiden_works_as_scanpy(mock_dataset_downsampled: AnnData):
    adata = mock_dataset_downsampled.copy()
    fp.subset_gate(adata, "live")
    fp.tl.pca(adata,
              gate = "live",
              layer = "compensated")
    scanpy_adata = adata.copy()
    scanpy_adata.X = scanpy_adata.layers["compensated"]
    fp.tl.neighbors(scanpy_adata,
                    gate = "live",
                    layer = "compensated")

    sc.tl.leiden(scanpy_adata,
                 neighbors_key = "live_compensated_neighbors")
    leiden_clusters = scanpy_adata.obs["leiden"]

    facspy_adata = adata.copy()
    fp.tl.leiden(facspy_adata,
                 gate = "live",
                 layer = "compensated",
                 random_seed = 187,
                 use_only_fluo = False,
                 scaling = None,
                 exclude = None)
    assert "live_compensated_leiden" in facspy_adata.obs.columns
    assert all(leiden_clusters == facspy_adata.obs["live_compensated_leiden"])

def test_leiden_works_as_scanpy_kwargs(mock_dataset_downsampled: AnnData):
    adata = mock_dataset_downsampled.copy()
    fp.subset_gate(adata, "live")
    fp.tl.pca(adata,
              gate = "live",
              layer = "compensated")
    fp.tl.neighbors(adata,
                    gate = "live",
                    layer = "compensated")
    scanpy_adata = adata.copy()
    scanpy_adata.X = scanpy_adata.layers["compensated"]

    sc.tl.leiden(scanpy_adata,
                 resolution = 0.1,
                 neighbors_key = "live_compensated_neighbors")
    leiden_clusters = scanpy_adata.obs["leiden"]


    facspy_adata = adata.copy()
    fp.tl.leiden(facspy_adata,
                 gate = "live",
                 layer = "compensated",
                 random_seed = 187,
                 use_only_fluo = False,
                 resolution = 0.1,
                 scaling = None,
                 exclude = None)
    assert "live_compensated_leiden" in facspy_adata.obs.columns
    assert all(leiden_clusters == facspy_adata.obs["live_compensated_leiden"])

def test_leiden_works_as_scanpy_all_scanpy_funcs(mock_dataset_downsampled: AnnData):
    adata = mock_dataset_downsampled.copy()
    fp.subset_gate(adata, "live")
    scanpy_adata = adata.copy()
    scanpy_adata.X = scanpy_adata.layers["compensated"]
    sc.pp.pca(scanpy_adata, random_state = 187)
    sc.pp.neighbors(scanpy_adata,
                    use_rep = "X_pca",
                    random_state = 187) 
    sc.tl.leiden(scanpy_adata,
                 resolution = 0.1,
                 random_state = 187)
    leiden_clusters = scanpy_adata.obs["leiden"]
    facspy_adata = adata.copy()
    fp.tl.leiden(facspy_adata,
                 gate = "live",
                 layer = "compensated",
                 random_seed = 187,
                 use_only_fluo = False,
                 resolution = 0.1,
                 scaling = None,
                 exclude = None)
    assert "live_compensated_leiden" in facspy_adata.obs.columns
    assert all(leiden_clusters == facspy_adata.obs["live_compensated_leiden"])

def test_decorator_default_gate_and_default_layer(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"

    fp.tl.leiden(mock_dataset_downsampled)
    assert "live_compensated_leiden" in mock_dataset_downsampled.obs.columns

def test_decorator_default_gate_and_default_layer_only_gate_provided(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_layer = "compensated"

    fp.tl.leiden(mock_dataset_downsampled, gate = "live")
    assert "live_compensated_leiden" in mock_dataset_downsampled.obs.columns

def test_decorator_default_gate_and_default_layer_only_layer_provided(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"

    fp.tl.leiden(mock_dataset_downsampled, layer = "compensated")
    assert "live_compensated_leiden" in mock_dataset_downsampled.obs.columns

def test_decorator_default_gate_and_default_layer_and_gate_alias(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"
    fp.settings.add_new_alias("live", "my_personal_gate")

    fp.tl.leiden(mock_dataset_downsampled)
    assert "live_compensated_leiden" in mock_dataset_downsampled.obs.columns

def test_decorator_default_gate_and_default_layer_and_gate_alias_use_alias_as_arg(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"
    fp.settings.add_new_alias("live", "my_personal_gate")

    fp.tl.leiden(mock_dataset_downsampled, "my_personal_gate")
    assert "live_compensated_leiden" in mock_dataset_downsampled.obs.columns

def test_decorator_default_gate_and_default_layer_and_gate_alias_use_alias_as_kwarg(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"
    fp.settings.add_new_alias("live", "my_personal_gate")

    fp.tl.leiden(mock_dataset_downsampled, gate = "my_personal_gate")
    assert "live_compensated_leiden" in mock_dataset_downsampled.obs.columns
