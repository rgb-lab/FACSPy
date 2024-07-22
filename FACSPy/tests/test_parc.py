
import pytest
from anndata import AnnData
import FACSPy as fp
import parc as _parc
from FACSPy.exceptions._exceptions import InvalidScalingError


def test_invalid_scaling_error(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    with pytest.raises(InvalidScalingError):
        fp.tl.parc(mock_dataset_downsampled,
                   gate = "live",
                   scaling = "CustomScaler")

def test_save_settings_from_parc(mock_dataset_downsampled: AnnData):
    adata = mock_dataset_downsampled.copy()
    fp.subset_gate(adata, "live")
    fp.tl.pca(adata,
              gate = "live",
              layer = "compensated")
    fp.tl.parc(adata,
               gate = "live",
               layer = "compensated",
               random_state = 187,
               use_only_fluo = False,
               dist_std_local = 1,
               scaling = None,
               exclude = None)
    assert "settings" in adata.uns
    assert "_parc_live_compensated" in adata.uns["settings"]
    settings = adata.uns["settings"]["_parc_live_compensated"]
    assert settings["gate"] == "live"
    assert settings["layer"] == "compensated"
    assert settings["dist_std_local"] == 1
    assert settings["use_only_fluo"] == False

def test_parc_works_as_parc(mock_dataset_downsampled: AnnData):
    adata = mock_dataset_downsampled.copy()
    fp.subset_gate(adata, "live")
    fp.tl.pca(adata,
              gate = "live",
              layer = "compensated")
    parc_adata = adata.copy()
    parc_adata.X = parc_adata.layers["compensated"]
    fp.tl.neighbors(parc_adata,
                    gate = "live",
                    layer = "compensated",
                    use_only_fluo = False,
                    exclude = None,
                    scaling = None,
                    random_state = 187)
    
    parcer = _parc.PARC(parc_adata.X,
                        neighbor_graph = parc_adata.obsp["live_compensated_neighbors_connectivities"],
                        random_seed = 187)
    parcer.run_PARC()
    parc_clusters = parcer.labels

    facspy_adata = adata.copy()
    fp.tl.parc(facspy_adata,
               gate = "live",
               layer = "compensated",
               random_seed = 187,
               use_only_fluo = False,
               scaling = None,
               exclude = None)
    assert "live_compensated_parc" in facspy_adata.obs.columns
    assert all(parc_clusters == facspy_adata.obs["live_compensated_parc"])

def test_parc_works_as_parc_kwargs(mock_dataset_downsampled: AnnData):
    adata = mock_dataset_downsampled.copy()
    fp.subset_gate(adata, "live")
    fp.tl.pca(adata,
              gate = "live",
              layer = "compensated")
    fp.tl.neighbors(adata,
                    gate = "live",
                    layer = "compensated")
    parc_adata = adata.copy()
    parc_adata.X = parc_adata.layers["compensated"]
    
    parcer = _parc.PARC(parc_adata.X,
                        n_iter_leiden = 15,
                        neighbor_graph = parc_adata.obsp["live_compensated_neighbors_connectivities"],
                        random_seed = 187)
    parcer.run_PARC()
    parc_clusters = parcer.labels

    facspy_adata = adata.copy()
    fp.tl.parc(facspy_adata,
               gate = "live",
               layer = "compensated",
               random_seed = 187,
               use_only_fluo = False,
               scaling = None,
               exclude = None,
               n_iter_leiden = 15)
    assert "live_compensated_parc" in facspy_adata.obs.columns
    assert all(parc_clusters == facspy_adata.obs["live_compensated_parc"])


def test_decorator_default_gate_and_default_layer(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"

    fp.tl.parc(mock_dataset_downsampled)
    assert "live_compensated_parc" in mock_dataset_downsampled.obs.columns

def test_decorator_default_gate_and_default_layer_only_gate_provided(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_layer = "compensated"

    fp.tl.parc(mock_dataset_downsampled, gate = "live")
    assert "live_compensated_parc" in mock_dataset_downsampled.obs.columns

def test_decorator_default_gate_and_default_layer_only_layer_provided(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"

    fp.tl.parc(mock_dataset_downsampled, layer = "compensated")
    assert "live_compensated_parc" in mock_dataset_downsampled.obs.columns

def test_decorator_default_gate_and_default_layer_and_gate_alias(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"
    fp.settings.add_new_alias("live", "my_personal_gate")

    fp.tl.parc(mock_dataset_downsampled)
    assert "live_compensated_parc" in mock_dataset_downsampled.obs.columns

def test_decorator_default_gate_and_default_layer_and_gate_alias_use_alias_as_arg(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"
    fp.settings.add_new_alias("live", "my_personal_gate")

    fp.tl.parc(mock_dataset_downsampled, "my_personal_gate")
    assert "live_compensated_parc" in mock_dataset_downsampled.obs.columns

def test_decorator_default_gate_and_default_layer_and_gate_alias_use_alias_as_kwarg(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"
    fp.settings.add_new_alias("live", "my_personal_gate")

    fp.tl.parc(mock_dataset_downsampled, gate = "my_personal_gate")
    assert "live_compensated_parc" in mock_dataset_downsampled.obs.columns
