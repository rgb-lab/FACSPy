import pytest
import os
from FACSPy.dataset._supplements import Metadata, Panel, CofactorTable
from FACSPy.dataset._workspaces import FlowJoWorkspace
from FACSPy.dataset._dataset import DatasetAssembler

from anndata import AnnData
import scanpy as sc

import FACSPy as fp
import pandas as pd

INPUT_DIRECTORY = "FACSPy/_resources/test_suite_dataset"

@pytest.fixture
def dummy_workspace():
    WSP_FILE_PATH = "FACSPy/_resources/"
    WSP_FILE_NAME = "test_wsp.wsp"
    return FlowJoWorkspace(os.path.join(WSP_FILE_PATH, WSP_FILE_NAME))

@pytest.fixture(scope = "session")
def mock_metadata():
    return pd.read_csv(os.path.join(INPUT_DIRECTORY, "metadata_test_suite.csv"), sep = ";")

@pytest.fixture(scope = "session")
def mouse_lineages_metadata():
    return pd.read_csv(os.path.join(INPUT_DIRECTORY, "metadata_mouse_lineages.csv"))

@pytest.fixture(scope = "session")
def supplement_objects():
    panel = Panel(os.path.join(INPUT_DIRECTORY, "panel.txt"))
    metadata = Metadata(os.path.join(INPUT_DIRECTORY, "metadata_test_suite.csv"))
    workspace = FlowJoWorkspace(os.path.join(INPUT_DIRECTORY, "test_suite.wsp"))
    return INPUT_DIRECTORY, panel, metadata, workspace

@pytest.fixture(scope = "session")
def input_directory_test_suite():
    return INPUT_DIRECTORY

@pytest.fixture(scope = "session")
def mock_dataset_assembler_object(supplement_objects):
    input_directory, panel, metadata, workspace = supplement_objects
    return DatasetAssembler(input_directory = input_directory,
                            panel = panel,
                            metadata = metadata,
                            workspace = workspace)

@pytest.fixture(scope = "session")
def mock_dataset_no_raw(supplement_objects):
    input_directory, panel, metadata, workspace = supplement_objects
    return fp.create_dataset(input_directory = input_directory,
                             panel = panel,
                             metadata = metadata,
                             workspace = workspace,
                             keep_raw = False)

@pytest.fixture(scope = "module")
def mock_dataset(mock_dataset_no_raw):
    return mock_dataset_no_raw

@pytest.fixture(scope = "module")
def mock_dataset_low_cell_count(supplement_objects) -> AnnData:
    input_directory, panel, metadata, workspace = supplement_objects
    adata = fp.create_dataset(input_directory = input_directory,
                              panel = panel,
                              metadata = metadata,
                              workspace = workspace,
                              subsample_fcs_to = 100)
    adata.layers["transformed"] = adata.layers["compensated"].copy()
    return adata

@pytest.fixture(scope = "module")
def mock_dataset_mfi_calc(mock_dataset_low_cell_count):
    mock_dataset_low_cell_count = mock_dataset_low_cell_count.copy()
    fp.tl.mfi(mock_dataset_low_cell_count, layer = "compensated")
    mfi_calc_dataset = mock_dataset_low_cell_count.copy()
    return mfi_calc_dataset

@pytest.fixture(scope = "module")
def mock_dataset_downsampled(mock_dataset_low_cell_count) -> AnnData:
    mock_dataset_low_cell_count = mock_dataset_low_cell_count.copy()
    mock_dataset_low_cell_count = sc.pp.subsample(
        mock_dataset_low_cell_count, n_obs = 200,
        random_state = 187, copy = True
    )
    return mock_dataset_low_cell_count

@pytest.fixture(scope = "module")
def mock_dataset_downsampled_with_cofactors(mock_dataset_downsampled):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    cofactors = CofactorTable(os.path.join(INPUT_DIRECTORY, "cofactors_test_suite.txt"))
    mock_dataset_downsampled.uns["cofactors"] = cofactors
    mock_dataset_downsampled.layers["transformed"] = mock_dataset_downsampled.layers["compensated"].copy()
    return mock_dataset_downsampled

@pytest.fixture(scope = "session")
def mock_dataset_with_raw(supplement_objects):
    input_directory, panel, metadata, workspace = supplement_objects
    return fp.create_dataset(input_directory = input_directory,
                             panel = panel,
                             metadata = metadata,
                             workspace = workspace,
                             keep_raw = True)

@pytest.fixture(scope = "session")
def mouse_data() -> AnnData:
    adata = fp.mouse_lineages()
    fp.tl.gate_frequencies(adata)
    fp.tl.mfi(adata, layer = "compensated")
    fp.tl.fop(adata, layer = "compensated")
    gate = "Neutrophils"
    layer = "compensated"
    fp.tl.pca(adata, gate = gate, layer = layer)
    fp.tl.pca_samplewise(adata, layer = layer)
    fp.tl.neighbors(adata, gate = gate, layer = layer)
    fp.tl.leiden(adata, gate = gate, layer = layer)
    fp.tl.mfi(adata,
              groupby = "Neutrophils_compensated_leiden",
              aggregate = False)
    fp.tl.fop(adata,
              groupby = "Neutrophils_compensated_leiden",
              aggregate = False)

    return adata
 

