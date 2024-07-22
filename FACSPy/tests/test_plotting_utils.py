import pytest
from anndata import AnnData

import FACSPy as fp
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from FACSPy.plotting._utils import (_remove_ticks,
                                    _remove_ticklabels,
                                    _label_metaclusters_in_dataset)
from FACSPy.exceptions._exceptions import MetaclusterOverwriteWarning

@pytest.fixture
def test_figure() -> tuple[Figure, Axes]:
    np.random.seed(187)
    data = pd.DataFrame(data = {"x": np.random.randint(1, 50,50),
                                "y": np.random.randint(1, 50, 50)},
                        index = list(range(50)))
    fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (3,3))
    sns.scatterplot(data = data,
                    x = "x",
                    y = "y",
                    ax = ax)
    return fig, ax


def test_remove_ticks_and_labels_x(test_figure: tuple[Figure, Axes]):
    _, ax = test_figure
    _remove_ticks(ax, which = "x")
    _remove_ticklabels(ax, which = "x")
    assert ax.get_xticklabels() == []

def test_remove_ticks_and_labels_y(test_figure: tuple[Figure, Axes]):
    _, ax = test_figure
    _remove_ticks(ax, which = "y")
    _remove_ticklabels(ax, which = "y")
    assert ax.get_yticklabels() == []

def test_remove_ticks_and_labels_both(test_figure: tuple[Figure, Axes]):
    _, ax = test_figure
    _remove_ticks(ax, which = "both")
    _remove_ticklabels(ax, which = "both")
    assert ax.get_yticklabels() == []
    assert ax.get_xticklabels() == []

def test_label_metaclusters_in_dataset(mock_dataset_downsampled: AnnData):
    adata = mock_dataset_downsampled.copy()
    data: pd.DataFrame = adata.uns["metadata"].dataframe.copy()
    data["metacluster"] = list(range(data.shape[0]))
    _label_metaclusters_in_dataset(adata = adata,
                                   data = data)
    metadata: pd.DataFrame = adata.uns["metadata"].dataframe
    assert "metacluster" in metadata.columns
    assert data[["sample_ID", "metacluster"]].to_dict() == metadata[["sample_ID", "metacluster"]].to_dict()

def test_label_metaclusters_in_dataset_warning(mock_dataset_downsampled: AnnData):
    adata = mock_dataset_downsampled.copy()
    data: pd.DataFrame = adata.uns["metadata"].dataframe.copy()
    adata.uns["metadata"].dataframe["metacluster"] = list(range(adata.uns["metadata"].dataframe.shape[0]))
    data["metacluster"] = list(range(data.shape[0]))
    with pytest.warns(MetaclusterOverwriteWarning):
        _label_metaclusters_in_dataset(adata = adata,
                                       data = data)
    _label_metaclusters_in_dataset(adata = adata,
                                   data = data)
    metadata: pd.DataFrame = adata.uns["metadata"].dataframe
    assert "metacluster" in metadata.columns
    assert data[["sample_ID", "metacluster"]].to_dict() == metadata[["sample_ID", "metacluster"]].to_dict()

def test_label_metaclusters_in_dataset_warning_II(mock_dataset_downsampled: AnnData):
    adata = mock_dataset_downsampled.copy()
    data = adata.uns["metadata"].dataframe.copy()
    adata.uns["metadata"].dataframe["my_key"] = list(range(adata.uns["metadata"].dataframe.shape[0]))
    data["metacluster"] = list(range(data.shape[0]))
    with pytest.warns(MetaclusterOverwriteWarning):
        _label_metaclusters_in_dataset(adata = adata,
                                       data = data,
                                       label_metaclusters_key = "my_key")
    _label_metaclusters_in_dataset(adata = adata,
                                   data = data,
                                   label_metaclusters_key = "my_key")
    metadata = adata.uns["metadata"].dataframe
    assert "my_key" in metadata.columns
    metadata.rename(columns = {"my_key": "metacluster"},
                    inplace = True)
    assert data[["sample_ID", "metacluster"]].to_dict() == metadata[["sample_ID", "metacluster"]].to_dict()


#def test_map_metaclusters_to_sample_ID():
#    np.random.seed(187)
#    df = pd.DataFrame(np.random.randint(0,100,100).reshape(10,10))
#    from FACSPy.plotting._utils import (_calculate_linkage,
#                                        _calculate_metaclusters,
#                                        _map_metaclusters_to_sample_ID)
#    linkage = _calculate_linkage(df)
#    sample_IDs = list(range(10))
#    metaclusters = _calculate_metaclusters(linkage, n_clusters = 3, sample_IDs = sample_IDs)
#    facspy_mapping: pd.DataFrame = _map_metaclusters_to_sample_ID(metaclusters, sample_IDs)
#    def control_map_func(metaclusters, sample_IDs):
#        sample_IDs = pd.DataFrame(sample_IDs, columns = ["sample_ID"])
#        for i, sample_ID in enumerate(sample_IDs["sample_ID"].to_list()):
#            sample_IDs.loc[sample_IDs["sample_ID"] == sample_ID, "metacluster"] = str(int([metacluster
#                                                                                       for metacluster in metaclusters
#                                                                                       if i in metaclusters[metacluster]][0]))
#        return sample_IDs
#    control_mapping: pd.DataFrame = control_map_func(metaclusters, sample_IDs)
#    assert facspy_mapping.equals(control_mapping)
#
def test_get_uns_dataframe_I(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.tl.mfi(mock_dataset_downsampled, layer = "compensated")
    from FACSPy.plotting._utils import _get_uns_dataframe
    from FACSPy._utils import _find_gate_path_of_gate
    df = _get_uns_dataframe(mock_dataset_downsampled,
                            gate = "live",
                            table_identifier = "mfi_sample_ID_compensated")
    assert all(df["gate"] == _find_gate_path_of_gate(mock_dataset_downsampled, "live"))
    assert all(col in df.columns for col in mock_dataset_downsampled.uns["metadata"].dataframe.columns)
    assert "sample_ID" in df.columns
    assert "gate" in df.columns

def test_get_uns_dataframe_II(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.tl.mfi(mock_dataset_downsampled, groupby = "condition1", layer = "compensated")
    from FACSPy.plotting._utils import _get_uns_dataframe
    from FACSPy._utils import _find_gate_path_of_gate
    df = _get_uns_dataframe(mock_dataset_downsampled,
                            gate = "live",
                            table_identifier = "mfi_condition1_compensated")
    assert all(df["gate"] == _find_gate_path_of_gate(mock_dataset_downsampled, "live"))
    assert all(col in df.columns for col in mock_dataset_downsampled.uns["metadata"].dataframe.columns)
    assert "sample_ID" in df.columns
    assert "gate" in df.columns
    assert "condition1" in df.columns


