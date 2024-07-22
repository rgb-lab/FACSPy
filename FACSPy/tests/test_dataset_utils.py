import os
import FACSPy as fp
from anndata import AnnData
import numpy as np
import pandas as pd
from FACSPy.dataset._utils import (_replace_missing_cofactors,
                                   _merge_cofactors_into_dataset_var,
                                   match_cell_numbers,
                                   # find_name_of_control_sample_by_metadata,
                                   # reindex_metadata,
                                   asinh_transform,
                                   transform_data_array,
                                   get_control_samples,
                                   get_stained_samples,
                                   create_sample_subset_with_controls,
                                   find_corresponding_control_samples)

def test_replace_missing_cofactors():
    df = pd.DataFrame(data = {"pns": ["BV421-A", "BB700-A"], "cofactors": [np.nan, 4]},
                      index = list(range(2)))
    df = _replace_missing_cofactors(df)
    assert df["cofactors"].tolist() == [1,4]

def test_merge_cofactors_into_dataset_var(mock_dataset: AnnData,
                                          input_directory_test_suite):
    cofactors = fp.dt.CofactorTable(os.path.join(input_directory_test_suite, "cofactors_test_suite.txt"))
    adata = mock_dataset
    assert "cofactors" not in adata.var.columns
    # tests if present cofactors are actually replaced
    adata.var["cofactors"] = [-1 for _ in range(adata.var.shape[0])]

    df = _merge_cofactors_into_dataset_var(adata, cofactors)
    df = df.loc[cofactors.dataframe["fcs_colname"].tolist(),:]
    merged_cofactors = df["cofactors"].tolist()
    assert df.index.tolist() == cofactors.dataframe["fcs_colname"].tolist()
    assert cofactors.dataframe["cofactors"].tolist() == merged_cofactors
    assert not any(k == -1 for k in merged_cofactors)
    assert df["cofactors"].dtype == np.float32

def test_merge_cofactors_into_dataset_var_missing_channel(mock_dataset: AnnData,
                                                          input_directory_test_suite):
    cofactors = fp.dt.CofactorTable(os.path.join(input_directory_test_suite, "cofactors_test_suite.txt"))
    cofactors.dataframe = cofactors.dataframe.loc[cofactors.dataframe["fcs_colname"] != "live_dead",:]
    adata = mock_dataset
    assert "cofactors" not in adata.var.columns
    df = _merge_cofactors_into_dataset_var(adata, cofactors)
    df = df.loc[cofactors.dataframe["fcs_colname"].tolist(),:]
    merged_cofactors = df["cofactors"].tolist()
    assert df.index.tolist() == cofactors.dataframe["fcs_colname"].tolist()
    print(df.loc[df["pns"] == "live_dead", "cofactors"])
    assert df.loc[df["pns"] == "live_dead", "cofactors"].isna().all()
    assert not any(k == -1 for k in merged_cofactors)
    assert df["cofactors"].dtype == np.float32

def test_merge_cofactors_into_dataset_var_additional_channel(mock_dataset: AnnData,
                                                             input_directory_test_suite):
    cofactors = fp.dt.CofactorTable(os.path.join(input_directory_test_suite, "cofactors_test_suite.txt"))
    cofactors.dataframe.loc[cofactors.dataframe["fcs_colname"] == "live_dead","fcs_colname"] = "lifez"
    adata = mock_dataset
    assert "cofactors" not in adata.var.columns
    df = _merge_cofactors_into_dataset_var(adata, cofactors)

    merged_cofactors = df["cofactors"].tolist()
    assert df.loc[df["pns"] == "live_dead", "cofactors"].isna().all()
    assert "lifez" not in df.index.tolist()
    assert not any(k == -1 for k in merged_cofactors)
    assert df["cofactors"].dtype == np.float32

def test_match_cell_numbers(mock_dataset: AnnData):
    adata = mock_dataset
    min_cell_number = adata.obs.groupby("sample_ID").size().min()
    subset = match_cell_numbers(adata)
    assert all(subset.obs.groupby("sample_ID").size() == min_cell_number)

def test_get_stained_samples(mock_metadata: pd.DataFrame):
    print(mock_metadata)
    ctrl = get_stained_samples(mock_metadata, by = "file_name")
    assert len(ctrl) == 3
    assert all("stained" in filename for filename in ctrl)
    ctrl = get_stained_samples(mock_metadata, by = "sample_ID")
    assert len(ctrl) == 3
    assert ctrl == [1,3,5]

def test_get_control_samples(mock_metadata: pd.DataFrame):
    ctrl = get_control_samples(mock_metadata, by = "file_name")
    assert len(ctrl) == 3
    assert all("unstained" in filename for filename in ctrl)
    ctrl = get_control_samples(mock_metadata, by = "sample_ID")
    assert len(ctrl) == 3
    assert ctrl == [2,4,6]

def test_get_stained_samples_anndata(mock_dataset: AnnData):
    stained_samples = get_stained_samples(mock_dataset.uns["metadata"].dataframe,
                                          by = "sample_ID")
    assert stained_samples == ["1", "3", "5"]
    stained_samples = get_stained_samples(mock_dataset.uns["metadata"].dataframe,
                                          by = "file_name")
    assert stained_samples == ["file1_stained.fcs", "file2_stained.fcs", "file3_stained.fcs"]


def test_get_control_samples_anndata(mock_dataset: AnnData):
    stained_samples = get_control_samples(mock_dataset.uns["metadata"].dataframe,
                                          by = "sample_ID")
    assert stained_samples == ["2", "4", "6"]
    stained_samples = get_control_samples(mock_dataset.uns["metadata"].dataframe,
                                          by = "file_name")
    assert stained_samples == ["file1_unstained.fcs", "file2_unstained.fcs", "file3_unstained.fcs"]

def test_reindex_metadata():
    pass

def test_find_name_of_control_sample_by_metadata(mock_dataset: AnnData):
    pass


def test_find_corresponding_control_samples_with_controls(mock_dataset: AnnData):
    adata = mock_dataset
    print(adata.uns["metadata"].to_df().shape)
    _, controls = find_corresponding_control_samples(adata, by = "file_name")
    assert isinstance(controls, dict)
    assert all(len(value) == 1 for _, value in controls.items())

    _, controls = find_corresponding_control_samples(adata, by = "sample_ID")
    assert isinstance(controls, dict)
    assert all(len(value) == 1 for _, value in controls.items())

def test_find_corresponding_control_samples_without_controls_mouse_lineages(mouse_lineages_metadata: pd.DataFrame):
    mouse_lineages_metadata = mouse_lineages_metadata[mouse_lineages_metadata["staining"] == "stained"]
    metadata = fp.dt.Metadata(metadata = mouse_lineages_metadata)
    adata = AnnData(uns = {"metadata": metadata})

    _, controls = find_corresponding_control_samples(adata, by = "file_name")
    assert isinstance(controls, dict)
    assert all(value == [] for _, value in controls.items())

    _, controls = find_corresponding_control_samples(adata, by = "sample_ID")
    assert isinstance(controls, dict)
    assert all(value == [] for _, value in controls.items())


def test_find_corresponding_control_samples_with_controls_mouse_lineages_by_sample_ID(mouse_lineages_metadata: pd.DataFrame):
    mouse_lineages_metadata["sample_ID"] = mouse_lineages_metadata["sample_ID"].astype(str)
    metadata = fp.dt.Metadata(metadata = mouse_lineages_metadata)
    adata = AnnData(uns = {"metadata": metadata})

    stained_samples, controls = find_corresponding_control_samples(adata, by = "sample_ID")
    all_control_samples = get_control_samples(mouse_lineages_metadata, by = "sample_ID")
    assert all(sample in mouse_lineages_metadata.loc[mouse_lineages_metadata["staining"] == "stained", "sample_ID"].tolist()
               for sample in stained_samples)
    assert all(sample in mouse_lineages_metadata.loc[mouse_lineages_metadata["staining"] == "unstained", "sample_ID"].tolist()
               for sample in all_control_samples)

    metadata_factors = metadata.get_factors()
    md_frame = metadata.dataframe
    for sample in stained_samples:
        ctrl_samples = controls[sample]
        try:
            # either all control samples have been assigned...
            assert (
                all(ctrl_s in all_control_samples
                for ctrl_s in ctrl_samples)
            )
        except AssertionError:
            # or the metadata have to match.
            for cs in ctrl_samples:
                sample_metadata = md_frame.loc[md_frame["sample_ID"] == sample,metadata_factors].values
                ctrl_metadata = md_frame.loc[md_frame["sample_ID"] == cs,metadata_factors].values
                np.testing.assert_array_equal(sample_metadata, ctrl_metadata)

def test_find_corresponding_control_samples_with_controls_mouse_lineages_by_file_name(mouse_lineages_metadata: pd.DataFrame):
    mouse_lineages_metadata["sample_ID"] = mouse_lineages_metadata["sample_ID"].astype(str)
    metadata = fp.dt.Metadata(metadata = mouse_lineages_metadata)
    adata = AnnData(uns = {"metadata": metadata})
    stained_samples, controls = find_corresponding_control_samples(adata, by = "file_name")
    all_control_samples = get_control_samples(mouse_lineages_metadata, by = "file_name")
    assert all(sample in mouse_lineages_metadata.loc[mouse_lineages_metadata["staining"] == "stained", "file_name"].tolist()
               for sample in stained_samples)
    assert all(sample in mouse_lineages_metadata.loc[mouse_lineages_metadata["staining"] == "unstained", "file_name"].tolist()
               for sample in all_control_samples)

    metadata_factors = metadata.get_factors()
    md_frame = metadata.dataframe
    for sample in stained_samples:
        ctrl_samples = controls[sample]
        try:
            # either all control samples have been assigned...
            assert (
                all(ctrl_s in all_control_samples
                for ctrl_s in ctrl_samples)
            )
        except AssertionError:
            # or the metadata have to match.
            for cs in ctrl_samples:
                sample_metadata = md_frame.loc[md_frame["file_name"] == sample,metadata_factors].values
                ctrl_metadata = md_frame.loc[md_frame["file_name"] == cs,metadata_factors].values
                np.testing.assert_array_equal(sample_metadata, ctrl_metadata)


def test_transform_data_array():
    cofactors = np.array([1,1,1], dtype = np.float64)
    input_data = np.array([1,1,1], dtype = np.float64)
    x = transform_data_array(input_data, cofactors)
    np.testing.assert_array_almost_equal(np.repeat([0.88137359],3), x)

    cofactors = [5,5,5]
    input_data = np.array([5,5,5])
    x = transform_data_array(input_data, cofactors)
    np.testing.assert_array_almost_equal(np.repeat([0.88137359],3), x)

def test_asinh_transformation():
    cofactors = [1,1,1]
    input_data = np.array([1,1,1])
    x = asinh_transform(input_data, cofactors)
    np.testing.assert_array_almost_equal(np.repeat([0.88137359],3), x)

    cofactors = [5,5,5]
    input_data = np.array([5,5,5])
    x = asinh_transform(input_data, cofactors)
    np.testing.assert_array_almost_equal(np.repeat([0.88137359],3), x)


def test_find_corresponding_control_samples(mock_dataset: AnnData):
    print(mock_dataset.uns["metadata"].to_df().shape)
    ccs = find_corresponding_control_samples(mock_dataset,
                                             by = "file_name")
    assert ccs[0] == ["file1_stained.fcs", "file2_stained.fcs", "file3_stained.fcs"]
    assert isinstance(ccs[1], dict)
    ccs_dict = ccs[1]
    assert "file1_stained.fcs" in ccs_dict.keys()
    assert "file2_stained.fcs" in ccs_dict.keys()
    assert "file3_stained.fcs" in ccs_dict.keys()
    assert ccs_dict["file1_stained.fcs"] == ["file1_unstained.fcs"]
    assert ccs_dict["file2_stained.fcs"] == ["file2_unstained.fcs"]
    assert ccs_dict["file3_stained.fcs"] == ["file3_unstained.fcs"]

def test_create_sample_subset_with_controls(mock_dataset: AnnData):
    ccs = find_corresponding_control_samples(mock_dataset,
                                             by = "file_name")
    x = create_sample_subset_with_controls(mock_dataset,
                                           "file1_stained.fcs",
                                           corresponding_controls = ccs[1],
                                           match_cell_number = False)
    assert x.shape == (99351, 21)
    assert len(x.obs["file_name"].unique()) == 2

def test_create_sample_subset_with_controls_matching(mock_dataset: AnnData):
    ccs = find_corresponding_control_samples(mock_dataset,
                                             by = "file_name")
    x = create_sample_subset_with_controls(mock_dataset,
                                           "file1_stained.fcs",
                                           corresponding_controls = ccs[1],
                                           match_cell_number = True)
    assert x.shape == (68116, 21)
    assert len(x.obs["file_name"].unique()) == 2


def test_find_corresponding_control_samples_anndata_no_controls(mock_dataset: AnnData):
    adata = mock_dataset
    
    # leave syntax like this as metadata is a fixture object!
    import copy
    metadata = copy.deepcopy(adata.uns["metadata"])
    metadata.subset("staining", "stained")
    adata.uns["metadata"] = metadata

    fp.sync.synchronize_dataset(adata)
    assert "unstained" not in adata.obs["staining"]
    assert not get_control_samples(adata.uns["metadata"].dataframe, by = "file_name")

    _, controls = find_corresponding_control_samples(adata, by = "file_name")
    assert isinstance(controls, dict)
    assert all(value == [] for _, value in controls.items())

    _, controls = find_corresponding_control_samples(adata, by = "sample_ID")
    assert isinstance(controls, dict)
    assert all(value == [] for _, value in controls.items())




