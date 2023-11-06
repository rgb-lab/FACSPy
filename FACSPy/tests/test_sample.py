import pytest
import pandas as pd

from FACSPy.dataset._sample import FCSFile
from FACSPy.transforms._matrix import Matrix
from FACSPy.exceptions._exceptions import NotCompensatedError



FCS_FILE_PATH = "FACSPy/_resources/"
FCS_FILE_NAME = "test_fcs.fcs"

def read_fcs_file():
    input_directory = FCS_FILE_PATH
    file_name = FCS_FILE_NAME
    return FCSFile(input_directory = input_directory,
                   file_name = file_name)

@pytest.fixture
def fcs_file():
    return read_fcs_file()

def test_read_function(fcs_file: FCSFile):
    assert isinstance(fcs_file, FCSFile)

def test_filename_attribute(fcs_file: FCSFile):
    assert fcs_file.original_filename == "test_fcs.fcs"

def test_event_count(fcs_file: FCSFile):
    ### as read with flowjo
    assert fcs_file.event_count == 13480

def test_version_status(fcs_file: FCSFile):
    assert fcs_file.version == "3.0"

def test_fcs_metadata(fcs_file: FCSFile):
    assert fcs_file.fcs_metadata
    assert fcs_file.fcs_metadata["byteord"] == "4,3,2,1"
    assert fcs_file.fcs_metadata["creator"] == "BD FACSDiva Software Version 9.1"

def test_channels(fcs_file: FCSFile):
    assert isinstance(fcs_file.channels, pd.DataFrame)
    assert all(fcs_file.channels["png"] == 1.0)
    assert all(fcs_file.channels["pnr"] == 262144)
    assert fcs_file.channels["pns"].dtype == "object"
    assert fcs_file.channels["png"].dtype == "float64"
    assert fcs_file.channels["pne"].dtype == "object"
    assert fcs_file.channels["pnr"].dtype == "int64"
    assert fcs_file.channels["channel_numbers"].dtype == "int64"

def test_fcs_compensation(fcs_file: FCSFile):
    assert isinstance(fcs_file.fcs_compensation, Matrix)
    assert fcs_file.fcs_compensation.matrix.shape == (14,14)
    assert all(fcs_file.fcs_compensation.fluorochromes == fcs_file.channels["pns"].to_list())
    assert fcs_file.fcs_compensation.id == "acquisition_defined"

def test_get_events(fcs_file: FCSFile):
    assert fcs_file.get_events(source = "raw").shape == (13480,21)

def test_get_comp_from_uncomped_file(fcs_file: FCSFile):
    with pytest.raises(NotCompensatedError):
        _ = fcs_file.get_events("comp")

def test_get_events_with_unknown_key(fcs_file: FCSFile):
    with pytest.raises(NotImplementedError):
        _ = fcs_file.get_events("something")

def test_get_raw_events(fcs_file: FCSFile):
    assert fcs_file._get_original_events().shape == (13480,21)

def test_get_channel_index(fcs_file: FCSFile):
    assert fcs_file.get_channel_index("Time") == 20

def test_time_channel_exists(fcs_file: FCSFile):
    assert fcs_file._time_channel_exists()

def test_find_time_channel(fcs_file: FCSFile):
    time_index, time_step = fcs_file._find_time_channel()
    assert time_index == 20
    assert time_step == 0.01

def test_remove_disallowed_characters(fcs_file: FCSFile):
    assert "/" not in fcs_file._remove_disallowed_characters_from_string("/WHAT/")