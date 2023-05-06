from flowio import FlowData
import warnings

def load_fcs_file_from_disk(input_directory: str,
                            ignore_offset_error: bool) -> FlowData:
    try:
        return FlowData(input_directory, ignore_offset_error)
    except ValueError:
        warnings.warn("FACSPy IO: FCS file could not be read with " + 
                      f"ignore_offset_error set to {ignore_offset_error}. " +
                      "Parameter is set to True.")
        return FlowData(input_directory, ignore_offset_error = True)
    