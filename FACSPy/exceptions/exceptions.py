from typing import Any
import warnings

class PanelMatchWarning():
    def __init__(self,
                 channel: str,
                 fcs_antigen: str,
                 panel_antigen: str) -> None:
        message = (
                    f"Antigens do not match for channel {channel}. " + 
                    f"FCS data documented {fcs_antigen} " +
                    f"while the user supplied panel documented {panel_antigen}. " +
                    f"The antigen will be referenced as described from the user supplied panel: {panel_antigen}"
                )
        warnings.warn(message, UserWarning)
    

class SupplementDataTypeError(Exception):
    
    def __init__(self,
                 data_type: Any,
                 class_name:str):
        self.data_type = data_type
        self.message = f"Please provide the {class_name} as a pandas dataframe, it was {self.data_type}"
        super().__init__(self.message)

class SupplementColumnError(Exception):
    
    def __init__(self,
                 colname: str,
                 class_name: str) -> None:
        self.message = (f"Column {colname} was not found in {class_name}. ")
        if class_name == "Panel":
            self.message += ("Please adjust data accordingly. Columns " +
                             "must be named 'fcs_colname' and antigens must " +
                             "be named 'antigens'.")
        elif class_name == "Metadata":
            self.message += ("Please make sure that the columns sample_ID and " +
                             "file_name are present. ")
        super().__init__(self.message)


class SupplementFileNotFoundError(Exception):

    def __init__(self,
                 panel_file):
        self.message = f"{panel_file} could not be found in the specified input directory!"
        super().__init__(self.message)


class SupplementCreationError(Exception):

    def __init__(self,
                 class_name):
        self.message = f"{class_name} could not be created because neither a file or a table was supplied and no flag to infer from data was created"
        super().__init__(self.message)

