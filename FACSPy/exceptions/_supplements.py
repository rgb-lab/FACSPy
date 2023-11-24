import warnings
from typing import Any

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
    
class SupplementNoInputDirectoryError(Exception):

    def __init__(self):
        self.message = (
            "You did not provide an input directory. If 'from_fcs' is set to True, " + 
            "an input directory containing the fcs files must be provided"
        )
        super().__init__(self.message)

class SupplementInputTypeError(Exception):
    
    def __init__(self,
                 data_type: Any,
                 class_name:str):
        self.data_type = data_type
        self.message = f"Please provide the {class_name} as a pandas dataframe, it was {self.data_type}"
        super().__init__(self.message)

class SupplementDataTypeError(Exception):
    
    def __init__(self,
                 data_type: Any,
                 class_name:str):
        self.data_type = data_type
        self.message = f"Please provide the {class_name} as a {class_name} object by calling fp.dt.{class_name}(), it was {self.data_type}"
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
                             "file_name are present.")
        super().__init__(self.message)


class SupplementFileNotFoundError(Exception):

    def __init__(self,
                 file):
        self.message = f"{file} could not be found in the specified input directory!"
        super().__init__(self.message)


class SupplementCreationError(Exception):

    def __init__(self,
                 class_name):
        keyword_map = {
            "Panel": "panel = ",
            "Metadata": "metadata = ",
            "CofactorTable": "cofactors = "
        }
        self.message = f"{class_name} could not be created because neither a file or a table "
        self.message += "was supplied and no flag to infer from data was created. "
        self.message += f"If you provided a dataframe, please use the appropriate keyword '{keyword_map[class_name]}'"
        super().__init__(self.message)


class SupplementFormatError(Exception):

    def __init__(self,
                 supplement,
                 instance_type):
        self.message = (
            f"You tried to supply a {supplement} object, but supplied an object of type {instance_type}. " + 
            f"Please run the analysis by providing a {supplement} object."
        )
        super().__init__(self.message)
