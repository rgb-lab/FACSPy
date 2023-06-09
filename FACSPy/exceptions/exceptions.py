from typing import Any
import warnings

class FileSaveError(Exception):

    def __init__(self):
        self.message = (
            "File has some entries that cannot be written."
        )

class FileIdentityError(Exception):

    def __init__(self):
        self.message = (
            "Identifiers are mismatched. The anndata and the uns were not saved at the same time."
        )

class ChannelSubsetError(Exception):

    def __init__(self):
        self.message = (
            "No channels for subsetting have been given. Please provide either a channel list or use the 'use_panel' parameter."
        )
        super().__init__(self.message)


class HierarchyError(Exception):

    def __init__(self):
        self.message = (
            "The specified parent gate is lower or equal in the gating hierarchy than the gate to display. " +
            "Please make sure that the parent is actually a parent."
        )
        super().__init__(self.message)


class CofactorsNotCalculatedError(Exception):

    def __init__(self):
        self.message = (
            "raw_cofactors has not been found in adata.uns. If you supplied a table of " +
            "cofactors, this is expected. No Distributions can be plotted. "
        )
        super().__init__(self.message)

class AnnDataSetupError(Exception):
    
    def __init__(self):
        self.message = (
            "This AnnData object has not been setup yet. Please call .setup_anndata() first."
        )
        super().__init__(self.message)

class ParentGateNotFoundError(Exception):

    def __init__(self,
                 parent_population):
        self.message = (
            f"The population {parent_population} was neither " +
             "found in the gating strategy provided by a workspace " +
             "or in the user-provided gating strategy. To avoid that, " +
             "make sure that all populations that are referred to are either " +
             "in the gating strategy provided or pregated in a workspace."
        )
        super().__init__(self.message)


class ClassifierNotImplementedError(Exception):

    def __init__(self,
                 classifier: Any,
                 implemented_classifiers: list[str]):
        self.message = (
            f"Classifier is not implemented. Please select one of {implemented_classifiers}, was {classifier}"
        )
        super().__init__(self.message)

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
                             "file_name are present.")
        super().__init__(self.message)


class SupplementFileNotFoundError(Exception):

    def __init__(self,
                 panel_file):
        self.message = f"{panel_file} could not be found in the specified input directory!"
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

