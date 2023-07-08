from typing import Any
import warnings


class AnalysisNotPerformedError(Exception):

    def __init__(self,
                 analysis):
        self.message = (
            f"You tried to access analysis values for {analysis}. Data were not found. " +
            f"Please run {analysis}() first."
        )
        super().__init__(self.message)


class FileSaveError(Exception):

    def __init__(self):
        self.message = (
            "File has some entries that cannot be written."
        )
        super().__init__(self.message)

class FileIdentityError(Exception):

    def __init__(self):
        self.message = (
            "Identifiers are mismatched. The anndata and the uns were not saved at the same time."
        )
        super().__init__(self.message)

class ChannelSubsetError(Exception):

    def __init__(self):
        self.message = (
            "No channels for subsetting have been given. Please provide either a channel list or use the 'use_panel' parameter."
        )
        super().__init__(self.message)

class NotSupportedStatisticalTestError(Exception):
    
    def __init__(self,
                 test: str,
                 available_tests) -> None:
        self.message = (
            f"The test ({test}) you provided is not supported. Please choose one of {available_tests}"
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


