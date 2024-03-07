from typing import Any
import warnings

class CofactorNotFoundWarning(Warning):
    def __init__(self,
                 message) -> None:
        self.message = message
        warnings.warn(message, UserWarning)
    
    def __str__(self):
        return repr(self.message)

class InfRemovalWarning(Warning):
    def __init__(self,
                 message) -> None:
        self.message = message
        warnings.warn(message, UserWarning)

    def __str__(self):
        return repr(self.message)

class NaNRemovalWarning(Warning):
    def __init__(self,
                 message) -> None:
        self.message = message
        warnings.warn(message, UserWarning)
    
    def __str__(self):
        return repr(self.message)

class InvalidTransformationError(Exception):

    def __init__(self,
                 transform,
                 implemented_transformations):
        self.message = (
            f"You selected {transform} for transformation, which is not implemented. " + 
            f"Please choose one of {implemented_transformations}"
        )
        super().__init__(self.message)

class MetaclusterOverwriteWarning(Warning):
    def __init__(self,
                 message) -> None:
        self.message = message
    
    def __str__(self):
        return repr(self.message)


class InsufficientSampleNumberWarning(Warning):
    def __init__(self,
                 message) -> None:
        self.message = message

    def __str__(self):
        return repr(self.message)

    @classmethod
    def _construct_message(cls,
                           gate: str,
                           n_samples_per_gate: int,
                           n_components: int) -> None:
        message = (
            f"The gate {gate} has only members in {n_samples_per_gate} samples, " +
            "which is insufficient for the dimensionality reduction with " +
            f"{n_components} components. The coordinates of the missing components " +
            f"are set to NaN."
        )
        return message


class DimredSettingModificationWarning(Warning):
    def __init__(self,
                 message) -> None:
        self.message = message

    def __str__(self):
        return repr(self.message)
    
    @classmethod
    def _construct_message(cls,
                           dimred,
                           parameter,
                           new_value,
                           reason) -> None:
        message = (
            f"{dimred}: The settings where changed, because {reason}. " + 
            f"The settings was {parameter} and the new value is {new_value}"
        )
        return message



class AnalysisNotPerformedError(Exception):

    def __init__(self,
                 analysis):
        self.message = (
            f"You tried to access analysis values for {analysis}. Data were not found. " +
            f"Please run {analysis}() first."
        )
        super().__init__(self.message)

class InvalidScalingError(Exception):

    def __init__(self,
                 scaler):
        from .._utils import IMPLEMENTED_SCALERS
        self.message = (
            f"Invalid scaling method {scaler}. Please select one of {IMPLEMENTED_SCALERS}"
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
            "The specified parent gate is lower or equal in the gating " +
            "hierarchy than the gate to display or is not a parent. " +
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

class PopulationAsGateError(Exception):
    def __init__(self,
                 gate: str):
        self.message = (
            f"You tried to access a gate path of {gate}. " +
            "There is no Gate Separator in the provided gate."
        )
        super().__init__(self.message)

class GateNameError(Exception):
    def __init__(self,
                 gate_separator):
        self.message = (
            f"To avoid ambiguities, please provide a gate without leading or trailing '{gate_separator}'"
        )
        super().__init__(self.message)

class ExhaustedGatePathError(Exception):
    def __init__(self,
                 n: int,
                 m: int):
        self.message = (
            f"You tried to access {n} elements of the gate path, " + 
            f"which has only {m} gates."
        )
        super().__init__(self.message)


class GateAmbiguityError(Exception):
    def __init__(self,
                 found_gates):
        self.message = (
            f"Multiple matching gates have been found ({found_gates}). " +
            f"Please select the appropriate gate by supplying a partial " +
            "gating strategy." 
        )
        super().__init__(self.message)

class GateNotFoundError(Exception):

    def __init__(self,
                 population):
        self.message = (
            f"The population {population} was neither " +
             "found in the gating strategy provided by a workspace " +
             "or in the user-provided gating strategy. To avoid that, " +
             "make sure that all populations that are referred to are either " +
             "in the gating strategy provided or pregated in a workspace."
        )
        super().__init__(self.message)

class ParentGateNotFoundError(GateNotFoundError):

    def __init__(self,
                 parent_population):
        super().__init__(parent_population)


class ClassifierNotImplementedError(Exception):

    def __init__(self,
                 classifier: Any,
                 implemented_classifiers: list[str]):
        self.message = (
            f"Classifier is not implemented. Please select one of {implemented_classifiers}, was {classifier}"
        )
        super().__init__(self.message)

class NotCompensatedError(Exception):

    def __init__(self):
        self.message = "File is not compensated. Please compensate first."
        super().__init__(self.message)

class InputDirectoryNotFoundError(Exception):

    def __init__(self):
        self.message = "The provided input directory was not found."
        super().__init__(self.message)
