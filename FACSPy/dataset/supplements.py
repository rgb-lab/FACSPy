from typing import Optional

import pandas as pd
import os
from ..exceptions.exceptions import (SupplementDataTypeError,
                                     SupplementFileNotFoundError,
                                     SupplementCreationError,
                                     SupplementColumnError)


class BaseSupplement:
    
    def __init__(self,
                 input_directory: str = '',
                 file_name: str = '',
                 data: Optional[pd.DataFrame] = None,
                 from_fcs: bool = False) -> None:
        
        self.source = self.fetch_data_source(input_directory,
                                             file_name,
                                             data,
                                             from_fcs)

        self.dataframe = self.fetch_data_from_source(input_directory,
                                                     file_name,
                                                     data,
                                                     from_fcs)

    def strip_prefixes(self,
                       dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Channels from FlowJoWorkspaces get appended by "Comp-"
        or "FJComp" in case of data export. Since this is incompatible with
        the FCS naming, prefixes have to be stripped.
        """
        channel_list: list[str] = dataframe["fcs_colname"].to_list()
        dataframe["fcs_colname"] = [channel.split("Comp-")[1] if "Comp-" in channel else channel for channel in channel_list]
        return dataframe
    
    def to_df(self):
        """returns the dataframe of the object"""
        return self.dataframe

    def open_from_file(self,
                       input_directory: Optional[str],
                       file_name: str) -> pd.DataFrame:
        try:
            delimiter = self.fetch_delimiter(input_directory, file_name)
            return pd.read_csv(os.path.join(input_directory, file_name), sep = delimiter)
        except FileNotFoundError as e:
            raise SupplementFileNotFoundError(file_name) from e
    
    def fetch_delimiter(self,
                        input_directory,
                        file_name) -> str:
        reader = pd.read_csv(os.path.join(input_directory, file_name),
                             sep = None,
                             iterator = True,
                             engine = "python")
        return reader._engine.data.dialect.delimiter    
    
    def fetch_data_source(self,
                          input_directory: Optional[str],
                          file_name: Optional[str],
                          data: Optional[pd.DataFrame],
                          from_fcs: bool) -> str:

        if data is not None:
            return "provided dataframe"
        elif file_name:
            return "provided file"
        elif from_fcs:
            return "read from fcs"
        else:
            raise SupplementCreationError(self.__class__.__name__)
    
    def validate_user_supplied_table(self,
                                     dataframe: pd.DataFrame,
                                     columns_to_check: list[str]) -> pd.DataFrame:
        if not isinstance(dataframe, pd.DataFrame):
            raise SupplementDataTypeError(data_type = type(dataframe),
                                          class_name = self.__class__.__name__)
        if any(k not in dataframe.columns for k in columns_to_check):
            for column in columns_to_check:
                if column not in dataframe.columns:
                    raise SupplementColumnError(column,
                                                self.__class__.__name__)

        return dataframe

    def fetch_data_from_source(self,
                               input_directory: Optional[str],
                               file_name: Optional[str],
                               data: Optional[pd.DataFrame],
                               from_fcs: bool) -> pd.DataFrame:
        if self.source == "provided dataframe":
            return data
            
        elif self.source == "provided file":
            return self.open_from_file(input_directory, file_name)
        
        elif from_fcs: 
            if self.__class__.__name__ == "Panel":
                return pd.DataFrame(columns = ["fcs_colname", "antigens"])
            elif self.__class__.__name__ == "Metadata":
                return pd.DataFrame(columns = ["sample_ID", "file_name"])
            elif self.__class__.__name__ == "CofactorTable":
                return pd.DataFrame(columns = ["fcs_colname", "cofactors"])


class Panel(BaseSupplement):
    
    """
    Panel class to represent and unify flow cytometry panel representations.
    The structure has to be at least to columns: fcs_colname with the channels
    and antigen with the custom antigen names.
    If the panel is supposed to be read from the FCS files directly, set the
    flag from_fcs. 
    """

    def __init__(self,
                 input_directory: str = '',
                 file_name: str = '',
                 panel: Optional[pd.DataFrame] = None,
                 from_fcs: bool = False) -> None:
        
        super().__init__(input_directory = input_directory,
                         file_name = file_name,
                         data = panel,
                         from_fcs = from_fcs)
        
        self.dataframe = self.validate_user_supplied_table(self.dataframe,
                                                           ["fcs_colname", "antigens"])
        self.dataframe = self.strip_prefixes(self.dataframe)
    

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(" +
            f"{len(self.dataframe)} channels, "+
            f"loaded as {self.source})"
        )  


class Metadata(BaseSupplement):
    """
    Metadata class to represent and unify flow cytometry metadata representations.
    The structure has to be at least to columns: sample_ID with ascending ints
    and file_name with the file names.
    If the metadata are supposed to be constructed from the read-in files directly, 
    set the flag from_fcs to True. 
    """ 
    def __init__(self,
                 input_directory: str = '',
                 file_name: str = '',
                 metadata: Optional[pd.DataFrame] = None,
                 from_fcs: bool = False) -> None:
        
        super().__init__(input_directory = input_directory,
                         file_name = file_name,
                         data = metadata,
                         from_fcs = from_fcs)
        
        self.dataframe = self.validate_user_supplied_table(self.dataframe,
                                                           ["sample_ID", "file_name"])
        self.factors = self.extract_metadata_factors()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(" +
            f"{len(self.dataframe)} entries with factors " +
            f"{self.factors})"
        )
    
    def extract_metadata_factors(self):
        return [
            col
            for col in self.dataframe.columns
            if all(k not in col for k in ["sample_ID", "sample_id", "file_name", "staining"])
        ]



class CofactorTable(BaseSupplement):

    def __init__(self,
                 input_directory: str = '',
                 file_name: str = '',
                 cofactors: Optional[pd.DataFrame] = None,
                 from_fcs: bool = False) -> None:
        
        super().__init__(input_directory = input_directory,
                         file_name = file_name,
                         data = cofactors,
                         from_fcs = from_fcs)
        
        self.dataframe = self.validate_user_supplied_table(self.dataframe,
                                                           ["fcs_colname", "cofactors"])
        self.dataframe = self.strip_prefixes(self.dataframe)
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(" +
            f"{self.dataframe.shape[0]} channels, "+
            f"loaded as {self.source})"
        ) 
    
    def get_cofactor(self,
                     channel_name) -> float:
        return self.dataframe.loc[self.dataframe["fcs_colname"] == channel_name, "cofactors"].iloc[0]
    
    def set_cofactor(self,
                     channel_name,
                     cofactor) -> None:
        self.dataframe.loc[self.dataframe["fcs_colname"] == channel_name, "cofactors"] = cofactor

