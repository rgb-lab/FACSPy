from typing import Optional, Union

import pandas as pd
import numpy as np
import os
from ..exceptions.supplements import (SupplementDataTypeError,
                                      SupplementFileNotFoundError,
                                      SupplementCreationError,
                                      SupplementColumnError,
                                      SupplementNoInputDirectoryError)

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
        
        self.input_directory = input_directory

    def write(self,
              output_directory: Union[str, os.PathLike] = ''):
        self.dataframe.to_csv(output_directory, index = False)    

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
    The structure has to be at least two columns: fcs_colname with the channels
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

    def get_antigens(self):
        return self.dataframe["antigens"].to_list()
    
    def get_channels(self):
        return self.dataframe["fcs_colname"].to_list()


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

        if from_fcs:
            if input_directory:
                self.append_metadata_from_folder(input_directory)

            else:
                raise SupplementNoInputDirectoryError
        
        self.factors = self.extract_metadata_factors()

        self.make_dataframe_categorical()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(" +
            f"{len(self.dataframe)} entries with factors " +
            f"{self.factors})"
        )
    
    def make_dataframe_categorical(self):
        self.dataframe = self.dataframe.astype("category")
    
    def append_metadata_from_folder(self,
                                    input_directory) -> None:
        files: list[str] = os.listdir(input_directory)
        fcs_files = [file for file in files if file.endswith(".fcs")]
        self.dataframe["file_name"] = fcs_files
        self.dataframe["sample_ID"] = range(1,len(fcs_files)+1)

    def annotate(self,
                 file_names: Union[str, list[str]],
                 column: str,
                 value: str) -> None:
        if not isinstance(file_names, list):
            file_names = [file_names]
        self.dataframe.loc[self.dataframe["file_name"].isin(file_names), column] = value

    def group_variable(self,
                       factor: str,
                       n_groups: int):
        try:
            self.dataframe[factor] = self.dataframe[factor].astype("float64")
        except ValueError as e:
            raise ValueError("Only numeric columns are supported") from e
        column = self.dataframe[factor]
        min_value = column.min()
        max_value = column.max()
        intervals = np.arange(min_value, max_value + min_value, (max_value - min_value) / n_groups)
        self.dataframe[f"{factor}_grouped"] = pd.cut(column, intervals)

    def extract_metadata_factors(self):
        return [
            col
            for col in self.dataframe.columns
            if all(k not in col for k in ["sample_ID", "sample_id", "file_name", "staining"])
        ]

    def get_factors(self):
        return self.extract_metadata_factors()


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
                     channel_name: str) -> float:
        return self.dataframe.loc[self.dataframe["fcs_colname"] == channel_name, "cofactors"].iloc[0]
    
    def set_cofactor(self,
                     channel_name: str,
                     cofactor: Union[int, float]) -> None:
        self.dataframe.loc[self.dataframe["fcs_colname"] == channel_name, "cofactors"] = cofactor

    def set_columns(self,
                    columns: list[str]) -> None:
        self.dataframe["fcs_colname"] = columns
    
    def set_cofactors(self,
                      cofactors: Optional[list[Union[str, int]]] = None,
                      cytof: bool = False) -> None:
        if cofactors is None and not cytof:
            raise ValueError("Please provide a list of cofactors or set the cytof flag to True")
        if cytof and cofactors is not None:
            print("... warning cytof flag has been set to True, cofactors will be 5 for each channel.")
        self.dataframe["cofactors"] = 5 if cytof else cofactors       
        

