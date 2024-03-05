from typing import Optional, Union, Mapping

import pandas as pd
from pandas.io.parsers.readers import TextFileReader
import numpy as np
import os
from ..exceptions._supplements import (SupplementInputTypeError,
                                      SupplementFileNotFoundError,
                                      SupplementCreationError,
                                      SupplementColumnError,
                                      SupplementNoInputDirectoryError)

class BaseSupplement:
    
    def __init__(self,
                 file: str = '',
                 data: Optional[pd.DataFrame] = None,
                 from_fcs: bool = False) -> None:
        
        self.source = self.fetch_data_source(file,
                                             data,
                                             from_fcs)

        self.dataframe = self.fetch_data_from_source(file,
                                                     data,
                                                     from_fcs)
        
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
                       file: str) -> pd.DataFrame:
        try:
            delimiter = self.fetch_delimiter(file)
            return pd.read_csv(file, sep = delimiter)
        except FileNotFoundError as e:
            raise SupplementFileNotFoundError(os.path.basename(file)) from e
    
    def fetch_delimiter(self,
                        file) -> str:
        reader: TextFileReader = pd.read_csv(file,
                                 sep = None,
                                 iterator = True,
                                 engine = "python")
        return reader._engine.data.dialect.delimiter    
    
    def fetch_data_source(self,
                          file: Optional[str],
                          data: Optional[pd.DataFrame],
                          from_fcs: bool) -> str:

        if from_fcs:
            return "read from fcs"
        elif data is not None:
            return "provided dataframe"
        elif file:
            return "provided file"
        else:
            raise SupplementCreationError(self.__class__.__name__)
    
    def validate_user_supplied_table(self,
                                     dataframe: pd.DataFrame,
                                     columns_to_check: list[str]) -> pd.DataFrame:
        if not isinstance(dataframe, pd.DataFrame):
            raise SupplementInputTypeError(data_type = type(dataframe),
                                          class_name = self.__class__.__name__)
        if any(k not in dataframe.columns for k in columns_to_check):
            for column in columns_to_check:
                if column not in dataframe.columns:
                    raise SupplementColumnError(column,
                                                self.__class__.__name__)

        return dataframe

    def fetch_data_from_source(self,
                               file: Optional[str],
                               data: Optional[pd.DataFrame],
                               from_fcs: bool) -> pd.DataFrame:
        if self.source == "provided dataframe":
            return data
            
        elif self.source == "provided file":
            return self.open_from_file(file)
        
        elif from_fcs: 
            if self.__class__.__name__ == "Panel":
                return pd.DataFrame(columns = ["fcs_colname", "antigens"])
            elif self.__class__.__name__ == "Metadata":
                return pd.DataFrame(columns = ["sample_ID", "file_name"])
            elif self.__class__.__name__ == "CofactorTable":
                return pd.DataFrame(columns = ["fcs_colname", "cofactors"])
    
    def rename_channel(self,
                       old_channel_name,
                       new_channel_name) -> None:
        self.dataframe.loc[self.dataframe["fcs_colname"] == old_channel_name, "fcs_colname"] = new_channel_name

    def select_channels(self,
                        channels: list[str]) -> None:
        if isinstance(self, Metadata):
            raise TypeError("Channels cannot be selected from metadata object")
        if not isinstance(channels, list):
            channels = [channels]
        if isinstance(self, Panel):
            self.dataframe = self.dataframe.loc[self.dataframe["antigens"].isin(channels)]
        if isinstance(self, CofactorTable):
            self.dataframe = self.dataframe.loc[self.dataframe["fcs_colname"].isin(channels)]

    def _remove_unnamed_columns(self):
        unnamed_columns = [col for col in self.dataframe.columns if "Unnamed:" in col]
        self.dataframe = self.dataframe.drop(unnamed_columns, axis = 1)

class Panel(BaseSupplement):
    
    """
    Panel class to represent and unify flow cytometry panel representations.
    The structure has to be at least two columns: fcs_colname with the channels
    and antigen with the custom antigen names.
    If the panel is supposed to be read from the FCS files directly, set the
    flag from_fcs. 
    """

    def __init__(self,
                 file: str = '',
                 panel: Optional[pd.DataFrame] = None,
                 from_fcs: bool = False) -> None:
        
        super().__init__(file = file,
                         data = panel,
                         from_fcs = from_fcs)
        
        self.dataframe = self.validate_user_supplied_table(self.dataframe,
                                                           ["fcs_colname", "antigens"])
        self._remove_unnamed_columns()
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
                 file: str = '',
                 metadata: Optional[pd.DataFrame] = None,
                 from_fcs: bool = False) -> None:

        if from_fcs and not file:
            raise SupplementNoInputDirectoryError
        
        super().__init__(file = file,
                         data = metadata,
                         from_fcs = from_fcs)
        self.dataframe: pd.DataFrame = self.validate_user_supplied_table(self.dataframe,
                                                                         ["sample_ID", "file_name"])
        self._remove_unnamed_columns()
        self.factors = self._extract_metadata_factors()
        self._manage_dtypes()
        self._make_dataframe_categorical()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(" +
            f"{len(self.dataframe)} entries with factors " +
            f"{self._extract_metadata_factors()})"
        )
    
    def _manage_dtypes(self):
        """collection of statements that manage dtypes. sample_IDs are strings"""
        self.dataframe["sample_ID"] = self.dataframe["sample_ID"].astype("str")
    
    def _make_dataframe_categorical(self):
        self.dataframe = self.dataframe.astype("category")
    
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
            self.dataframe[factor] = self.dataframe[factor].astype("float32")
        except ValueError as e:
            raise ValueError("Only numeric columns are supported") from e
        column = self.dataframe[factor]
        min_value, max_value = column.min(), column.max()
        intervals = np.arange(min_value, max_value + min_value, (max_value - min_value) / n_groups)
        self.dataframe[f"{factor}_grouped"] = pd.cut(column, intervals)
    
    def rename(self,
               current_name: str,
               new_name: str) -> None:
        self.dataframe[new_name] = self.dataframe[current_name]
        self.dataframe = self.dataframe.drop(current_name, axis = 1)

    def rename_factors(self,
                       column: Union[str, pd.Index],
                       replacement: Union[Mapping, list[Union[str, float, int]]]) -> None:
        
        if isinstance(replacement, dict):
            self.dataframe[column].replace(replacement,
                                           inplace = True)
        else:
            self.dataframe[column] = replacement

    def subset(self,
               column: str,
               values: list[Union[str, int]]) -> None:
        if not isinstance(values, list):
            values = [values]
        self.dataframe = self.dataframe.loc[self.dataframe[column].isin(values)]

    def _extract_metadata_factors(self):
        return [
            col
            for col in self.dataframe.columns
            if all(k not in col for k in ["sample_ID", "sample_id", "file_name", "staining"])
        ]

    def get_factors(self):
        return self._extract_metadata_factors()
    
    def _sanitize_categoricals(self):
        for column in self.dataframe:
            if isinstance(self.dataframe[column].dtype, pd.CategoricalDtype):
                self.dataframe[column] = self.dataframe[column].cat.remove_unused_categories()
        return


class CofactorTable(BaseSupplement):

    def __init__(self,
                 file: str = '',
                 cofactors: Optional[pd.DataFrame] = None,
                 from_fcs: bool = False) -> None:
        
        super().__init__(file = file,
                         data = cofactors,
                         from_fcs = from_fcs)
        
        self.dataframe = self.validate_user_supplied_table(self.dataframe,
                                                           ["fcs_colname", "cofactors"])
        self._remove_unnamed_columns()
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
        

