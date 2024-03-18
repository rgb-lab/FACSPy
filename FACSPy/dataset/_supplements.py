import numpy as np
import os
import pandas as pd

from pandas.io.parsers.readers import TextFileReader

from typing import Optional, Union, Mapping

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
        
        self.source = self._fetch_data_source(file,
                                              data,
                                              from_fcs)

        self.dataframe = self._fetch_data_from_source(file,
                                                      data,
                                                      from_fcs)
        
    def write(self,
              output_directory: Union[str, os.PathLike] = ''):
        """writes the underlying table to disk."""
        self.dataframe.to_csv(output_directory, index = False)    

    def _strip_prefixes(self,
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

    def _open_from_file(self,
                        file: str) -> pd.DataFrame:
        """opens a file and returns a pd.DataFrame"""
        try:
            delimiter = self._fetch_delimiter(file)
            return pd.read_csv(file, sep = delimiter)
        except FileNotFoundError as e:
            raise SupplementFileNotFoundError(os.path.basename(file)) from e
    
    def _fetch_delimiter(self,
                         file) -> str:
        """pandas finds the correct delimiter"""
        reader: TextFileReader = pd.read_csv(file,
                                             sep = None,
                                             iterator = True,
                                             engine = "python")
        return reader._engine.data.dialect.delimiter    
    
    def _fetch_data_source(self,
                           file: Optional[str],
                           data: Optional[pd.DataFrame],
                           from_fcs: bool) -> str:
        """
        logic to find the data source.
        if from_fcs is set, this will override everything,
        if data are supplied, these are used,
        if a file was provided the file is opened
        """

        if from_fcs:
            return "read from fcs"
        elif data is not None:
            return "provided dataframe"
        elif file:
            return "provided file"
        else:
            raise SupplementCreationError(self.__class__.__name__)
    
    def _validate_user_supplied_table(self,
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

    def _fetch_data_from_source(self,
                                file: Optional[str],
                                data: Optional[pd.DataFrame],
                                from_fcs: bool) -> pd.DataFrame:
        """actually converts the operations set in `fetch_data_source`"""
        if self.source == "provided dataframe":
            return data
            
        elif self.source == "provided file":
            return self._open_from_file(file)
        
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
        """renames a channel"""
        self.dataframe.loc[self.dataframe["fcs_colname"] == old_channel_name, "fcs_colname"] = new_channel_name

    def select_channels(self,
                        channels: list[str]) -> None:
        """selects channels and subsets dataframe"""
        if isinstance(self, Metadata):
            raise TypeError("Channels cannot be selected from metadata object")
        if not isinstance(channels, list):
            channels = [channels]
        if isinstance(self, Panel):
            self.dataframe = self.dataframe.loc[self.dataframe["antigens"].isin(channels),:]
        if isinstance(self, CofactorTable):
            self.dataframe = self.dataframe.loc[self.dataframe["fcs_colname"].isin(channels),:]

    def _remove_unnamed_columns(self):
        """removes unnamed columns that happen upon pandas import"""
        unnamed_columns = [col for col in self.dataframe.columns if "Unnamed:" in col]
        self.dataframe = self.dataframe.drop(unnamed_columns, axis = 1)

class Panel(BaseSupplement):
    
    """\
    Panel class to represent and unify cytometry panel representations.
    The structure has to be at least two columns: `fcs_colname` with the channels
    (e.g. `BV421-A`) and `antigens` with the custom antigen names (e.g. `CD3`).
    If the panel is supposed to be read from the FCS files directly, set the
    flag from_fcs. The will create a completely empty panel file that is
    later filled by the function fp.create_dataset().

    Parameters
    ----------

    file
        The path or filename pointing to the table. Can be .txt, .csv.
    panel
        Optional. If the dataframe has been assembled with pandas,
        supply this panel dataframe.
    from_fcs
        If True, returns an empty Panel object which will be filled
        during dataset assembly by fp.create_dataset().

    Returns
    -------

    The dataset object of :class:`~FACSPy.dataset._supplements.Panel`


    Examples
    --------

    >>> import FACSPy as fp
    >>> import pandas as pd

    >>> panel = fp.dt.panel("panel.csv") # creates a panel from the local file `panel.csv`
    >>> panel
    Panel(12 channels, loaded as provided file)

    >>> panel_frame = pd.DataFrame(
    ...     data = {
    ...         "fcs_colname": ["FSC-A", "SSC-A", "BV421-A"],
    ...         "antigens" : ["FSC-A", "SSC-A", "CD3"]
    ...     },
    ...     index = list(range(3))
    ... )
    >>> panel = fp.dt.Panel(panel = panel_frame) # creates a panel from a pd.DataFrame
    >>> panel
    Panel(3 channels, loaded as provided dataframe)

    >>> panel = fp.dt.Panel(from_fcs = True) # creates an empty Panel
    Panel(0 channels, loaded as read from FCS)

    Notes
    -----

    See further usage examples in the following tutorials:
    ### NEEDS DOCSTRING LINKS!
    vignettes/panel_vignette.ipynb

    """

    def __init__(self,
                 file: str = '',
                 panel: Optional[pd.DataFrame] = None,
                 from_fcs: bool = False) -> None:
        
        super().__init__(file = file,
                         data = panel,
                         from_fcs = from_fcs)
        
        self.dataframe = self._validate_user_supplied_table(self.dataframe,
                                                            ["fcs_colname", "antigens"])
        self._remove_unnamed_columns()
        self.dataframe = self._strip_prefixes(self.dataframe)
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(" +
            f"{len(self.dataframe)} channels, "+
            f"loaded as {self.source})"
        ) 

    def get_antigens(self):
        """returns the antigens from the panel as a list"""
        return self.dataframe["antigens"].to_list()
    
    def get_channels(self):
        """returns the channel names from the panel as a list"""
        return self.dataframe["fcs_colname"].to_list()
    

class Metadata(BaseSupplement):
    """\
    Metadata class to represent and unify cytometry metadata representations.
    The structure has to be at least to columns: sample_ID with ascending ints
    and file_name with the file names.
    If the metadata are supposed to be constructed from the read-in files directly, 
    set the flag from_fcs to True. 
    
    Parameters
    ----------

    file
        The path or filename pointing to the table. Can be .txt, .csv.
    panel
        Optional. If the dataframe has been assembled with pandas,
        supply this metadata dataframe.
    from_fcs
        If True, returns an empty Metadata object which will be filled
        during dataset assembly by fp.create_dataset().

    Returns
    -------

    The dataset object of :class:`~FACSPy.dataset._supplements.Metadata`


    Examples
    --------

    >>> import FACSPy as fp
    >>> import pandas as pd

    >>> metadata = fp.dt.Metadata("metadata.csv") # creates metadata from the local file `metadata.csv`
    >>> metadata
    Metadata(28 entries with factors ["condition", "organ"])

    >>> metadata_frame = pd.DataFrame(
    ...     data = {
    ...         "sample_ID": ["1", "2", "3"],
    ...         "file_name" : ["1.fcs", "2.fcs", "3.fcs"],
    ...         "condition" : ["healthy", "healthy", "disease"]
    ...     },
    ...     index = list(range(3))
    ... )
    >>> metadata = fp.dt.Metadata(metadata = metadata_frame) # creates metadata from a pd.DataFrame
    >>> metadata
    Metadata(3 entries with factors ["condition"])

    >>> metadata = fp.dt.Metadata(from_fcs = True) # creates an empty metadata object
    Metadata(0 entries with factors [])

    Notes
    -----

    See further usage examples in the following tutorials:
    ### NEEDS DOCSTRING LINKS!
    vignettes/metadata_vignette.ipynb
    
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
        self.dataframe: pd.DataFrame = self._validate_user_supplied_table(self.dataframe,
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
        """all columns are converted to categoricals"""
        self.dataframe = self.dataframe.astype("category")
    
    def annotate(self,
                 file_names: Union[str, list[str]],
                 column: str,
                 value: str) -> None:
        """allows the annotation of new metadata"""
        if not isinstance(file_names, list):
            file_names = [file_names]
        self.dataframe.loc[self.dataframe["file_name"].isin(file_names), column] = value

    def group_variable(self,
                       factor: str,
                       n_groups: int):
        """groups continous variables into n_groups"""
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
        """renames a column from the metadata dataframe"""
        self.dataframe[new_name] = self.dataframe[current_name]
        self.dataframe = self.dataframe.drop(current_name, axis = 1)

    def rename_factors(self,
                       column: Union[str, pd.Index],
                       replacement: Union[Mapping, list[Union[str, float, int]]]) -> None:
        """renames the instances of a factor"""
        if isinstance(replacement, dict):
            self.dataframe[column].replace(replacement,
                                           inplace = True)
        else:
            self.dataframe[column] = replacement

    def subset(self,
               column: str,
               values: list[Union[str, int]]) -> None:
        """subsets the metadata based on metadata values"""
        if not isinstance(values, list):
            values = [values]
        self.dataframe = self.dataframe.loc[self.dataframe[column].isin(values)]

    def _extract_metadata_factors(self):
        """returns all metadata columns that are not `sample_ID`, `file_name` or `staining`"""
        return [
            col
            for col in self.dataframe.columns
            if all(k not in col for k in ["sample_ID", "sample_id", "file_name", "staining"])
        ]

    def get_factors(self):
        """returns all metadata columns that are not `sample_ID`, `file_name` or `staining`"""
        return self._extract_metadata_factors()
    
    def _sanitize_categoricals(self):
        """removes unused categories"""
        for column in self.dataframe:
            if isinstance(self.dataframe[column].dtype, pd.CategoricalDtype):
                self.dataframe[column] = self.dataframe[column].cat.remove_unused_categories()
        return


class CofactorTable(BaseSupplement):
    """\
    CofactorTable class to represent and unify cytometry cofactor representations.
    The structure has to be at least two columns: `fcs_colname` with the antigen names
    (e.g. `CD3`) and `cofactors` with the corresponding cofactors.
    If the table is supposed to be constructed from the read-in files directly, 
    set the flag from_fcs to True. 
    
    Parameters
    ----------

    file
        The path or filename pointing to the table. Can be .txt, .csv.
    cofactors
        Optional. If the dataframe has been assembled with pandas,
        supply this metadata dataframe.
    from_fcs
        If True, returns an empty CofactorTable object which will be filled
        during cofactor_calculation by fp.calculate_cofactors().

    Returns
    -------

    The dataset object of :class:`~FACSPy.dataset._supplements.CofactorTable`

    Examples
    --------

    >>> import FACSPy as fp
    >>> import pandas as pd

    >>> cof_table = fp.dt.CofactorTable("cofactors.csv") # creates table from the local file `cofactors.csv`
    >>> cof_table
    CofactorTable(12 channels, loaded as provided file)

    >>> cof_table_frame = pd.DataFrame(
    ...     data = {
    ...         "fcs_colname": ["CD3", "CD4", "CD5"],
    ...         "cofactors" : [200, 400, 200],
    ...     },
    ...     index = list(range(3))
    ... )
    >>> cof_table = fp.dt.CofactorTable(cofactors = cof_table_frame) # creates table from a pd.DataFrame
    >>> cof_table
    CofactorTable(3 channels, loaded as provided dataframe)

    Notes
    -----

    See further usage examples in the following tutorials:
    ### NEEDS DOCSTRING LINKS!
    vignettes/cofactor_table_vignette.ipynb

    """

    def __init__(self,
                 file: str = '',
                 cofactors: Optional[pd.DataFrame] = None,
                 from_fcs: bool = False) -> None:
        
        super().__init__(file = file,
                         data = cofactors,
                         from_fcs = from_fcs)
        
        self.dataframe = self._validate_user_supplied_table(self.dataframe,
                                                            ["fcs_colname", "cofactors"])
        self._remove_unnamed_columns()
        self.dataframe = self._strip_prefixes(self.dataframe)
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(" +
            f"{self.dataframe.shape[0]} channels, "+
            f"loaded as {self.source})"
        ) 
    
    def get_cofactor(self,
                     channel_name: str) -> float:
        """returns cofactor of a given channel"""
        return self.dataframe.loc[self.dataframe["fcs_colname"] == channel_name, "cofactors"].iloc[0]
    
    def set_cofactor(self,
                     channel_name: str,
                     cofactor: Union[int, float]) -> None:
        """sets cofactor of a given channel"""
        self.dataframe.loc[self.dataframe["fcs_colname"] == channel_name, "cofactors"] = cofactor

    def set_columns(self,
                    columns: list[str]) -> None:
        """converts a column to fcs_colname column"""
        self.dataframe["fcs_colname"] = columns
    
    def set_cofactors(self,
                      cofactors: Optional[list[Union[str, int]]] = None,
                      cytof: bool = False) -> None:
        """sets cofactors. If cytof == True, cofactors are set to 5"""
        if cofactors is None and not cytof:
            raise ValueError("Please provide a list of cofactors or set the cytof flag to True")
        if cytof and cofactors is not None:
            print("... warning cytof flag has been set to True, cofactors will be 5 for each channel.")
        self.dataframe["cofactors"] = 5 if cytof else cofactors       