import contextlib
import os
from typing import Union, Optional

from anndata import AnnData
from scipy.sparse import csr_matrix
import anndata as ad
import numpy as np
import pandas as pd

import gc

from ._sample import FCSFile
from ._supplements import Panel, Metadata
from ._workspaces import FlowJoWorkspace, DivaWorkspace

from ..exceptions._exceptions import InputDirectoryNotFoundError
from ..exceptions._supplements import SupplementDataTypeError, PanelMatchWarning
from ..gates.gating_strategy import GatingStrategy, GateTreeError
from ..synchronization._synchronize import _hash_dataset
from ..transforms._matrix import Matrix
from .._utils import (scatter_channels,
                      time_channels,
                      cytof_technical_channels,
                      spectral_flow_technical_channels)

def create_dataset(metadata: Metadata,
                   panel: Panel,
                   workspace: Union[FlowJoWorkspace, DivaWorkspace],
                   input_directory: Optional[str] = None,
                   subsample_fcs_to: Optional[int] = None,
                   truncate_max_range: bool = True,
                   keep_raw: bool = False) -> AnnData:
    """
    Creates the dataset.

    Parameters
    ----------

    metadata
        The metadata object of :class:`~FACSPy.Metadata` 
    panel
        The panel object of :class:`~FACSPy.Panel` 
    workspace
        The accompanying workspace of :class:`~FACSPy.FlowJoWorkspace`
    input_directory
        path that points to the FCS files. If no input directory is
        specified, the current working directory is assumed.
    subsample_fcs_to
        Parameter that specifies how many cells of an FCS file are
        read. Per default, all cells are read in
    truncate_max_range
        Parameter that controls if the FCS-File data should be truncated
        to their pnr value. Defaults to True.
    keep_raw
        Whether to keep the raw, uncompensated events. Defaults to False.

    Returns
    -------
    The dataset object of :class:`~anndata.AnnData`

    Examples
    --------

    >>> import FACSPy as fp
    >>> metadata = fp.dt.Metadata("metadata.csv") # creates the Metadata object

    >>> panel = fp.dt.Panel("panel.csv") # creates the Panel object

    >>> # alternatively, if the panel is stored in the FCS files:
    >>> panel = fp.create_panel_from_fcs() # assumes that FCS files are in the current working directory

    >>> workspace = fp.dt.FlowJoWorkspace("workspace.wsp") # creates the FlowJoWorkspace object

    >>> dataset = fp.create_dataset(
    ...    panel = panel,
    ...    metadata = metadata,
    ...    workspace = workspace,
    ...    subsample_fcs_to = 10_000,
    ...    truncate_max_range = True,
    ...    keep_raw = False
    ... )

    >>> dataset
    AnnData object with n_obs × n_vars = 615936 × 22
    obs: 'sample_ID', 'file_name', 'condition'
    var: 'pns', 'png', 'pne', 'pnr', 'type', 'pnn'
    uns: 'metadata', 'panel', 'workspace', 'gating_cols', 'dataset_status_hash'
    obsm: 'gating'
    layers: 'compensated'

    Notes
    -----

    See further explanation about the created dataset in the following tutorials:
    :doc:`/vignettes/dataset_structure`

    """

    if input_directory is None:
        input_directory = os.getcwd()
    
    if not os.path.exists(input_directory):
        raise InputDirectoryNotFoundError()

    if not isinstance(metadata, Metadata):
        raise SupplementDataTypeError(data_type = type(metadata),
                                      class_name = "Metadata")

    if not isinstance(panel, Panel):
        raise SupplementDataTypeError(data_type = type(panel),
                                      class_name = "Panel")

    if not isinstance(workspace, FlowJoWorkspace) or isinstance(workspace, DivaWorkspace):
        raise SupplementDataTypeError(data_type = type(workspace),
                                      class_name = "FlowJoWorkspace")

    if subsample_fcs_to is not None:
        if not isinstance(subsample_fcs_to, int) or isinstance(subsample_fcs_to, float):
            raise ValueError(
                (
                    "Please provide the subsample_fcs_to parameter as a number, " + 
                    f"it was {type(subsample_fcs_to)}"
                )
            )

    return DatasetAssembler(input_directory = input_directory,
                            metadata = metadata,
                            panel = panel,
                            workspace = workspace,
                            subsample_fcs_to = subsample_fcs_to,
                            truncate_max_range = truncate_max_range,
                            keep_raw = keep_raw).get_dataset()

class DatasetAssembler:

    """
    Class to assemble the initial dataset containing the FCS data.
    Class is not meant to be used by the user and is only called
    internally by fp.create_dataset.
    """

    def __init__(self,
                 input_directory: str,
                 metadata: Metadata,
                 panel: Panel,
                 workspace: Union[FlowJoWorkspace, DivaWorkspace],
                 subsample_fcs_to: Optional[int] = None,
                 truncate_max_range: bool = False,
                 keep_raw: bool = False) -> None:
        """automatically creates the dataset"""

        file_list: list[FCSFile] = self._fetch_fcs_files(input_directory,
                                                         metadata,
                                                         subsample_fcs_to,
                                                         truncate_max_range)
        
        self._append_comp_matrices(file_list,
                                   workspace)
        
        gates = self._gate_samples(file_list,
                                   workspace)

        file_list: list[FCSFile] = self._compensate_samples(file_list,
                                                            workspace,
                                                            keep_raw)

        gates = self._fill_empty_gates(file_list, gates)

        dataset_list = self._construct_dataset(file_list,
                                               metadata,
                                               panel)
        
        dataset = self._concatenate_dataset(dataset_list)

        dataset = self._append_supplements(dataset,
                                           metadata,
                                           panel,
                                           workspace)
        
        self.dataset = self._append_gates(dataset,
                                          gates)

        _hash_dataset(self.dataset)

        gc.collect()

    def _fill_empty_gates(self,
                          file_list: list[FCSFile],
                          gates: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """function that looks for ungated samples and appends DataFrames with pd.NA"""
        for i, (file, gate_table) in enumerate(zip(file_list, gates)):
            if gate_table.shape[0] == 0:
                gates[i] = pd.DataFrame(index = range(file.event_count))
        return gates

    def _append_gates(self,
                      dataset: AnnData,
                      gates: list[pd.DataFrame]) -> AnnData:
        """\
        function fills the .obsm["gating"] slot with the gating information
        and the .uns["gating_cols"] with the gate names
        """
        gatings = pd.concat(gates, axis = 0).fillna(0).astype("bool")
        dataset.obsm["gating"] = csr_matrix(gatings.values)
        dataset.uns["gating_cols"] = gatings.columns
        return dataset

    def _append_supplements(self,
                            dataset: AnnData,
                            metadata: Metadata,
                            panel: Panel,
                            workspace: Union[FlowJoWorkspace, DivaWorkspace]) -> AnnData:
        """
        function fills the .uns["metadata"] slot with the Metadata object,
        the .uns["panel"] slot with the Panel object and
        the .uns["workspace"] slot with the workspace object
        """
        dataset.uns["metadata"] = metadata
        dataset.uns["panel"] = panel
        dataset.uns["workspace"] = workspace.wsp_dict
        return dataset

    def get_dataset(self) -> AnnData:
        """returns the dataset"""
        return self.dataset

    def _create_gating_strategy(self,
                                file: FCSFile,
                                workspace: Union[FlowJoWorkspace, DivaWorkspace]) -> GatingStrategy:
        """creates a gating strategy from the workspace groups"""
        gating_strategy = GatingStrategy()
        file_containing_workspace_groups = [group for group in workspace.wsp_dict.keys()
                                            if file.original_filename in workspace.wsp_dict[group].keys()]

        for group in file_containing_workspace_groups:
            workspace_subset = workspace.wsp_dict[group][file.original_filename]
            for gate_dict in workspace_subset["gates"]:
                with contextlib.suppress(GateTreeError): ## if gate exists
                    gating_strategy.add_gate(gate_dict["gate"], gate_path = gate_dict["gate_path"])
            ### potential bug: if comp matrices/transformations are different per group
            if group == "All Samples":
                gating_strategy.add_comp_matrix(workspace_subset["compensation"])
                gating_strategy.transformations = {xform.id: xform for xform in workspace_subset["transforms"]}

        return gating_strategy

    def _gate_sample(self,
                     file: FCSFile,
                     workspace: Union[FlowJoWorkspace, DivaWorkspace]) -> pd.DataFrame:
        """gates a specific sample"""
        print(f"... gating sample {file.original_filename}")
        gating_strategy: GatingStrategy = self._create_gating_strategy(file, workspace)
        gating_results = gating_strategy.gate_sample(file)
        gate_table = pd.DataFrame(columns = ["/".join([path, gate]) for gate, path in gating_results._raw_results.keys()])
        for gate, path in gating_results._raw_results.keys():
            gate_table["/".join([path, gate])] = gating_results.get_gate_membership(gate_name = gate, gate_path = path)
        gate_table.columns = gate_table.columns.str.replace(" ", "_")

        return gate_table

    def _gate_samples(self,
                      file_list: list[FCSFile],
                      workspace: Union[FlowJoWorkspace, DivaWorkspace]) -> list[pd.DataFrame]:
        """gates all samples"""
        return [self._gate_sample(file, workspace) for file in file_list]

    def _concatenate_dataset(self,
                             file_list: list[AnnData]):
        """concatenates all singular anndata representations into the dataset"""
        return ad.concat(
            file_list,
            merge = "same",
            index_unique = "-",
            keys = range(len(file_list))
        )
    
    def _create_obs_from_metadata(self,
                                  file: FCSFile,
                                  metadata: Metadata) -> pd.DataFrame:
        """creates .obs dataframe from the Metadata object"""
        metadata_df = metadata.to_df()
        file_row = metadata_df.loc[metadata_df["file_name"] == file.original_filename]
        cell_number = file.event_count
        metadata_frame = pd.DataFrame(np.repeat(file_row.values,
                                                cell_number,
                                                axis = 0),
                                      columns = file_row.columns,
                                      dtype = "category")
        metadata_frame.index = metadata_frame.index.astype("str")
        return metadata_frame

    def _fetch_panel_antigen(self,
                             panel_channels: list[str],
                             channel: str,
                             panel_df: pd.DataFrame) -> Optional[str]:
        """retrieves the antigens from the panel"""
        if channel in panel_channels:
            return panel_df.loc[panel_df["fcs_colname"] == channel, "antigens"].item()
        else:
            return None
    
    def _get_final_antigen(self,
                           fcs_antigen: Optional[str],
                           panel_antigen: Optional[str],
                           channel: str) -> str:
        """\
        decides with antigen to use. If both an antigen
        was provided by the user via the Panel object as well
        as from the FCS file, a warning is raised and the user
        supplied antigen is kept.
        """
        
        if not panel_antigen:
            return fcs_antigen or channel
        
        if fcs_antigen and fcs_antigen != panel_antigen:
            PanelMatchWarning(channel, fcs_antigen, panel_antigen)
            return panel_antigen
        
        return panel_antigen

    def _fetch_fcs_antigen(self,
                           fcs_panel_df: pd.DataFrame,
                           channel: str) -> str:
        """retrieves the antigen as supplied by the FCS file"""
        return fcs_panel_df.loc[fcs_panel_df.index == channel, "pns"].item()
    
    def _create_var_from_panel(self,
                               file: FCSFile,
                               panel: Panel) -> pd.DataFrame:
        """fills the .var slot by using the FCS data and the user supplied panel"""
        panel_df = panel.to_df()
        fcs_panel_df = file.channels
        panel_channels: list[str] = panel_df["fcs_colname"].tolist()
        for channel in fcs_panel_df.index:
            fcs_antigen = self._fetch_fcs_antigen(fcs_panel_df, channel)
            panel_antigen = self._fetch_panel_antigen(panel_channels,
                                                      channel,
                                                      panel_df)
            fcs_panel_df.loc[fcs_panel_df.index == channel, "pns"] = self._get_final_antigen(fcs_antigen,
                                                                                             panel_antigen,
                                                                                             channel)
            
        fcs_panel_df = fcs_panel_df.drop("channel_numbers", axis = 1)
        fcs_panel_df.index = fcs_panel_df.index.astype("str")

        
        fcs_panel_df["type"] = ["scatter" if any(k in channel for k in scatter_channels)
                                else "time" if any(k in channel for k in time_channels)
                                else "technical" if any(k in channel for k in cytof_technical_channels + spectral_flow_technical_channels)
                                else "fluo"
                                for channel in fcs_panel_df.index]
        fcs_panel_df["pnn"] = fcs_panel_df.index.to_list()
        fcs_panel_df.index = fcs_panel_df["pns"].to_list()
        return fcs_panel_df

    def _create_anndata_representation(self,
                                       file: FCSFile,
                                       metadata: Metadata,
                                       panel: Panel) -> AnnData:
        """\
        creates an AnnData representation from the FCS data,
        the metadata and the panel
        """
        obs = self._create_obs_from_metadata(file,
                                             metadata)
        var = self._create_var_from_panel(file,
                                          panel)
        if hasattr(file, "original_events"):
            assert file.compensated_events is not None
            layers = {"raw": file.original_events.astype(np.float32),
                      "compensated": file.compensated_events.astype(np.float32)}
        else:
            assert file.compensated_events is not None
            layers = {"compensated": file.compensated_events.astype(np.float32)}

        return AnnData(X = None,
                       obs = obs,
                       var = var,
                       layers = layers)

    def _create_anndata_representations(self,
                                        file_list: list[FCSFile],
                                        metadata: Metadata,
                                        panel: Panel) -> list[AnnData]:
        """creates AnnData representations for all files"""
        return [self._create_anndata_representation(file, metadata, panel) for file in file_list]

    def _construct_dataset(self,
                           file_list: list[FCSFile],
                           metadata: Metadata,
                           panel: Panel) -> list[AnnData]:
        """constructs a list of AnnData representations for all FCS data"""
        return self._create_anndata_representations(file_list, metadata, panel)

    def _compensate_samples(self,
                            file_list: list[FCSFile],
                            workspace: Union[FlowJoWorkspace, DivaWorkspace],
                            keep_raw: bool) -> list[FCSFile]:
        """compensates all available samples"""
        return [self._compensate_sample(sample, workspace, keep_raw) for sample in file_list]

    def _compensate_sample(self,
                           sample: FCSFile,
                           workspace: Union[FlowJoWorkspace, DivaWorkspace],
                           keep_raw: bool) -> FCSFile:
        """Function finds compensation matrix and applies it to raw data"""
        print(f"... compensating sample {sample.original_filename}")
        comp_matrix = self._find_comp_matrix(sample, workspace)
        sample.compensated_events = comp_matrix.apply(sample)
        sample.compensation_status = "compensated"
        if not keep_raw:
            del sample.original_events
            #sample.original_events = None
        return sample

    def _find_comp_matrix(self,
                          file: FCSFile,
                          workspace: Union[FlowJoWorkspace, DivaWorkspace]) -> Matrix:
        """\
        Returns compensation matrix. If matrix is within the workspace,
        this matrix is used preferentially. Otherwise use the compensation matrix
        from the FCS file
        """
        return workspace.wsp_dict["All Samples"][file.original_filename]["compensation"]

    def _append_comp_matrices(self,
                              file_list: list[FCSFile],
                              workspace: Union[FlowJoWorkspace, DivaWorkspace]) -> None:
        """loops through files and appends the comp matrix to the workspace"""
        for file in file_list:
            self._append_comp_matrix(file, workspace)

    def _append_comp_matrix(self,
                            file: FCSFile,
                            workspace: Union[FlowJoWorkspace, DivaWorkspace]) -> None:
        """\
        If matrix is within the workspace, this matrix is used preferentially.
        Otherwise use the compensation matrix from the FCS file.
        """
        if not self._comp_matrix_within_workspace(file, workspace):
            workspace.wsp_dict["All Samples"][file.original_filename]["compensation"] = file.fcs_compensation
        return

    def _comp_matrix_within_workspace(self,
                                      file: FCSFile,
                                      workspace: Union[FlowJoWorkspace, DivaWorkspace]) -> bool:
        """returns True if there is a comp matrix supplied within the workspace"""
        return isinstance(
            workspace.wsp_dict["All Samples"][file.original_filename]["compensation"],
            Matrix
        )

    def _convert_fcs_to_FCSFile(self,
                                input_directory: str,
                                metadata_fcs_files: list[str],
                                subsample_fcs_to: Optional[int],
                                truncate_max_range) -> list[FCSFile]:
        """converts FCS raw data to FCSFile objects"""
        return [
            FCSFile(input_directory,
                    file_name,
                    subsample = subsample_fcs_to,
                    truncate_max_range = truncate_max_range)
            for file_name in metadata_fcs_files
        ]
        
    def _fetch_fcs_files(self,
                         input_directory: str,
                         metadata: Metadata,
                         subsample_fcs_to: Optional[int],
                         truncate_max_range) -> list[FCSFile]:
        """\
        fetches the FCS file names from the metadata object
        and converts to FCSFile objects
        """
        
        # sourcery skip: use-named-expression
        metadata_fcs_files = metadata.dataframe["file_name"].to_list()
        
        if metadata_fcs_files:
            return self._convert_fcs_to_FCSFile(input_directory,
                                                metadata_fcs_files,
                                                subsample_fcs_to,
                                                truncate_max_range)   
        
        available_fcs_files = [file for file in os.listdir(input_directory)
                                if file.endswith(".fcs")]
        metadata = self._append_empty_metadata(metadata,
                                               available_fcs_files)
        return self._convert_fcs_to_FCSFile(input_directory,
                                            available_fcs_files,
                                            subsample_fcs_to,
                                            truncate_max_range)
    
    def _append_empty_metadata(self,
                               metadata: Metadata,
                               fcs_files: list[str]) -> Metadata:
        """
        appends empty metadata based on the fcs files in
        the input directory
        """
        metadata.dataframe["file_name"] = fcs_files
        metadata.dataframe["sample_ID"] = range(1, len(fcs_files) + 1)
        return metadata
