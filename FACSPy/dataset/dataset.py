import os
from typing import Union, Optional

import anndata as ad
import numpy as np
import pandas as pd

from .supplements import Panel, Metadata
from .workspaces import FlowJoWorkspace, DivaWorkspace
from .sample import FCSFile
from ..transforms._matrix import Matrix
from ..exceptions.exceptions import PanelMatchWarning

class DatasetAssembler:

    """
    Class to assemble the initial dataset
    containing the compensated data.
    """

    def __init__(self,
                 input_directory: str,
                 metadata: Metadata,
                 panel: Panel,
                 workspace: Union[FlowJoWorkspace, DivaWorkspace]) -> ad.AnnData:

        file_list: list[FCSFile] = self.fetch_fcs_files(input_directory,
                                                        metadata)
        file_list: list[FCSFile] = self.compensate_samples(file_list,
                                                           workspace)
        file_list: list[FCSFile] = self.gate_samples(file_list,
                                                     workspace)

        dataset_list = self.construct_dataset(file_list,
                                              metadata,
                                              panel)
        self.dataset = self.concatenate_dataset(dataset_list)

        self.dataset.obs = self.dataset.obs.astype("category")

    def gate_samples(self,
                     file_list: list[FCSFile],
                     workspace: Union[FlowJoWorkspace, DivaWorkspace]) -> list[FCSFile]:
        return file_list

    def concatenate_dataset(self,
                            file_list: list[ad.AnnData]):
        return file_list[0].concatenate(file_list[1:],
                                        batch_key = None)
    
    def create_obs_from_metadata(self,
                                 file: FCSFile,
                                 metadata: Metadata) -> pd.DataFrame:
        metadata_df = metadata.to_df()
        file_row = metadata_df.loc[metadata_df["file_name"] == file.original_filename]
        cell_number = file.original_events.shape[0]
        metadata_frame = pd.DataFrame(np.repeat(file_row.values,
                                      cell_number,
                                      axis = 0),
                                      columns = file_row.columns)
        metadata_frame.index = metadata_frame.index.astype("str")
        return metadata_frame

    def fetch_panel_antigen(self,
                            panel_channels: pd.Series,
                            channel: str,
                            panel_df: pd.DataFrame) -> Optional[str]:
        if channel in panel_channels:
            return panel_df.loc[panel_df["fcs_colname"] == channel, "antigens"].item()
        else:
            return None
    
    def get_final_antigen(self,
                          fcs_antigen: Optional[str],
                          panel_antigen: Optional[str],
                          channel: str) -> str:
        
        if not panel_antigen:
            return fcs_antigen or channel
        
        if fcs_antigen and fcs_antigen != panel_antigen:
            PanelMatchWarning(channel, fcs_antigen, panel_antigen)
            return panel_antigen
        
        return panel_antigen

    def fetch_fcs_antigen(self,
                          fcs_panel_df: pd.DataFrame,
                          channel: str):
        return fcs_panel_df.loc[fcs_panel_df.index == channel, "pns"].item()
    
    def create_var_from_panel(self,
                              file: FCSFile,
                              panel: Panel) -> pd.DataFrame:
        
        """Logic to compare FCS metadata and the provided panel"""
        panel_df = panel.to_df()
        fcs_panel_df = file.channels

        panel_channels = panel_df["fcs_colname"].to_list()
        for channel in fcs_panel_df.index:
            fcs_antigen = self.fetch_fcs_antigen(fcs_panel_df, channel)
            panel_antigen = self.fetch_panel_antigen(panel_channels,
                                                     channel,
                                                     panel_df)
            fcs_panel_df.loc[fcs_panel_df.index == channel, "pns"] = self.get_final_antigen(fcs_antigen,
                                                                                            panel_antigen,
                                                                                            channel)
            
        fcs_panel_df = fcs_panel_df.drop("channel_numbers", axis = 1)
        fcs_panel_df.index = fcs_panel_df.index.astype("str")
        
        return fcs_panel_df

    def create_anndata_representation(self,
                                      file: FCSFile,
                                      metadata: Metadata,
                                      panel: Panel) -> ad.AnnData:
        obs = self.create_obs_from_metadata(file,
                                            metadata)
        var = self.create_var_from_panel(file,
                                         panel)
        return ad.AnnData(X = None,
                          obs = obs,
                          var = var,
                          layers = {"raw": file.original_events.astype(np.float32),
                                    "compensated": file.compensated_events.astype(np.float32)})

    def create_anndata_representations(self,
                                       file_list: list[FCSFile],
                                       metadata: Metadata,
                                       panel: Panel) -> list[ad.AnnData]:
        return [self.create_anndata_representation(file, metadata, panel) for file in file_list]
        

    def construct_dataset(self,
                          file_list: list[FCSFile],
                          metadata: Metadata,
                          panel: Panel) -> ad.AnnData:
        return self.create_anndata_representations(file_list, metadata, panel)

    def compensate_samples(self,
                           file_list: list[FCSFile],
                           workspace: Union[FlowJoWorkspace, DivaWorkspace]) -> list[FCSFile]:
        return [self.compensate_sample(sample, workspace) for sample in file_list]

    def compensate_sample(self,
                          sample: FCSFile,
                          workspace: Union[FlowJoWorkspace, DivaWorkspace]) -> FCSFile:
        """Function finds compensation matrix and applies it to raw data"""
        comp_matrix = self.find_comp_matrix(sample, workspace)
        sample.compensated_events = comp_matrix.apply(sample)
        sample.compensation_status = "compensated"
        return sample

    def find_comp_matrix(self,
                         file: FCSFile,
                         workspace: Union[FlowJoWorkspace, DivaWorkspace]) -> Matrix:
        """Returns compensation matrix. If matrix is within the workspace,
        this matrix is used preferentially. Otherwise use the compensation matrix
        from the FCS file"""
        if self.comp_matrix_within_workspace(file, workspace):
            return workspace.wsp_dict["All Samples"][file.original_filename]["compensation"]
        else:
            return file.fcs_compensation    

    def comp_matrix_within_workspace(self,
                                     file: FCSFile,
                                     workspace: Union[FlowJoWorkspace, DivaWorkspace]) -> bool:
        return isinstance(workspace.wsp_dict["All Samples"][file.original_filename]["compensation"], Matrix)
    

    def convert_fcs_to_FCSFile(self,
                               input_directory: str,
                               metadata_fcs_files: list[str]) -> list[FCSFile]:
        return [FCSFile(input_directory, file_name) for file_name in metadata_fcs_files]
    
    def fetch_fcs_files(self,
                        input_directory: str,
                        metadata: Metadata) -> list[FCSFile]:
        metadata_fcs_files = metadata.dataframe["file_name"].to_list()
        
        if metadata_fcs_files:
            return self.convert_fcs_to_FCSFile(input_directory, metadata_fcs_files)   
        
        available_fcs_files = [file for file in os.listdir(input_directory)
                                if file.endswith(".fcs")]
        return self.convert_fcs_to_FCSFile(input_directory, available_fcs_files)
        