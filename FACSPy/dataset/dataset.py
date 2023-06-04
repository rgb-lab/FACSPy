import contextlib
import os
from typing import Union, Optional

import anndata as ad
from anndata import AnnData
import numpy as np
import pandas as pd
import scipy.signal as scs
from scipy.sparse import csr_matrix

from .supplements import Panel, Metadata, CofactorTable
from .workspaces import FlowJoWorkspace, DivaWorkspace
from .sample import FCSFile
from .utils import (find_corresponding_control_samples,
                    get_histogram_curve,
                    transform_data_array,
                    create_sample_subset_with_controls)

from ..transforms._matrix import Matrix
from ..exceptions.exceptions import PanelMatchWarning
from ..gates.gating_strategy import GatingStrategy, GateTreeError
from ..utils import fetch_fluo_channels

class Transformer:

    def __init__(self,
                 dataset: AnnData,
                 cofactor_table: Optional[CofactorTable] = None) -> None:
        ### Notes: Takes approx. (80-100.000 cells * 17 channels) / second 
        if not cofactor_table:
            cofactor_table, raw_cofactor_table = self.calculate_cofactors(dataset)
            dataset.uns["raw_cofactors"] = raw_cofactor_table
        
        dataset.uns["cofactors"] = cofactor_table    
        self.dataset = self.transform_dataset(dataset, cofactor_table)

    def calculate_cofactors(self,
                            dataset: AnnData) -> tuple[CofactorTable, pd.DataFrame]:
        
        (stained_samples,
         corresponding_control_samples) = find_corresponding_control_samples(dataset,
                                                                             by = "file_name")
        cofactors = {}
        for sample in stained_samples:
            cofactors[sample] = {}
            fluo_channels = fetch_fluo_channels(dataset)
            sample_subset = create_sample_subset_with_controls(dataset,
                                                               sample,
                                                               corresponding_control_samples,
                                                               match_cell_number = True)
            for channel in fluo_channels:
                data_array = sample_subset[:, sample_subset.var.index == channel].layers["compensated"]
                cofactor_stained_sample = self.estimate_cofactor_on_stained_sample(data_array,
                                                                                   200)
                if corresponding_control_samples[sample]:
                    control_sample = sample_subset[sample_subset.obs["staining"] != "stained", sample_subset.var.index == channel]
                    data_array = control_sample.layers["compensated"]
                    cofactor_unstained_sample = self.estimate_cofactor_on_unstained_sample(data_array, 20)
                    cofactor_by_percentile = self.estimate_cofactor_from_control_quantile(control_sample)
                
                    cofactors[sample][channel] = np.mean([cofactor_stained_sample,
                                                          cofactor_unstained_sample,
                                                          cofactor_by_percentile])
                    
                    continue
                cofactors[sample][channel] = cofactor_stained_sample
        return self.create_cofactor_tables(cofactors)

    def create_cofactor_tables(self,
                               cofactors: dict[str, list[float]],
                               reduction_method: str = "mean") -> tuple[CofactorTable, pd.DataFrame]:
        raw_table = pd.DataFrame(data = cofactors).T
        if reduction_method == "mean":
            reduced = pd.DataFrame(cofactors).mean(axis = 1)
        elif reduction_method == "median":
            reduced = pd.DataFrame(cofactors).median(axis = 1)
        reduced_table = pd.DataFrame({"fcs_colname": reduced.index,
                                      "cofactors": reduced.values})
        return CofactorTable(cofactors = reduced_table), raw_table

    def only_one_peak(self,
                      peaks: np.ndarray) -> bool:
        return peaks.shape[0] == 1

    def two_peaks(self,
                  peaks: np.ndarray) -> bool:
        return peaks.shape[0] == 2

    def estimate_cofactor_on_stained_sample(self,
                                            data_array: np.ndarray,
                                            cofactor: int) -> float:
        data_array = transform_data_array(data_array, cofactor)
        x, curve = get_histogram_curve(data_array)
        
        peak_output = scs.find_peaks(curve, prominence = 0.001, height = 0.01)
        peaks: np.ndarray = peak_output[0] ## array with the locs of found peaks
        peak_characteristics: dict = peak_output[1]

        if peaks.shape[0] >= 2: ## more than two peaks have been found it needs to be subset
            peaks, peak_characteristics = self.subset_two_highest_peaks(peak_output)
        
        right_indents = self.find_curve_indent_right_side(curve, peak_output, x)
        
        if right_indents:
            indent_idx = right_indents[0][0]
            return abs(np.sinh(x[indent_idx]) * cofactor)
        
        if self.two_peaks(peaks): ## two peaks have been found
            if np.argmax(peak_characteristics["peak_heights"]) == 0:
                return abs(np.sinh(x[peak_characteristics["left_bases"][1]]) * cofactor)
            
            assert np.argmax(peak_characteristics["peak_heights"]) == 1
            return abs(np.sinh(x[peak_characteristics["right_bases"][0]]) * cofactor)
        
        if self.only_one_peak(peaks): ## one peak has been found
            return self.find_root_of_tangent_line_at_turning_point(x, curve)


    def subset_two_highest_peaks(self,
                                 peak_output: tuple[np.ndarray, dict]) -> tuple[np.ndarray, np.ndarray]:
        peaks: np.ndarray = peak_output[0] ## array with the locs of found peaks
        peak_characteristics: dict = peak_output[1]
        
        highest_peak_indices = self.find_index_of_two_highest_peaks(peak_characteristics)
        
        peaks: tuple = peaks[highest_peak_indices], peak_characteristics
        for key, value in peak_characteristics.items():
            peak_characteristics[key] = value[highest_peak_indices]
        
        return peaks[0], peaks[1]

    def find_index_of_two_highest_peaks(self,
                                        peak_characteristics: dict) -> np.ndarray:
        return np.sort(np.argpartition(peak_characteristics["peak_heights"], -2)[-2:])
    
    def find_curve_indent_right_side(self,
                                     curve: np.ndarray,
                                     peaks: tuple[np.ndarray, dict],
                                     x: np.ndarray) -> Optional[np.ndarray]:
        try:
            right_peak_index = peaks[0][1]
        except IndexError:
            right_peak_index = peaks[0][0]

        curve = curve / np.max(curve)
        first_derivative = np.gradient(curve)
        second_derivative = np.gradient(first_derivative)

        second_derivative = second_derivative / np.max(second_derivative)

        indents = scs.find_peaks(second_derivative, prominence = 1, height = 1)

        right_indents = indents[0][indents[0] > right_peak_index], indents[1]

        for key, value in right_indents[1].items():
            right_indents[1][key] = value[indents[0] > right_peak_index]

        if right_indents[0].any() and curve[right_indents[0]] > 0.2 and x[right_indents[0]] < 4:
            return right_indents

        return None

    def estimate_cofactor_on_unstained_sample(self,
                                              data_array: np.ndarray,
                                              cofactor: int) -> float:
        data_array = transform_data_array(data_array, cofactor)
        x, curve = get_histogram_curve(data_array)
        
        root = self.find_root_of_tangent_line_at_turning_point(x, curve)

        return abs(np.sinh(root) * cofactor)

    def find_root_of_tangent_line_at_turning_point(self,
                                                   x: np.ndarray,
                                                   curve: np.ndarray) -> float:
        first_derivative = np.gradient(curve)
        turning_point_index = np.argmin(first_derivative),
        ## y = mx+n
        m = np.diff(curve)[turning_point_index] * 1/((np.max(x) - np.min(x)) * 0.01)
        n = curve[turning_point_index] - m * x[turning_point_index]
        return -n/m                 

    def estimate_cofactor_from_control_quantile(self,
                                                dataset: AnnData) -> float:
        return np.quantile(dataset[dataset.obs["staining"] != "stained"].layers["compensated"], 0.95)

    def transform_dataset(self,
                          dataset: AnnData,
                          cofactor_table: CofactorTable) -> AnnData:
        dataset.var = self.merge_cofactors_into_dataset_var(dataset, cofactor_table)
        dataset.var = self.replace_missing_cofactors(dataset.var)
        dataset.layers["transformed"] = transform_data_array(compensated_data = dataset.layers["compensated"],
                                                             cofactors = dataset.var["cofactors"].values)
        return dataset
    
    def get_dataset(self):
        return self.dataset

    def replace_missing_cofactors(self,
                                  dataframe: pd.DataFrame) -> pd.DataFrame:
        """ 
        Missing cofactors can indicate Scatter-Channels and Time Channels
        or not-measured channels. In any case, cofactor is set to 1 for now.
        """
        return dataframe.fillna(1)

    def merge_cofactors_into_dataset_var(self,
                                         dataset: AnnData,
                                         cofactor_table: CofactorTable):
        
        dataset_var = dataset.var.merge(cofactor_table.dataframe,
                                        left_index = True,
                                        right_on = "fcs_colname",
                                        how = "left").set_index("fcs_colname")
        dataset_var["cofactors"] = dataset_var["cofactors"].astype(np.float32)
        return dataset_var


class DatasetAssembler:

    """
    Class to assemble the initial dataset
    containing the compensated data.
    """

    def __init__(self,
                 input_directory: str,
                 metadata: Metadata,
                 panel: Panel,
                 workspace: Union[FlowJoWorkspace, DivaWorkspace]) -> AnnData:

        file_list: list[FCSFile] = self.fetch_fcs_files(input_directory,
                                                        metadata)
        
        file_list: list[FCSFile] = self.compensate_samples(file_list,
                                                           workspace)
        
        gates = self.gate_samples(file_list,
                                  workspace)

        gates = self.fill_empty_gates(file_list, gates)

        dataset_list = self.construct_dataset(file_list,
                                              metadata,
                                              panel)
        
        dataset = self.concatenate_dataset(dataset_list)

        dataset = self.append_supplements(dataset,
                                          metadata,
                                          panel,
                                          workspace)
        
        self.dataset = self.append_gates(dataset,
                                         gates)

        self.dataset.obs = self.dataset.obs.astype("category")

    def fill_empty_gates(self,
                         file_list: list[FCSFile],
                         gates: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """function that looks for ungated samples and appends DataFrames with pd.NA"""
        for i, (file, gate_table) in enumerate(zip(file_list, gates)):
            if gate_table.shape[0] == 0:
                gates[i] = pd.DataFrame(index = range(file.event_count))

        return gates


    def append_gates(self,
                     dataset: AnnData,
                     gates: list[pd.DataFrame]) -> AnnData:
        gatings = pd.concat(gates, axis = 0).fillna(0).astype("bool")
        dataset.obsm["gating"] = csr_matrix(gatings.values)
        dataset.uns["gating_cols"] = gatings.columns
        return dataset

    def append_supplements(self,
                           dataset: AnnData,
                           metadata: Metadata,
                           panel: Panel,
                           workspace: Union[FlowJoWorkspace, DivaWorkspace]) -> AnnData:
        dataset.uns["metadata"] = metadata
        dataset.uns["panel"] = panel
        dataset.uns["workspace"] = workspace.wsp_dict
        return dataset

    def get_dataset(self) -> AnnData:
        return self.dataset

    def create_gating_strategy(self,
                               file: FCSFile,
                               workspace: Union[FlowJoWorkspace, DivaWorkspace]) -> GatingStrategy:
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

    def gate_sample(self,
                    file: FCSFile,
                    workspace: Union[FlowJoWorkspace, DivaWorkspace]) -> pd.DataFrame:
        gating_strategy: GatingStrategy = self.create_gating_strategy(file, workspace)
        gating_results = gating_strategy.gate_sample(file)
        gate_table = pd.DataFrame(columns = ["/".join([path, gate]) for gate, path in gating_results._raw_results.keys()])
        for gate, path in gating_results._raw_results.keys():
            gate_table["/".join([path, gate])] = gating_results.get_gate_membership(gate_name = gate, gate_path = path)
        gate_table.columns = gate_table.columns.str.replace(" ", "_")

        return gate_table

    def gate_samples(self,
                     file_list: list[FCSFile],
                     workspace: Union[FlowJoWorkspace, DivaWorkspace]) -> list[pd.DataFrame]:
        
        return [self.gate_sample(file, workspace) for file in file_list]

    def concatenate_dataset(self,
                            file_list: list[AnnData]):
        return ad.concat(file_list,
                         merge = "same",
                         index_unique = "-",
                         keys = range(len(file_list))
                         )
    
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

        scatter_channels = ["FSC", "SSC", "fsc", "ssc"]
        time_channel = ["time", "Time"]
        fcs_panel_df["type"] = ["scatter" if any(k in channel for k in scatter_channels)
                                else "time" if any(k in channel for k in time_channel) 
                                else "fluo"
                                for channel in fcs_panel_df.index]
        fcs_panel_df["pnn"] = fcs_panel_df.index.to_list()
        fcs_panel_df.index = fcs_panel_df["pns"].to_list()
        return fcs_panel_df

    def create_anndata_representation(self,
                                      file: FCSFile,
                                      metadata: Metadata,
                                      panel: Panel) -> AnnData:
        obs = self.create_obs_from_metadata(file,
                                            metadata)
        var = self.create_var_from_panel(file,
                                         panel)
        return AnnData(X = None,
                          obs = obs,
                          var = var,
                          layers = {"raw": file.original_events.astype(np.float32),
                                    "compensated": file.compensated_events.astype(np.float32)})

    def create_anndata_representations(self,
                                       file_list: list[FCSFile],
                                       metadata: Metadata,
                                       panel: Panel) -> list[AnnData]:
        return [self.create_anndata_representation(file, metadata, panel) for file in file_list]
        

    def construct_dataset(self,
                          file_list: list[FCSFile],
                          metadata: Metadata,
                          panel: Panel) -> AnnData:
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
        metadata = self.append_empty_metadata(metadata,
                                              available_fcs_files)
        return self.convert_fcs_to_FCSFile(input_directory, available_fcs_files)
    
    def append_empty_metadata(self,
                              metadata: Metadata,
                              fcs_files: list[str]) -> Metadata:
        metadata.dataframe["file_name"] = fcs_files
        metadata.dataframe["sample_ID"] = range(1, len(fcs_files) + 1)
        return metadata