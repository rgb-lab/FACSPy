import os
from typing import Union, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scipy.signal as scs
from scipy.sparse import csr_matrix

from KDEpy import FFTKDE

from .supplements import Panel, Metadata, CofactorTable
from .workspaces import FlowJoWorkspace, DivaWorkspace
from .sample import FCSFile

from ..transforms._matrix import Matrix
from ..exceptions.exceptions import PanelMatchWarning
from ..gates.gating_strategy import GatingStrategy


class Transformer:

    def __init__(self,
                 dataset: ad.AnnData,
                 cofactor_table: Optional[CofactorTable] = None) -> None:
        ### Notes: Takes approx. (80-100.000 cells * 17 channels) / second 
        if not cofactor_table:
            cofactor_table, raw_cofactor_table = self.calculate_cofactors(dataset)
            dataset.uns["raw_cofactors"] = raw_cofactor_table
        
        dataset.uns["cofactors"] = cofactor_table    
        self.dataset = self.transform_dataset(dataset, cofactor_table)

    def calculate_cofactors(self,
                            dataset: ad.AnnData) -> tuple[CofactorTable, pd.DataFrame]:
        
        (stained_samples,
         corresponding_control_samples) = self.find_corresponding_control_samples(dataset)
        cofactors = {}
        for sample in stained_samples:
            cofactors[sample] = {}
            fluo_channels = self.fetch_fluo_channels(dataset)
            sample_subset = self.create_sample_subset_with_controls(dataset,
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
        x, curve = self.get_histogram_curve(data_array,
                                            cofactor)
        
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
        x, curve = self.get_histogram_curve(data_array,
                                            cofactor)
        
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
    
    def get_histogram_curve(self,
                            data_array: np.ndarray,
                            cofactor: int) -> tuple[np.ndarray, np.ndarray]:
        transformed = self.transform_data(data_array, cofactor)
        _, x = np.histogram(transformed, bins = 100)
        _, curve = FFTKDE(kernel = "gaussian",
                          bw = "silverman"
                          ).fit(transformed).evaluate(100)
        return x, curve

    def estimate_cofactor_from_control_quantile(self,
                                                dataset: ad.AnnData) -> float:
        return np.quantile(dataset[dataset.obs["staining"] != "stained"].layers["compensated"], 0.95)

    def create_sample_subset_with_controls(self,
                                           dataset: ad.AnnData,
                                           sample: str,
                                           corresponding_controls: dict,
                                           match_cell_number: bool) -> ad.AnnData:
        controls: list[str] = corresponding_controls[sample]
        sample_list = [sample] + controls
        if match_cell_number:
            return self.match_cell_numbers(dataset[dataset.obs["file_name"].isin(sample_list)])
        return dataset[dataset.obs["file_name"].isin(sample_list)]

    def match_cell_numbers(self,
                           dataset: ad.AnnData) -> ad.AnnData:
        return dataset

    def fetch_fluo_channels(self,
                            dataset: ad.AnnData) -> list[str]:
        return [
            channel
            for channel in dataset.var.index.to_list()
            if all(k not in channel.lower() for k in ["fsc", "ssc", "time"])
        ]

    def find_corresponding_control_samples(self,
                                           dataset: ad.AnnData) -> tuple[list[str], dict[str, str]]:
        corresponding_controls = {}
        metadata: Metadata = dataset.uns["metadata"]
        metadata_frame = metadata.to_df()
        indexed_metadata = self.reindex_metadata(metadata_frame,
                                                 metadata.factors)
        
        stained_samples = self.get_stained_samples(metadata_frame)
        control_samples = self.get_control_samples(metadata_frame)
        
        for sample in stained_samples:
            sample_metadata = metadata_frame.loc[metadata_frame["file_name"] == sample, metadata.factors]
            matching_control_samples = self.find_name_of_control_sample_by_metadata(sample,
                                                                                    sample_metadata,
                                                                                    indexed_metadata)
            corresponding_controls[sample] = matching_control_samples or control_samples

        return stained_samples, corresponding_controls
    
    def reindex_metadata(self,
                         metadata: pd.DataFrame,
                         indices: list[str]) -> pd.DataFrame:
        return metadata.set_index(indices)


    def find_name_of_control_sample_by_metadata(self,
                                                sample,
                                                metadata_to_compare: pd.DataFrame,
                                                indexed_frame: pd.DataFrame) -> list[str]:
        matching_metadata = indexed_frame.loc[tuple(metadata_to_compare.values[0])]
        return matching_metadata.loc[matching_metadata["file_name"] != sample, "file_name"].to_list()

    def get_control_samples(self,
                            dataframe: pd.DataFrame) -> list[str]:
        return dataframe.loc[dataframe["staining"] != "stained", "file_name"].to_list()

    def get_stained_samples(self,
                            dataframe: pd.DataFrame) -> list[str]:
        return dataframe.loc[dataframe["staining"] == "stained", "file_name"].to_list()

    def transform_dataset(self,
                          dataset: ad.AnnData,
                          cofactor_table: CofactorTable) -> ad.AnnData:
        dataset.var = self.merge_cofactors_into_dataset_var(dataset, cofactor_table)
        dataset.var = self.replace_missing_cofactors(dataset.var)
        dataset.layers["transformed"] = self.transform_data(compensated_data = dataset.layers["compensated"],
                                                                cofactors = dataset.var["cofactors"].values)
        return dataset
    
    def get_dataset(self):
        return self.dataset
   
    def transform_data(self,
                       compensated_data: np.ndarray,
                       cofactors: Union[np.ndarray, int, float]) -> np.ndarray:
        return np.arcsinh(np.divide(compensated_data, cofactors))

    def replace_missing_cofactors(self,
                                  dataframe: pd.DataFrame) -> pd.DataFrame:
        """ 
        Missing cofactors can indicate Scatter-Channels and Time Channels
        or not-measured channels. In any case, cofactor is set to 1 for now.
        """
        return dataframe.fillna(1)

    def merge_cofactors_into_dataset_var(self,
                                         dataset: ad.AnnData,
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
                 workspace: Union[FlowJoWorkspace, DivaWorkspace]) -> ad.AnnData:

        file_list: list[FCSFile] = self.fetch_fcs_files(input_directory,
                                                        metadata)
        file_list: list[FCSFile] = self.compensate_samples(file_list,
                                                           workspace)
        gates = self.gate_samples(file_list,
                                  workspace)

        dataset_list = self.construct_dataset(file_list,
                                              metadata,
                                              panel)
        dataset = self.concatenate_dataset(dataset_list)
        dataset = self.append_supplements(dataset,
                                          metadata,
                                          panel)
        self.dataset = self.append_gates(dataset,
                                         gates)

        self.dataset.obs = self.dataset.obs.astype("category")

    def append_gates(self,
                     dataset: ad.AnnData,
                     gates: list[pd.DataFrame]) -> ad.AnnData:
        gatings = pd.concat(gates, axis = 0).fillna(0).astype("bool")
        dataset.obsm["gating"] = csr_matrix(gatings.values)
        dataset.uns["gating_cols"] = gatings.columns
        return dataset

    def append_supplements(self,
                           dataset: ad.AnnData,
                           metadata: Metadata,
                           panel: Panel) -> ad.AnnData:
        dataset.uns["metadata"] = metadata
        dataset.uns["panel"] = panel
        return dataset

    def get_dataset(self) -> ad.AnnData:
        return self.dataset

    def create_gating_strategy(self,
                               file: FCSFile,
                               workspace: Union[FlowJoWorkspace, DivaWorkspace]) -> GatingStrategy:
        gating_strategy = GatingStrategy()
        workspace_subset = workspace.wsp_dict["All Samples"][file.original_filename]
        for gate_dict in workspace_subset["gates"]:
            gating_strategy.add_gate(gate_dict["gate"], gate_path = gate_dict["gate_path"])
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

        scatter_channels = ["FSC", "SSC", "fsc", "ssc"]
        time_channel = ["time", "Time"]
        fcs_panel_df["type"] = ["scatter" if any(k in channel for k in scatter_channels)
                                else "time" if any(k in channel for k in time_channel) 
                                else "fluo"
                                for channel in fcs_panel_df.index]
        
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
        