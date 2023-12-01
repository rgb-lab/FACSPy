from anndata import AnnData
import numpy as np
import pandas as pd
import scipy.signal as scs
from flowutils import transforms

from typing import Literal, Optional

from ._utils import (find_corresponding_control_samples,
                     get_histogram_curve,
                     transform_data_array,
                     create_sample_subset_with_controls,
                     _merge_cofactors_into_dataset_var,
                     _replace_missing_cofactors,
                     asinh)
from ._supplements import CofactorTable

from .._utils import _fetch_fluo_channels
from ..exceptions._exceptions import InvalidTransformationError
from ..exceptions._supplements import SupplementFormatError


IMPLEMENTED_TRANSFORMS = ["asinh", "logicle", "hyperlog", "log"]

def _transform_array(data: np.ndarray,
                     transform: Literal["asinh", "logicle", "hyperlog", "log"],
                     transform_kwargs: dict,
                     cofactors: Optional[np.ndarray]) -> np.ndarray:
    channel_indices = np.array(list(range(data.shape[1]))) # for now, transform every channel regardless
    m = transform_kwargs.get("m", 4.5)
    t = transform_kwargs.get("t", 262144)
    w = transform_kwargs.get("w", 0.5)
    a = transform_kwargs.get("a", 0)
    
    if transform == "log":
        ## TODO: Needs revision, quick fix for values below 0
        data[data <= 1] = 1
        return transforms.log(data,
                              channel_indices = channel_indices,
                              m = m,
                              t = t)

    elif transform == "hyperlog":
        return transforms.hyperlog(data,
                                   channel_indices = channel_indices,
                                   m = m,
                                   t = t,
                                   w = w,
                                   a = a)

    elif transform == "logicle":
        return transforms.logicle(data,
                                  channel_indices = channel_indices,
                                  m = m,
                                  t = t,
                                  w = w,
                                  a = a)

    else:
        assert transform == "asinh"
        return asinh(data,
                     cofactors)

def transform(adata: AnnData,
              transform: Literal["asinh", "logicle", "hyperlog", "log"],
              transform_kwargs: Optional[dict] = None,
              cofactor_table: Optional[pd.DataFrame] = None,
              key_added: str = None,
              layer: str = "compensated",
              copy: bool = False):
    
    if transform not in IMPLEMENTED_TRANSFORMS:
        raise InvalidTransformationError(transform, IMPLEMENTED_TRANSFORMS)
    
    if not isinstance(cofactor_table, CofactorTable) and cofactor_table is not None:
        raise SupplementFormatError(supplement = "CofactorTable",
                                    instance_type = type(cofactor_table))
    
    if transform_kwargs is None:
        transform_kwargs = {}
    
    
    adata = adata.copy() if copy else adata

    if key_added is None:
        key_added = transform

    if transform == "asinh" and not cofactor_table:
        cofactor_table, raw_cofactor_table = CofactorCalculator(adata).get_cofactors()
        adata.uns["raw_cofactors"] = raw_cofactor_table

    if cofactor_table:
        adata.uns["cofactor_table"] = cofactor_table
        adata.var = _merge_cofactors_into_dataset_var(adata, cofactor_table)
        adata.var = _replace_missing_cofactors(adata.var)
        cofactors = adata.var["cofactors"].values
    else:
        cofactors = None

    adata.layers[key_added] = _transform_array(data = adata.layers[layer],
                                               transform = transform,
                                               transform_kwargs = transform_kwargs,
                                               cofactors = cofactors)
    
    return adata if copy else None

def calculate_cofactors(adata,
                        add_to_adata: bool = True,
                        return_dataframe: bool = False,
                        copy: bool = False) -> Optional[AnnData]:
    adata = adata.copy() if copy else adata
    cofactor_calc = CofactorCalculator(adata = adata,
                                       add_to_adata = add_to_adata)
    if return_dataframe:
        return cofactor_calc.get_cofactors()
    
    return adata if copy else None
    

class CofactorCalculator:

    def __init__(self,
                 adata: AnnData,
                 add_to_adata: bool = True,
                 use_gate: Optional[str] = None) -> None:
        ### Notes: Takes approx. (80-100.000 cells * 17 channels) / second 
        print("... calculating cofactors")
        self.cofactor_table, self.raw_cofactor_table = self.calculate_cofactors(adata)
        if add_to_adata:
            adata.uns["cofactors"] = self.cofactor_table
            adata.uns["raw_cofactors"] = self.raw_cofactor_table
    
    def get_cofactors(self):
        return self.cofactor_table, self.raw_cofactor_table
    
    def calculate_cofactors(self,
                            adata: AnnData) -> tuple[CofactorTable, pd.DataFrame]:
        
        (stained_samples,
         corresponding_control_samples) = find_corresponding_control_samples(adata,
                                                                             by = "file_name")
        cofactors = {}
        for sample in stained_samples:
            print(f"    ... sample {sample}")
            cofactors[sample] = {}
            fluo_channels = _fetch_fluo_channels(adata)
            sample_subset = create_sample_subset_with_controls(adata,
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
        # sourcery skip: use-named-expression
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
                                                adata: AnnData) -> float:
        return np.quantile(adata[adata.obs["staining"] != "stained"].layers["compensated"], 0.95)

    


