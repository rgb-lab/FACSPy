from anndata import AnnData
import numpy as np
import pandas as pd
import scipy.signal as scs

from typing import Literal, Optional, Union

from ._utils import (find_corresponding_control_samples,
                     _get_histogram_curve,
                     transform_data_array,
                     create_sample_subset_with_controls,
                     _merge_cofactors_into_dataset_var,
                     _replace_missing_cofactors,
                     asinh_transform,
                     log_transform,
                     hyperlog_transform,
                     logicle_transform)
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
        return log_transform(data = data,
                             m = m,
                             t = t,
                             channel_indices = channel_indices)

    elif transform == "hyperlog":
        return hyperlog_transform(data = data,
                                  channel_indices = channel_indices,
                                  m = m,
                                  w = w,
                                  a = a,
                                  t = t)

    elif transform == "logicle":
        return logicle_transform(data = data,
                                 channel_indices = channel_indices,
                                 m = m,
                                 w = w,
                                 a = a,
                                 t = t)

    else:
        assert transform == "asinh"
        assert isinstance(cofactors, np.ndarray)
        return asinh_transform(data,
                               cofactors)

def transform(adata: AnnData,
              transform: Literal["asinh", "logicle", "hyperlog", "log"],
              transform_kwargs: Optional[dict] = None,
              cofactor_table: Optional[CofactorTable] = None,
              key_added: Optional[str] = None,
              layer: str = "compensated",
              copy: bool = False) -> Optional[AnnData]:
    """\
        
    Transforms the data. The data are transformed by either `asinh`, `log`,
    `logicle` or `hyperlog`. If `asinh` is selected, a cofactor table can
    be supplied mapping the channel names to their respective cofactor. If 
    there is no cofactor table, the cofactors will be calculated.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where Rows correspond to cells and columns to the channels
    transform
        The transform method to be used. Can be any of `asinh`, `log`, 
        `logicle` or `hyperlog`. If `log`, values below 1 are set to 1.
    transform_kwargs
        Keyword arguments passed to the respective transform function as a dictionary.
        Please refer to their documentation
    cofactor_table
        A table mapping the channels (e.g. CD3) to their respective cofactors.
        Please refer to the documentation of the CofactorTable class for
        further information. Note that missing cofactors will be set to 1.
        This is mostly relevant for the scatter/time channels where there is
        no reasonable cofactor to be set.
    key_added
        The name of the layer that is created. Defaults to `transformed`.
    layer
        The name of the layer that stores the data to be transformed.
        Defaults to `compensated`.
    copy
        Whether to copy the dataset and return the copy.


    Returns
    -------
    :class:`~anndata.AnnData` or None
        Returns adata if `copy = True`, otherwise adds fields to the anndata
        object:

        `.uns[layer]`
            The transformed data
        `.var['cofactors']`
            added cofactors from the cofactor table if provided.

    Examples
    --------

    >>> import FACSPy as fp
    >>> cof_table = CofactorTable("cofactors.csv")
    >>> dataset = fp.create_dataset(...)
    >>> dataset
    AnnData object with n_obs × n_vars = 615936 × 22
    obs: 'sample_ID', 'file_name', 'condition'
    var: 'pns', 'png', 'pne', 'pnr', 'type', 'pnn'
    uns: 'metadata', 'panel', 'workspace', 'gating_cols', 'dataset_status_hash'
    obsm: 'gating'
    layers: 'compensated'
    >>> fp.transform(
    ...     dataset,
    ...     transform = "asinh",
    ...     cofactor_table = cof_table,
    ...     key_added = "transformed"
    ... )
    >>> dataset
    AnnData object with n_obs × n_vars = 615936 × 22
    obs: 'sample_ID', 'file_name', 'condition'
    var: 'pns', 'png', 'pne', 'pnr', 'type', 'pnn'
    uns: 'metadata', 'panel', 'workspace', 'gating_cols', 'dataset_status_hash'
    obsm: 'gating'
    layers: 'compensated', 'transformed'

    """

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
        adata.uns["cofactors"] = cofactor_table
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

def calculate_cofactors(adata: AnnData,
                        add_to_adata: bool = True,
                        return_dataframe: bool = False,
                        copy: bool = False
                        ) -> Optional[Union[AnnData, tuple[CofactorTable, pd.DataFrame]]]:
    """\
    
    Calculates the cofactors based on the channel histograms.

    Parameters
    ----------

    adata
        The anndata object of shape `n_obs` x `n_vars`
        where Rows correspond to cells and columns to the channels

    add_to_adata
        Whether to add the calculated cofactors to the adata object.
        Defaults to True.
    return_dataframe
        Whether to return the calculated cofactors as a pd.DataFrame.
        Defaults to False
    copy
        Whether to copy the dataset and return the copy.

    Returns
    -------
    If copy is True:
        The AnnData object. If `add_to_adata` is True,
        the .uns['cofactors'] and .uns['raw_cofactors']
        slots are filled with the corresponding tables.
    If return_dataframe:
        a tuple of the cofactor table and the raw cofactors per sample
    
    Examples
    --------
    >>> import FACSPy as fp
    >>> dataset = fp.create_dataset(...)
    >>> dataset
    AnnData object with n_obs × n_vars = 615936 × 22
    obs: 'sample_ID', 'file_name', 'condition'
    var: 'pns', 'png', 'pne', 'pnr', 'type', 'pnn'
    uns: 'metadata', 'panel', 'workspace', 'gating_cols', 'dataset_status_hash'
    obsm: 'gating'
    layers: 'compensated'
    >>> fp.calculate_cofactors(dataset)
    >>> dataset
    AnnData object with n_obs × n_vars = 615936 × 22
    obs: 'sample_ID', 'file_name', 'condition'
    var: 'pns', 'png', 'pne', 'pnr', 'type', 'pnn'
    uns: 'metadata', 'panel', 'workspace', 'gating_cols', 'dataset_status_hash',
    'raw_cofactors', 'cofactors'
    obsm: 'gating'
    layers: 'compensated'
    >>> fp.transform(
    ...     dataset,
    ...     transform = "asinh",
    ...     cofactor_table = dataset.uns["cofactors"],
    ...     key_added = "transformed"
    ... )
    >>> dataset
    AnnData object with n_obs × n_vars = 615936 × 22
    obs: 'sample_ID', 'file_name', 'condition'
    var: 'pns', 'png', 'pne', 'pnr', 'type', 'pnn'
    uns: 'metadata', 'panel', 'workspace', 'gating_cols', 'dataset_status_hash',
    'raw_cofactors', 'cofactors'
    obsm: 'gating'
    layers: 'compensated', 'transformed'
    
    """
    adata = adata.copy() if copy else adata
    cofactor_calc = CofactorCalculator(adata = adata)
    cofactor_table, raw_cofactor_table = cofactor_calc.get_cofactors()

    if add_to_adata:
        adata.uns["cofactors"] = cofactor_table
        adata.uns["raw_cofactors"] = raw_cofactor_table

    if return_dataframe:
        return cofactor_calc.get_cofactors()
    
    return adata if copy else None
    

class CofactorCalculator:

    """\
    Runs Cofactor calculation
    Class is not meant to be used by the user and is only called
    internally by fp.calculate_cofactors.
    """

    def __init__(self,
                 adata: AnnData) -> None:

        ### Notes: Takes approx. (80-100.000 cells * 17 channels) / second 
        print("... calculating cofactors")
        self.cofactor_table, self.raw_cofactor_table = self._calculate_cofactors(adata)
    
    def get_cofactors(self) -> tuple[CofactorTable, pd.DataFrame]:
        """returns the calculated cofactors as a CofactorTable object and the raw cofactors per sample as a pd.DataFrame"""
        return self.cofactor_table, self.raw_cofactor_table
    
    def _calculate_cofactors(self,
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
                cofactor_stained_sample = self._estimate_cofactor_on_stained_sample(data_array,
                                                                                    200)
                if corresponding_control_samples[sample]:
                    control_sample = sample_subset[sample_subset.obs["staining"] != "stained", sample_subset.var.index == channel]
                    data_array = control_sample.layers["compensated"]
                    cofactor_unstained_sample = self._estimate_cofactor_on_unstained_sample(data_array, 20)
                    cofactor_by_percentile = self._estimate_cofactor_from_control_quantile(control_sample)
                
                    cofactors[sample][channel] = np.mean([cofactor_stained_sample,
                                                          cofactor_unstained_sample,
                                                          cofactor_by_percentile])
                    
                    continue
                cofactors[sample][channel] = cofactor_stained_sample
        return self._create_cofactor_tables(cofactors)

    def _create_cofactor_tables(self,
                                cofactors: dict[str, list[float]],
                                reduction_method: str = "mean") -> tuple[CofactorTable, pd.DataFrame]:
        """creates the CofactorTable object from the means or medians of the raw cofactors per sample"""
        raw_table = pd.DataFrame(data = cofactors).T
        if reduction_method == "mean":
            reduced = pd.DataFrame(cofactors).mean(axis = 1)
        elif reduction_method == "median":
            reduced = pd.DataFrame(cofactors).median(axis = 1)
        reduced_table = pd.DataFrame({"fcs_colname": reduced.index,
                                      "cofactors": reduced.values})
        return CofactorTable(cofactors = reduced_table), raw_table

    def _only_one_peak(self,
                       peaks: np.ndarray) -> bool:
        """returns True if only one peak has been found"""
        return peaks.shape[0] == 1

    def _two_peaks(self,
                   peaks: np.ndarray) -> bool:
        """returns True if two peaks have been found"""
        return peaks.shape[0] == 2

    def _estimate_cofactor_on_stained_sample(self,
                                             data_array: np.ndarray,
                                             cofactor: int) -> float:
        """wrapper method to estimate cofactors on a stained sample"""
        data_array = transform_data_array(data_array, cofactor)
        x, curve = _get_histogram_curve(data_array)
        
        peak_output = scs.find_peaks(curve, prominence = 0.001, height = 0.01)
        peaks: np.ndarray = peak_output[0] ## array with the locs of found peaks
        peak_characteristics: dict = peak_output[1]

        if peaks.shape[0] >= 2: ## more than two peaks have been found it needs to be subset
            peaks, peak_characteristics = self._subset_two_highest_peaks(peak_output)
        # sourcery skip: use-named-expression
        right_indents = self._find_curve_indent_right_side(curve, peak_output, x)
        
        if right_indents:
            indent_idx = right_indents[0][0]
            return abs(np.sinh(x[indent_idx]) * cofactor)
        
        if self._two_peaks(peaks): ## two peaks have been found
            if np.argmax(peak_characteristics["peak_heights"]) == 0:
                return abs(np.sinh(x[peak_characteristics["left_bases"][1]]) * cofactor)
            
            assert np.argmax(peak_characteristics["peak_heights"]) == 1
            return abs(np.sinh(x[peak_characteristics["right_bases"][0]]) * cofactor)
        
        if self._only_one_peak(peaks): ## one peak has been found
            return self._find_root_of_tangent_line_at_turning_point(x, curve)

    def _subset_two_highest_peaks(self,
                                  peak_output: tuple[np.ndarray, dict]
                                  ) -> tuple[np.ndarray, np.ndarray]:
        """finds the two highest peaks and returns their locations"""
        peaks: np.ndarray = peak_output[0] ## array with the locs of found peaks
        peak_characteristics: dict = peak_output[1]
        
        highest_peak_indices = self._find_index_of_two_highest_peaks(peak_characteristics)
        
        peaks: tuple = peaks[highest_peak_indices], peak_characteristics
        for key, value in peak_characteristics.items():
            peak_characteristics[key] = value[highest_peak_indices]
        
        return peaks[0], peaks[1]

    def _find_index_of_two_highest_peaks(self,
                                         peak_characteristics: dict) -> np.ndarray:
        """returns index of the two highest peaks"""
        return np.sort(np.argpartition(peak_characteristics["peak_heights"], -2)[-2:])
    
    def _find_curve_indent_right_side(self,
                                      curve: np.ndarray,
                                      peaks: tuple[np.ndarray, dict],
                                      x: np.ndarray) -> Optional[np.ndarray]:
        """returns the inflection point of the curve on the right side of the peak"""
        try:
            right_peak_index = peaks[0][1]
        except IndexError:
            right_peak_index = peaks[0][0]

        curve = curve / np.max(curve)
        first_derivative = np.gradient(curve)
        second_derivative = np.gradient(first_derivative)

        second_derivative = second_derivative / np.max(second_derivative)

        indents = scs.find_peaks(second_derivative, prominence = 1, height = 1)

        right_indents: tuple[np.ndarray, dict] = indents[0][indents[0] > right_peak_index], indents[1]

        for key, value in right_indents[1].items():
            right_indents[1][key] = value[indents[0] > right_peak_index]

        if right_indents[0].any() and curve[right_indents[0]] > 0.2 and x[right_indents[0]] < 4:
            return right_indents

        return None

    def _estimate_cofactor_on_unstained_sample(self,
                                              data_array: np.ndarray,
                                              cofactor: int) -> float:
        """wrapper method to estimate the cofactor on an unstained sample"""
        data_array = transform_data_array(data_array, cofactor)
        x, curve = _get_histogram_curve(data_array)
        
        root = self._find_root_of_tangent_line_at_turning_point(x, curve)

        return abs(np.sinh(root) * cofactor)

    def _find_root_of_tangent_line_at_turning_point(self,
                                                    x: np.ndarray,
                                                    curve: np.ndarray) -> float:
        """calculates the root of the tangent line at the turning point of the histogram function"""
        first_derivative = np.gradient(curve)
        turning_point_index = np.argmin(first_derivative),
        ## y = mx+n
        m = np.diff(curve)[turning_point_index] * 1/((np.max(x) - np.min(x)) * 0.01)
        n = curve[turning_point_index] - m * x[turning_point_index]
        return -n/m                 

    def _estimate_cofactor_from_control_quantile(self,
                                                 adata: AnnData) -> float:
        """calculates the 95th quantile of the unstained data"""
        return np.quantile(adata[adata.obs["staining"] != "stained"].layers["compensated"], 0.95)
