import pandas as pd
from anndata import AnnData
import copy

from typing import Optional

from ..dataset._supplements import Panel, CofactorTable
from .._utils import IMPLEMENTED_SAMPLEWISE_DIMREDS

def synchronize_vars(adata: AnnData,
                     recalculate: bool = False) -> None:
    """
    Vars are synchronized so that only channels are kept that
    are the current dataset var names.

    For varm this is done by the anndata slicing, we need to keep
    track for the .uns dataframes that stores all the mfi/fop and so on

    Importantly! The panel and cofactor metadata are unchanged for now!

    """

    current_var_names = adata.var_names.to_list()

    mfi_frames = [key for key in adata.uns if "mfi" in key]
    fop_frames = [key for key in adata.uns if "fop" in key]

    for frame_id in mfi_frames + fop_frames:
        if recalculate:
            ## to recalculate, we need to re-perform the dimensionality reductions.
            ## the MFI/fop values do not change if a channel is kicked out
            _recalculate_samplewise_dimreds(adata, frame_id)
        
        _synchronize_uns_frame(adata,
                               identifier = frame_id,
                               var_names = current_var_names)
    
    panel: Panel = adata.uns["panel"]
    panel.select_channels(current_var_names)
    print("     ... updated panel")

    try:
        cofactor_table: CofactorTable = adata.uns["cofactors"]
        cofactor_table.select_channels(current_var_names)
        print("     ... updated cofactors")
    except KeyError:
        """means no cofactors were supplied"""
        pass

def _recalculate_samplewise_dimreds(adata,
                                    frame_id: str) -> Optional[AnnData]:
    from ..tools._dr_samplewise import (pca_samplewise,
                                        mds_samplewise,
                                        tsne_samplewise,
                                        umap_samplewise)
    calculated_dimreds = _get_present_samplewise_dimreds(adata.uns[frame_id])
    data_metric, data_origin, _ = _get_frame_metrics(frame_id)
    settings_finder_dict = {
        "adata": adata,
        "data_metric": data_metric,
        "data_origin": data_origin
    }
    if "PCA" in calculated_dimreds:
        print(f"     ... Recalculating samplewise pca for {frame_id}")
        settings = _find_corresponding_settings(assay = "_pca_samplewise",
                                                **settings_finder_dict)
        pca_samplewise(adata,
                       **settings)
    if "MDS" in calculated_dimreds:
        print(f"     ... Recalculating samplewise mds for {frame_id}")
        settings = _find_corresponding_settings(assay = "_mds_samplewise",
                                                **settings_finder_dict)
        mds_samplewise(adata,
                       **settings)
    if "UMAP" in calculated_dimreds:
        print(f"     ... Recalculating samplewise umap for {frame_id}")
        settings = _find_corresponding_settings(assay = "_umap_samplewise",
                                                **settings_finder_dict)
        umap_samplewise(adata,
                        **settings)
    if "TSNE" in calculated_dimreds:
        print(f"     ... Recalculating samplewise tsne for {frame_id}")
        settings = _find_corresponding_settings(assay = "_tsne_samplewise",
                                                **settings_finder_dict)
        tsne_samplewise(adata,
                        **settings)
 
def _find_corresponding_settings(adata: AnnData,
                                 assay: str,
                                 data_metric: str,
                                 data_origin: str) -> dict:
    return adata.uns["settings"][f"{assay}_{data_metric}_{data_origin}"]

def _get_frame_metrics(frame_id: str) -> tuple[str]:
    split_frame_id = frame_id.split("_")
    data_metric = split_frame_id[0]
    data_origin = split_frame_id[-1]
    data_group = "_".join([entry for entry in split_frame_id[1:-1]])
    return (data_metric, data_origin, data_group)

def _get_present_samplewise_dimreds(df: pd.DataFrame) -> list[str]:
    dimred_cols = _get_samplewise_dimred_columns(df)
    performed_dimreds = []
    for dimred_col in dimred_cols:
        if "PCA" in dimred_col:
            performed_dimreds.append("PCA")
        elif "UMAP" in dimred_col:
            performed_dimreds.append("UMAP")
        elif "MDS" in dimred_col:
            performed_dimreds.append("MDS")
        elif "TSNE" in dimred_col:
            performed_dimreds.append("TSNE")
    return list(set(performed_dimreds))
        
def _get_samplewise_dimred_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns
            if any(k in col for k in IMPLEMENTED_SAMPLEWISE_DIMREDS)]

def _synchronize_uns_frame(adata: AnnData,
                           identifier: str,
                           var_names = None) -> None:
    # we keep all the remaining var_names
    columns_to_keep = copy.copy(var_names)

    # we append columns that originated from the samplewise dimreds
    columns_to_keep += _get_samplewise_dimred_columns(adata.uns[identifier])

    adata.uns[identifier] = adata.uns[identifier].loc[:, columns_to_keep]
