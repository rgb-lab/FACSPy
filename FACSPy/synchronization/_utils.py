import pandas as pd
from anndata import AnnData
from typing import Optional, Literal
from .._utils import IMPLEMENTED_SAMPLEWISE_DIMREDS
from ._hash_generation import HASH_FUNCTION_DICT

def _reset_hash(adata: AnnData,
                metric: Literal["adata_obs_names",
                                "adata_sample_ids",
                                "adata_var_names",
                                "panel_var_names",
                                "metadata_sample_ids",
                                "metadata_columns",
                                "adata_obs_columns"]) -> None:
    """function that creates a current hash for the indicated metric"""
    adata.uns["dataset_status_hash"][metric] = HASH_FUNCTION_DICT[metric](adata)
    return

def _sync_uns_frames(adata: AnnData,
                     recalculate: bool = False) -> None:
    """
    Samples are synchronized so that only samples are kept that
    are the current dataset sample_IDs.

    First, the metadata are subset for the unique sampleIDs in
    adata.obs["sample_ID"]

    Args:
        adata (AnnData): the anndata object
        recalculate (bool): whether or not to recalculate the mfi/fop/samplewise dr frames
        
    """

    current_obs_sample_IDs = adata.obs["sample_ID"].unique().tolist()
    current_var_names = adata.var_names.tolist()

    mfi_frames = [key for key in adata.uns if "mfi" in key]
    fop_frames = [key for key in adata.uns if "fop" in key]

    for frame_id in mfi_frames + fop_frames:
        calculated_dimreds = _get_present_samplewise_dimreds(adata.uns[frame_id])
        if recalculate:
            if "mfi" in frame_id:
                _recalculate_mfi(adata = adata,
                                 frame_id = frame_id)
            if "fop" in frame_id:
                _recalculate_fop(adata = adata,
                                 frame_id = frame_id)
            _recalculate_samplewise_dimreds(adata, frame_id, calculated_dimreds)

        _synchronize_uns_frame(adata = adata,
                               identifier = frame_id,
                               sample_IDs = current_obs_sample_IDs,
                               var_names = current_var_names,
                               calculated_dimreds = calculated_dimreds)
        print(f"     ... synchronized frame {frame_id}")

def _recalculate_mfi(adata: AnnData,
                     frame_id: str) -> None:
    from ..tools._mfi import mfi
    _, data_origin, data_group = _get_frame_metrics(frame_id)
    settings_dict = adata.uns["settings"][f"_mfi_{data_group}_{data_origin}"]
    mfi(adata,
        **settings_dict)

def _recalculate_fop(adata: AnnData,
                     frame_id: str) -> None:
    from ..tools._fop import fop
    _, data_origin, data_group = _get_frame_metrics(frame_id)
    settings_dict = adata.uns["settings"][f"_fop_{data_group}_{data_origin}"]
    fop(adata,
        **settings_dict)

def _append_calculated_dimred_dimensions(calculated_dimreds: list[str],
                                         uns_frame: pd.DataFrame):
    dimred_dimensions = []
    for dimred in calculated_dimreds:
        for col in uns_frame.columns:
            if dimred in col:
                dimred_dimensions.append(col)
    
    return dimred_dimensions

def _synchronize_uns_frame(adata: AnnData,
                           identifier: str,
                           sample_IDs: list[str],
                           var_names: list[str],
                           calculated_dimreds: list[str]) -> None:
    """
    synchronizes uns frame using the sample_IDs and var_names
    and calculated samplewise dimreds
    """
    print(f"... synchronizing dataframe: {identifier}")
    uns_frame: pd.DataFrame = adata.uns[identifier]
    dimred_dimensions = _append_calculated_dimred_dimensions(calculated_dimreds = calculated_dimreds,
                                                             uns_frame = uns_frame)
    if "sample_ID" in uns_frame.index.names:
        adata.uns[identifier] = uns_frame.loc[
            uns_frame.index.get_level_values("sample_ID").isin(sample_IDs),
            var_names + dimred_dimensions
        ]
    else:
        adata.uns[identifier] = uns_frame.loc[
            :,
            var_names + dimred_dimensions
        ]
    return

def _get_frame_metrics(frame_id: str) -> tuple[str]:
    split_frame_id = frame_id.split("_")
    data_metric = split_frame_id[0]
    data_origin = split_frame_id[-1]
    data_group = "_".join([entry for entry in split_frame_id[1:-1]])
    return (data_metric, data_origin, data_group)

def _find_corresponding_settings(adata: AnnData,
                                 assay: str,
                                 data_metric: str,
                                 data_origin: str) -> dict:
    return adata.uns["settings"][f"{assay}_{data_metric}_{data_origin}"]

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

def _recalculate_samplewise_dimreds(adata,
                                    frame_id: str,
                                    calculated_dimreds: Optional[list[str]]) -> None:
    from ..tools._pca import pca_samplewise
    from ..tools._mds import mds_samplewise
    from ..tools._umap import umap_samplewise
    from ..tools._tsne import tsne_samplewise 

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
        return
    if "MDS" in calculated_dimreds:
        print(f"     ... Recalculating samplewise mds for {frame_id}")
        settings = _find_corresponding_settings(assay = "_mds_samplewise",
                                                **settings_finder_dict)
        mds_samplewise(adata,
                       **settings)
        return
    if "UMAP" in calculated_dimreds:
        print(f"     ... Recalculating samplewise umap for {frame_id}")
        settings = _find_corresponding_settings(assay = "_umap_samplewise",
                                                **settings_finder_dict)
        umap_samplewise(adata,
                        **settings)
        return
    if "TSNE" in calculated_dimreds:
        print(f"     ... Recalculating samplewise tsne for {frame_id}")
        settings = _find_corresponding_settings(assay = "_tsne_samplewise",
                                                **settings_finder_dict)
        tsne_samplewise(adata,
                        **settings)
        return

