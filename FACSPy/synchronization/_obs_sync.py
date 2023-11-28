import pandas as pd
from ..dataset._supplements import Metadata
from anndata import AnnData

from ._utils import (_get_frame_metrics,
                     _recalculate_samplewise_dimreds,
                     _get_present_samplewise_dimreds)

def synchronize_samples(adata: AnnData,
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

    mfi_frames = [key for key in adata.uns if "mfi" in key]
    fop_frames = [key for key in adata.uns if "fop" in key]

    for frame_id in mfi_frames + fop_frames:
        if recalculate:
            calculated_dimreds = _get_present_samplewise_dimreds(adata.uns[frame_id])
            if "mfi" in frame_id:
                _recalculate_mfi(adata = adata,
                                 frame_id = frame_id)
            if "fop" in frame_id:
                _recalculate_fop(adata = adata,
                                 frame_id = frame_id)
            _recalculate_samplewise_dimreds(adata, frame_id, calculated_dimreds)

        _synchronize_uns_frame(adata = adata,
                               identifier = frame_id,
                               sample_IDs = current_obs_sample_IDs)
        print(f"     ... synchronized frame {frame_id}")

    _synchronize_metadata_object(adata,
                                 current_obs_sample_IDs)
    
    _sanitize_categoricals(adata)
    print("     ... updated metadata")

def _sanitize_categoricals(adata: AnnData):
    obs_frame = adata.obs.copy()
    for column in obs_frame.columns:
        if isinstance(obs_frame[column].dtype, pd.CategoricalDtype):
            obs_frame[column] = obs_frame[column].cat.remove_unused_categories()
    adata.obs = obs_frame
    return
    
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

def _synchronize_uns_frame(adata: AnnData,
                           identifier: str,
                           sample_IDs: list[str]) -> None:
    uns_frame: pd.DataFrame = adata.uns[identifier]
    if "sample_ID" in uns_frame.index.names:
        adata.uns[identifier] = uns_frame.loc[uns_frame.index.get_level_values("sample_ID").isin(sample_IDs),:]
    else:
        ## potentially add warning?
        adata.uns[identifier] = uns_frame
    return

def _synchronize_metadata_object(adata: AnnData,
                                 current_obs_sample_IDs: pd.Series) -> None:
    metadata: Metadata = adata.uns["metadata"]
    metadata.subset("sample_ID", current_obs_sample_IDs)
    metadata._sanitize_categoricals()
    adata.uns["metadata"] = metadata
    return