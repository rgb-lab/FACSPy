from anndata import AnnData
import copy

from ._utils import (_get_samplewise_dimred_columns,
                     _recalculate_samplewise_dimreds,
                     _get_present_samplewise_dimreds)
from ..dataset._supplements import Panel, CofactorTable

def synchronize_vars(adata: AnnData,
                     recalculate: bool = False) -> None:
    """
    Vars are synchronized so that only channels are kept that
    are the current dataset var names.

    For varm this is done by the anndata slicing, we need to keep
    track for the .uns dataframes that stores all the mfi/fop and so on

    """

    current_var_names = adata.var_names.tolist()

    mfi_frames = [key for key in adata.uns if "mfi" in key]
    fop_frames = [key for key in adata.uns if "fop" in key]

    for frame_id in mfi_frames + fop_frames:
        if recalculate:
            ## to recalculate, we need to re-perform the dimensionality reductions.
            ## the MFI/fop values do not change if a channel is kicked out
            calculated_dimreds = _get_present_samplewise_dimreds(adata.uns[frame_id])
            _recalculate_samplewise_dimreds(adata, frame_id, calculated_dimreds)
        
        _synchronize_uns_frame(adata,
                               identifier = frame_id,
                               var_names = current_var_names)
    
    _synchronize_panel_object(adata, current_var_names)
    _synchronize_cofactors_object(adata, current_var_names)

def _synchronize_uns_frame(adata: AnnData,
                           identifier: str,
                           var_names = None) -> None:
    # we keep all the remaining var_names
    columns_to_keep = copy.copy(var_names)

    # we append columns that originated from the samplewise dimreds
    columns_to_keep += _get_samplewise_dimred_columns(adata.uns[identifier])

    adata.uns[identifier] = adata.uns[identifier].loc[:, columns_to_keep]

def _synchronize_panel_object(adata,
                              var_names) -> None:
    panel: Panel = adata.uns["panel"]
    panel.select_channels(var_names)
    print("     ... updated panel")
    return

def _synchronize_cofactors_object(adata,
                              var_names) -> None:
    try:
        cofactor_table: CofactorTable = adata.uns["cofactors"]
        cofactor_table.select_channels(var_names)
        print("     ... updated cofactors")
    except KeyError:
        """means no cofactors were supplied"""
        pass
    return