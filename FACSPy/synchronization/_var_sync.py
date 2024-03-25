from anndata import AnnData
import pandas as pd

from ._utils import _reset_hash

from ..dataset._supplements import Panel, CofactorTable

def _sync_var_from_panel(adata: AnnData) -> None:
    print("\t... synchronizing dataset object to contain channels of the panel object")
    panel_var_names = adata.uns["panel"].dataframe["antigens"].tolist()
    from .._utils import time_channels, cytof_technical_channels, spectral_flow_technical_channels
    # in reality, scatter/technical channels are
    # probably often not included in the panel
    # file, as there is no need to do so.
    # however, we need to keep them.
    scatter_channels = ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W"]
    non_fluo_channels = scatter_channels + time_channels + cytof_technical_channels + spectral_flow_technical_channels
    present_technical_channels = [ch for ch in non_fluo_channels
                                  if ch in adata.var_names]
    panel_var_names += present_technical_channels
    adata._inplace_subset_var(panel_var_names)
    _reset_hash(adata, "adata_var_names")
    return

def _sync_panel_from_var(adata: AnnData) -> None:
    print("\t... synchronizing panel object to contain channels of the dataset")
    panel_frame: pd.DataFrame = adata.uns["panel"].dataframe
    panel_antigens = panel_frame["antigens"].tolist()

    var = adata.var
    var_names = adata.var_names.tolist()
    var_frame = pd.DataFrame(data = {"fcs_colname": var["pnn"].tolist(),
                                     "antigens": var["pns"].tolist()},
                             index = list(range(var.shape[0])))

    appended_channels = list(set(var_names).difference(panel_antigens))
    if appended_channels:
        panel_frame = pd.concat(
            [
                panel_frame.loc[~panel_frame["antigens"].isin(appended_channels),:],
                var_frame.loc[var_frame["antigens"].isin(appended_channels)]
            ],
            axis = 0
        )
        panel_frame = panel_frame.reset_index(drop = True)

    # we read antigens again so that they are not counted as removed
    # if they were just appended.
    panel_antigens = panel_frame["antigens"].tolist()
    removed_channels = list(set(panel_antigens).difference(var_names))
    if removed_channels:
        print("removed channels: ", removed_channels)
        panel_frame = panel_frame.loc[~panel_frame["antigens"].isin(removed_channels),:]
    
    adata.uns["panel"] = Panel(panel = panel_frame)

    _reset_hash(adata, "panel_var_names")
    return

def _sync_cofactors_from_panel(adata: AnnData) -> None:
    # CAVE technical and scatter channels!
    var_names = adata.uns["panel"].dataframe["fcs_colname"].tolist()
    try:
        cofactor_table: CofactorTable = adata.uns["cofactors"]
        cofactor_table.select_channels(var_names)
        print("     ... updated cofactors")
    except KeyError:
        """means no cofactors were supplied"""
        pass
    return

def _sync_cofactors_from_var(adata: AnnData) -> None:
    var_names = adata.var_names.tolist()
    try:
        cofactor_table: CofactorTable = adata.uns["cofactors"]
        cofactor_table.select_channels(var_names)
        print("     ... updated cofactors")
    except KeyError:
        """means no cofactors were supplied"""
        pass
    return
