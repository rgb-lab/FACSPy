import pandas as pd
from ..dataset.supplements import Metadata
from anndata import AnnData
from typing import Optional, Union

def synchronize_vars(adata: AnnData,
                     recalculate: bool = False) -> None:
    """
    Vars are synchronized so that only channels are kept that
    are the current dataset var names.

    Importantly! The panel and cofactor metadata are unchanged for now!

    """

    current_var_names = adata.var_names.to_list()

    mfi_frames = [key for key in adata.uns if "mfi" in key]
    fop_frames = [key for key in adata.uns if "fop" in key]

    for uns_frame in mfi_frames + fop_frames:
        if recalculate:
            _placeholder()
        _synchronize_uns_frame(adata,
                                identifier = uns_frame,
                                var_names = current_var_names)


def _placeholder(): pass

def _synchronize_uns_frame(adata: AnnData,
                           identifier: str,
                           var_names = None) -> None:
    adata.uns[identifier] = adata.uns[identifier].loc[:, var_names]
