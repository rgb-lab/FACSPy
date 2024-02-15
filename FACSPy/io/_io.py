from anndata import AnnData

import os
import pickle

import pandas as pd

from ..synchronization._synchronize import _hash_dataset
from ._utils import (_make_obs_valid,
                     _make_obsm_valid,
                     _make_obsp_valid,
                     _make_var_valid,
                     _make_varm_valid,
                     _make_varp_valid,
                     _make_layers_valid,
                     _restore_obsm_keys,
                     _restore_obsp_keys,
                     _restore_varm_keys,
                     _restore_varp_keys,
                     _restore_layers_keys)

def save_dataset(adata: AnnData,
                 output_dir: str,
                 file_name: str,
                 overwrite: bool = False) -> None:
    """Current workaround for the fact that we store custom objects in adata.uns"""

    if os.path.isfile(os.path.join(output_dir, f"{file_name}.h5ad")) and not overwrite:
        raise FileExistsError("The file already exists. Please set 'overwrite' to True")
    
    try:
        uns = adata.uns.copy()
        del adata.uns
        _make_obs_valid(adata)
        _make_var_valid(adata) 
        _make_obsm_valid(adata)
        _make_varm_valid(adata)
        _make_obsp_valid(adata)
        _make_varp_valid(adata)
        _make_layers_valid(adata)

        adata.write(os.path.join(output_dir, f"{file_name}.h5ad"))
        with open(os.path.join(output_dir, f"{file_name}.uns"), "wb") as uns_metadata:
            pickle.dump(uns, uns_metadata)

        _restore_obsm_keys(adata)
        _restore_obsp_keys(adata)
        _restore_varm_keys(adata)
        _restore_varp_keys(adata)
        _restore_layers_keys(adata)

    except Exception as e:
        ## if something fails, the adata object gets the uns slot back
        ## so that the user does not have to create the dataset again
        ## no harm done for the obs and var, but we have to restore the other slot names
        _restore_obsm_keys(adata)
        _restore_obsp_keys(adata)
        _restore_varm_keys(adata)
        _restore_varp_keys(adata)
        _restore_layers_keys(adata)
        adata.uns = uns
        raise e

    adata.uns = uns

    print("File saved successfully")


def read_dataset(input_dir: str,
                 file_name: str) -> AnnData:
    
    import scanpy as sc
    adata = sc.read_h5ad(os.path.join(input_dir, f"{file_name}.h5ad"))
    _restore_obsm_keys(adata)
    _restore_obsp_keys(adata)
    _restore_varm_keys(adata)
    _restore_varp_keys(adata)
    _restore_layers_keys(adata)

    with open(os.path.join(input_dir, f"{file_name}.uns"), "rb") as uns_metadata:
        uns = pd.read_pickle(uns_metadata)
    adata.uns = uns
    ### because the PYTHONHASHSEED is changed for every session,
    ### dataset needs to be rehashed here
    _hash_dataset(adata)
    return adata