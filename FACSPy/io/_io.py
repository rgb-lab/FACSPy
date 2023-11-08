from anndata import AnnData

import os
import pickle

import pandas as pd
from pandas import DatetimeIndex

from ..synchronization._synchronize import hash_dataset

def make_var_valid(adata: AnnData) -> bool:
    for col in adata.var.columns:
        if adata.var[col].dtype != "category":
            adata.var[col] = adata.var[col].astype("str")
            continue
        if isinstance(adata.var[col].cat.categories, DatetimeIndex):
            adata.obs[col] = adata.obs[col].astype("str").astype("category")
            ### add warning!
    return adata

def make_obs_valid(adata: AnnData) -> bool:
    for col in adata.obs.columns:
        if adata.obs[col].dtype != "category":
            continue
        if isinstance(adata.obs[col].cat.categories, DatetimeIndex):
            adata.obs[col] = adata.obs[col].astype("str").astype("category")
            ### add warning!
    
    return adata

def save_dataset(adata: AnnData,
                 output_dir: str,
                 file_name: str,
                 overwrite: bool = False) -> None:
    """Current workaround for the fact that we store custom objects in adata.uns"""
    uns = adata.uns.copy()

    del adata.uns

    adata = make_obs_valid(adata)
    adata = make_var_valid(adata) 

    ### implement "check if file exists"
    try:
        adata.write(os.path.join(output_dir, f"{file_name}.h5ad"))
        with open(os.path.join(output_dir, f"{file_name}.uns"), "wb") as uns_metadata:
            pickle.dump(uns, uns_metadata)
    except Exception as e:
        ## if something fails, the adata object gets the uns slot back
        ## so that the user does not have to create the dataset again
        adata.uns = uns
        raise e

    adata.uns = uns

    print("File saved successfully")


def read_dataset(input_dir: str,
                 file_name: str,
                 read_uns: bool = True) -> AnnData:
    
    import scanpy as sc
    adata = sc.read_h5ad(os.path.join(input_dir, f"{file_name}.h5ad"))
    with open(os.path.join(input_dir, f"{file_name}.uns"), "rb") as uns_metadata:
        uns = pd.read_pickle(uns_metadata)
    adata.uns = uns
    ### because the PYTHONHASHSEED is changed for every session,
    ### dataset needs to be rehashed here
    hash_dataset(adata)
    return adata