
import os
from os import PathLike
import pickle
from anndata import AnnData
import scanpy as sc
import pandas as pd
import warnings

from typing import Optional, MutableMapping

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
from ..synchronization._synchronize import _hash_dataset

def save_dataset(adata: AnnData,
                 output_dir: Optional[PathLike] = None,
                 file_name: Optional[PathLike] = None,
                 overwrite: bool = False,
                 **kwargs
                 ) -> None:
    """\
    Saves the dataset as an .h5ad file. Because we are storing custom objects in adata.uns,
    the function saves the .uns metadata separately and reads it back upon import.

    Parameters
    ----------

    adata
        The anndata object to save.
    output_dir
        Deprecated in favor of a PathLike file_name.
    file_name
        Path to save to.
    overwrite
        If set to True, the function will overwrite the file that has the same filename
    **kwargs
        Keyword arguments passed to the AnnData.write() method. Please refer to their documentation.

    Returns
    -------
    None

    Examples
    --------

    >>> import FACSPy as fp
    >>> dataset = fp.create_dataset(...)
    >>> fp.save_dataset(dataset, "raw_dataset.h5ad", overwrite = True)
    
    """
    if output_dir:
        warnings.warn("The parameter `output_dir` is deprecated and will be removed in future versions", DeprecationWarning)
        file_name = os.path.join(output_dir, file_name)

    file: str = os.path.basename(file_name)
    if not file.endswith(".h5ad"):
        file_name: str = file_name + ".h5ad"

    if os.path.isfile(file_name) and not overwrite:
        raise FileExistsError("The file already exists. Please set 'overwrite' to True")
        
    uns = adata.uns.copy()
    try:
        del adata.uns
        _make_obs_valid(adata)
        _make_var_valid(adata) 
        _make_obsm_valid(adata)
        _make_varm_valid(adata)
        _make_obsp_valid(adata)
        _make_varp_valid(adata)
        _make_layers_valid(adata)

        adata.write(file_name, **kwargs)
        uns_name = file_name.replace(".h5ad", ".uns")
        with open(uns_name, "wb") as uns_metadata:
            pickle.dump(uns, uns_metadata)

        _restore_obsm_keys(adata)
        _restore_obsp_keys(adata)
        _restore_varm_keys(adata)
        _restore_varp_keys(adata)
        _restore_layers_keys(adata)

    except Exception as e:
        # if something fails, the adata object gets the uns slot back
        # so that the user does not have to create the dataset again
        # no harm done for the obs and var, but we have to restore the other slot names
        _restore_obsm_keys(adata)
        _restore_obsp_keys(adata)
        _restore_varm_keys(adata)
        _restore_varp_keys(adata)
        _restore_layers_keys(adata)
        adata.uns = uns
        raise e

    adata.uns = uns

    print("File saved successfully")


def read_dataset(input_dir: Optional[str] = None,
                 file_name: Optional[PathLike] = None) -> AnnData:
    """\
    Reads the dataset from the hard drive. 

    Parameters
    ----------

    input_dir
        Deprecated in favor of a PathLike file_name.
    file_name
        Path to read from.

    Returns
    -------
    :class:`~anndata.AnnData`

    Examples
    --------

    >>> import FACSPy as fp
    >>> dataset = fp.read_dataset(file_name = "../raw_data.h5ad")
    
    """

    if not file_name:
        raise ValueError("Please provide a file name")

    if input_dir:
        warnings.warn("The parameter `input_dir` is deprecated and will be removed in future versions", DeprecationWarning)
        file_name = os.path.join(input_dir, file_name)

    file: str = os.path.basename(file_name)
    if not file.endswith(".h5ad"):
        file_name: str = file_name + ".h5ad"

    adata = sc.read_h5ad(file_name)
    _restore_obsm_keys(adata)
    _restore_obsp_keys(adata)
    _restore_varm_keys(adata)
    _restore_varp_keys(adata)
    _restore_layers_keys(adata)
    
    uns_name = file_name.replace(".h5ad", ".uns")
    with open(uns_name, "rb") as uns_metadata:
        uns: MutableMapping = pd.read_pickle(uns_metadata)
    adata.uns = uns

    # because the PYTHONHASHSEED is changed for every session,
    # dataset needs to be rehashed here
    _hash_dataset(adata)
    return adata
