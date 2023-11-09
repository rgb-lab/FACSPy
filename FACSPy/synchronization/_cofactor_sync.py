from anndata import AnnData
import pandas as pd

from typing import Optional

from ..dataset._supplements import CofactorTable

def _replace_cofactors_in_var(cofactor_table: CofactorTable,
                              current_var: pd.DataFrame) -> pd.DataFrame:
    current_var["cofactors"] = current_var["cofactors"].astype(float)
    for antigen in cofactor_table.dataframe["fcs_colname"]:
        current_var.loc[current_var["pns"] == antigen, "cofactors"] = cofactor_table.get_cofactor(antigen)
    current_var["cofactors"] = current_var["cofactors"].astype("category")
    return current_var

def _replace_cofactors_in_uns(cofactor_table: CofactorTable,
                              current_var: pd.DataFrame) -> CofactorTable:
    current_var["cofactors"] = current_var["cofactors"].astype(float)
    cofactor_dataframe = cofactor_table.to_df()
    for antigen, cofactor in zip(current_var["pns"], current_var["cofactors"]):
        cofactor_dataframe.loc[cofactor_dataframe["fcs_colname"] == antigen, "cofactors"] = cofactor

    return CofactorTable(cofactors = cofactor_dataframe)   

def sync_cofactors_from_uns(adata: AnnData,
                            copy: bool = False) -> Optional[AnnData]:
    adata = adata.copy() if copy else adata
    cofactor_table: CofactorTable = adata.uns["cofactors"]
    current_var = adata.var.copy()
    adata.var = _replace_cofactors_in_var(cofactor_table,
                                          current_var)
    print("finished successfully")
    
    return adata if copy else None

def sync_cofactors_from_var(adata: AnnData,
                            copy: bool = False) -> Optional[AnnData]:
    adata = adata.copy() if copy else adata
    cofactor_table: CofactorTable = adata.uns["cofactors"]
    current_var = adata.var.copy()
    adata.uns["cofactors"] = _replace_cofactors_in_uns(cofactor_table,
                                                       current_var)

    return adata if copy else None