from anndata import AnnData
import pandas as pd

from typing import Optional
from ._utils import _concat_gate_info_and_sample_ID
from .._utils import _find_parents_recursively, _flatten_nested_list

def _calculate_gate_freq_per_parent(df: pd.DataFrame,
                                    gate,
                                    gates) -> pd.DataFrame:
    parents = _find_parents_recursively(gate) if gate != "root" else []
    gates_of_interest = [
        goi for goi in gates if goi not in parents and goi != gate
    ] + ["sample_ID"]
    grouped_frame = df.loc[df[gate], gates_of_interest].groupby("sample_ID", observed = True)
    freq_frame = grouped_frame.sum() / grouped_frame.count()
    freq_frame["freq_of"] = gate
    freq_frame = freq_frame.set_index("freq_of", append = True)
    freq_frame = freq_frame.melt(var_name='gate',
                                 value_name='freq',
                                 ignore_index = False).set_index("gate", append = True)
    return freq_frame

#TODO: Does not reasonably support multiple parallel gating strategies, these will be mixed.
#TODO: if user is dumb, could lead to errors. lol
def gate_frequencies(adata: AnnData,
                     copy: bool = False) -> Optional[AnnData]:
    
    adata = adata.copy() if copy else adata
    gates = adata.uns["gating_cols"].tolist()
    
    data = _concat_gate_info_and_sample_ID(adata)
    data["root"] = True
    unique_gates = list(
        set(
            _flatten_nested_list(
                [_find_parents_recursively(gate) for gate in gates]
            ) + gates
        )
    )
    adata.uns["gate_frequencies"] = pd.concat(
        [_calculate_gate_freq_per_parent(data, gate, gates)
         for gate in unique_gates],
         axis = 0
    )
    return adata if copy else None