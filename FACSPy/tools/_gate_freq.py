from anndata import AnnData
import pandas as pd

from typing import Optional
from ._utils import assemble_dataframe
from ..utils import find_parents_recursively, flatten_nested_list

# Timer unit: 1e-07 s

# Total time: 0.0619396 s
# File: C:\Users\Tarik Exner\Python\FACSPy\FACSPy\tools\_gate_freq.py
# Function: calculate_gate_freq_per_parent at line 8

# Line #      Hits         Time  Per Hit   % Time  Line Contents
# ==============================================================
#      8                                           def calculate_gate_freq_per_parent(df: pd.DataFrame,
#      9                                                                              gate,
#     10                                                                              gates) -> pd.DataFrame:
#     11         1          8.0      8.0      0.0      parents = find_parents_recursively(gate) if gate != "root" else []
#     12         1         76.0     76.0      0.0      gates_of_interest = [
#     13         1          3.0      3.0      0.0          goi for goi in gates if goi not in parents and goi != gate
#     14         1          4.0      4.0      0.0      ] + ["sample_ID"]
#     15         1     380135.0 380135.0     61.4      grouped_frame = df.loc[df[gate], gates_of_interest].groupby("sample_ID")
#     16         1     124212.0 124212.0     20.1      freq_frame = grouped_frame.sum() / grouped_frame.count()
#     17         1       4903.0   4903.0      0.8      freq_frame["freq_of"] = gate
#     18         1      17158.0  17158.0      2.8      freq_frame = freq_frame.set_index("freq_of", append = True)
#     19         1      92869.0  92869.0     15.0      freq_frame = freq_frame.melt(var_name='gate',
#     20         1          4.0      4.0      0.0                                   value_name='freq',
#     21         2         20.0     10.0      0.0                                   ignore_index = False).set_index("gate", append = True)
#     22         1          4.0      4.0      0.0      return freq_frame

def calculate_gate_freq_per_parent(df: pd.DataFrame,
                                   gate,
                                   gates) -> pd.DataFrame:
    parents = find_parents_recursively(gate) if gate != "root" else []
    gates_of_interest = [
        goi for goi in gates if goi not in parents and goi != gate
    ] + ["sample_ID"]
    grouped_frame = df.loc[df[gate], gates_of_interest].groupby("sample_ID")
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
    gates = adata.uns["gating_cols"].to_list()
    
    data = assemble_dataframe(adata,
                              expression_data = False)
    data["root"] = True
    unique_gates = list(
        set(
            flatten_nested_list(
                [find_parents_recursively(gate) for gate in gates]
            ) + gates
        )
    )
    adata.uns["gate_frequencies"] = pd.concat(
        [calculate_gate_freq_per_parent(data, gate, gates)
         for gate in unique_gates],
         axis = 0
    )
    return adata if copy else None




# def gate_frequencies_old(dataset: AnnData,
#                      copy: bool = False):
    
#     gates = dataset.uns["gating_cols"].to_list()

#     gate_freqs = {}
#     for sample_id in dataset.obs["sample_ID"].unique():
#         gate_freqs[sample_id] = {}
        
#         tmp = dataset[dataset.obs["sample_ID"] == sample_id]
#         for i, gate in enumerate(gates):
            
#             gate_freqs[sample_id][gate] = {}
#             parent_list = find_parents_recursively(gate)
            
#             for parent_gate in parent_list:
                
#                 gate_freqs[sample_id][gate][parent_gate] = {}
#                 if parent_gate != "root":
#                     parent_gate_index = gates.index(parent_gate)
#                     parent_positive = tmp[tmp.obsm["gating"][:,parent_gate_index] == 1,:]
#                 else:
#                     parent_positive = tmp
                
#                 gate_freqs[sample_id][gate][parent_gate] = parent_positive.obsm["gating"][:,i].sum() / parent_positive.shape[0]
    
#     gate_freqs = {
#         (outer_key, int_key, inner_key): values
#         for outer_key, int_dict in gate_freqs.items()
#         for int_key, inner_dict in int_dict.items()
#         for inner_key, values in inner_dict.items()
#     }
    
#     da = pd.DataFrame(
#         data = {
#             "freq": gate_freqs.values()
#             }, 
#         index = pd.MultiIndex.from_tuples(gate_freqs.keys(),
#                                           names = ["sample_ID", "gate", "freq_of"])
#         )
#     print("finished...")
#     return dataset if copy else None