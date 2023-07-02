import anndata as ad
import pandas as pd

from ..utils import find_parents_recursively

def gate_frequencies(dataset: ad.AnnData,
                     copy: bool = False):
    
    gates = dataset.uns["gating_cols"].to_list()

    gate_freqs = {}
    for sample_id in dataset.obs["sample_ID"].unique():
        gate_freqs[sample_id] = {}
        
        tmp = dataset[dataset.obs["sample_ID"] == sample_id]
        for i, gate in enumerate(gates):
            
            gate_freqs[sample_id][gate] = {}
            parent_list = find_parents_recursively(gate)
            
            for parent_gate in parent_list:
                
                gate_freqs[sample_id][gate][parent_gate] = {}
                if parent_gate != "root":
                    parent_gate_index = gates.index(parent_gate)
                    parent_positive = tmp[tmp.obsm["gating"][:,parent_gate_index] == 1,:]
                else:
                    parent_positive = tmp
                
                gate_freqs[sample_id][gate][parent_gate] = parent_positive.obsm["gating"][:,i].sum() / parent_positive.shape[0]
    
    gate_freqs = {
        (outer_key, int_key, inner_key): values
        for outer_key, int_dict in gate_freqs.items()
        for int_key, inner_dict in int_dict.items()
        for inner_key, values in inner_dict.items()
    }
    
    dataset.uns["gate_frequencies"] = pd.DataFrame(
        data = {
            "freq": gate_freqs.values()
            }, 
        index = pd.MultiIndex.from_tuples(gate_freqs.keys(),
                                          names = ["sample_ID", "gate", "freq_of"])
        )

    return dataset if copy else None