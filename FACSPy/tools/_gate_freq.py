from anndata import AnnData
import pandas as pd
import numpy as np

from typing import Optional
from ._utils import _concat_gate_info_and_sample_ID
from .._utils import _find_parents_recursively, _flatten_nested_list
from scipy.sparse import csr_matrix, hstack

from typing import Any

def gate_frequencies_mem(adata: AnnData,
                         copy: bool = False) -> Optional[AnnData]:
    """same as gate_frequencies, but less memory. Slightly slower."""
    
    adata = adata.copy() if copy else adata

    sample_ID_cat_codes = adata.obs["sample_ID"].cat.codes.to_numpy()
    sample_IDs = adata.obs["sample_ID"].to_numpy()
    sample_ID_cat_map = dict(zip(sample_ID_cat_codes, sample_IDs))

    gates = adata.uns["gating_cols"].tolist()
    unique_gates = list(
        set(
            _flatten_nested_list(
                [_find_parents_recursively(gate) for gate in gates]
            ) + gates
        )
    )
    gates = ["root"] + gates

    gate_matrix = adata.obsm["gating"]
    # we have to append a "root" column that is all True
    gate_matrix = hstack([
        csr_matrix(np.full(shape = (gate_matrix.shape[0], 1), fill_value = True)),
        gate_matrix
    ])

    # sort if necessary
    is_sorted = (np.diff(sample_ID_cat_codes) >= 0).all()
    if not is_sorted:
        sorted_sample_IDs = np.argsort(sample_ID_cat_codes)
        sample_ID_cat_codes = sample_ID_cat_codes[sorted_sample_IDs]
        gate_matrix = gate_matrix[sorted_sample_IDs,:]

    adata.uns["gate_frequencies"] = pd.concat(
        [_calculate_gate_freq_per_parent_mem(gate_matrix, gate, gates, sample_ID_cat_codes, sample_ID_cat_map)
         for gate in unique_gates],
         axis = 0
    )
    return adata if copy else None

def _calculate_gate_freq_per_parent_mem(gate_mtx: csr_matrix,
                                        gate: str,
                                        gates: list[str],
                                        sample_IDs: list[str],
                                        sample_ID_int_map: dict) -> Any:
    parents = _find_parents_recursively(gate) if gate != "root" else []
    gates_of_interest = [
        goi for goi in gates if goi not in parents and goi != gate
    ]
    # we look up the indices of the gates in order to index the csr matrix
    gate_indices = [gates.index(gate) for gate in gates_of_interest]

    # we select for the gate-postive events in order to calculate the children frequencies
    gate_index = gates.index(gate)
    # note this takes a lot of time...
    gate_positive_events = gate_mtx[:, gate_index] == True

    gate_positive_events = gate_positive_events.toarray().ravel() 
    gate_mtx = gate_mtx[gate_positive_events,:]
    sample_IDs = sample_IDs[gate_positive_events]

    # we calculate the unique sampleIDs and their indices
    # note that sample_IDs here are the int cat codes
    uniques, sid_idxs, sid_counts = np.unique(sample_IDs, return_index = True, return_counts = True)

    # we append the last index of the matrix in order to slice the last slice precisely
    sid_idxs = np.append(sid_idxs, gate_mtx.shape[0])

    # positive counts are defined as non-zero entries.
    # we index the columns by the gate indices and the rows by unique sample IDs
    # (essentially a groupby function but without the memory consumption)
    positive_counts = [
        gate_mtx[
            sid_idxs[i]:sid_idxs[i+1],
            gate_indices
        ].getnnz(axis = 0)
        for i, _ in enumerate(sid_idxs)
        if i < sid_idxs.shape[0]-1
    ]
    
    # frequencies in percent are calculated by dividing the positive count array by
    # the total count per sample_ID
    freqs = np.array(
        [np.divide(pos_count, sid_count)
         for pos_count, sid_count
         in zip(positive_counts, sid_counts)]
    )
    uniques = [sample_ID_int_map[entry] for entry in uniques]
    # we construct a small dataframe and perform reindexing and melting
    freq_frame = pd.DataFrame(data = freqs,
                              index = uniques,
                              columns = gates_of_interest)
    freq_frame.index.name = "sample_ID"

    freq_frame["freq_of"] = gate
    freq_frame = freq_frame.set_index("freq_of", append = True)
    freq_frame = freq_frame.melt(var_name='gate',
                                 value_name='freq',
                                 ignore_index = False).set_index("gate", append = True)
    return freq_frame


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