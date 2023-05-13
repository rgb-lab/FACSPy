 
from ..utils import subset_fluo_channels

import scanpy as sc
import anndata as ad

from typing import Optional

def leiden(dataset: ad.AnnData,
           copy: bool = False) -> Optional[ad.AnnData]:
    
    data = subset_fluo_channels(dataset)
    sc.pp.neighbors