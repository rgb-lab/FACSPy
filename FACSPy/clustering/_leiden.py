 
from ..utils import subset_fluo_channels

import scanpy as sc
import anndata as ad

from typing import Optional#

from scanpy.preprocessing import neighbors

from scanpy.tools import umap

def leiden(dataset: ad.AnnData,
           copy: bool = False) -> Optional[ad.AnnData]:
    
    data = subset_fluo_channels(dataset)
    return sc.pp.neighbors(dataset)