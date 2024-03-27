from pathlib import Path
import os 
from ..io._io import read_dataset

def mouse_lineages():
    dataset_dir = Path(__file__).parent
    file = os.path.join(dataset_dir, "mouse_lineages_downsampled")
    return read_dataset(file_name = file)
