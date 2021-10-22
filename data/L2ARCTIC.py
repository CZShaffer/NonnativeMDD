import os
from typing import Tuple, Union
from pathlib import Path

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)

class L2ARCTIC(Dataset):

    def __init__(self):
        pass

    def __len__(self) -> int:
        pass