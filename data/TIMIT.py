# created following guidelines of following tutorial: https://www.youtube.com/watch?v=88FFnqt5MNI
import json
import os
from typing import Tuple, Union

from torch.utils.data import Dataset
from timit_prepare import prepare_timit
from speechbrain.dataio.dataio import read_audio

URL = "https://data.deepai.org/timit.zip"

# function from previous version of torchaudio.datasets.utils
def walk_files(root: str,
               suffix: Union[str, Tuple[str]],
               prefix: bool = False,
               remove_suffix: bool = False):
    """List recursively all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the full path to each result, otherwise
            only returns the name of the files found (Default: ``False``)
        remove_suffix (bool, optional): If true, removes the suffix to each result defined in suffix,
            otherwise will return the result as found (Default: ``False``).
    """

    root = os.path.expanduser(root)

    for dirpath, dirs, files in os.walk(root):
        dirs.sort()
        # `dirs` is the list used in os.walk function and by sorting it in-place here, we change the
        # behavior of os.walk to traverse sub directory alphabetically
        # see also
        # https://stackoverflow.com/questions/6670029/can-i-force-python3s-os-walk-to-visit-directories-in-alphabetical-order-how#comment71993866_6670926
        files.sort()
        for f in files:
            if f.endswith(suffix):

                if remove_suffix:
                    f = f[: -len(suffix)]

                if prefix:
                    f = os.path.join(dirpath, f)

                yield f

class TIMIT(Dataset):
    def __init__(self, root: str, mode="train", download=False):
        if download:
            # TODO: implement automatic download and extraction
            pass
            # grab from https://data.deepai.org/timit.zip
        prepare_timit(root, "timit_train.json", "timit_valid.json", "timit_test.json", uppercase=True)
        if mode == "train":
            with open("timit_train.json") as json_file:
                self.data = json.load(json_file)
        elif mode == "valid":
            with open("timit_valid.json") as json_file:
                self.data = json.load(json_file)
        else:
            with open("timit_test.json") as json_file:
                self.data = json.load(json_file)

    def __len__(self):
        return len(self.data.items())

    def __getitem__(self, item):
        '''
        wav: path to wav file
        signal: tensor of raw audio data
        duration: duration of the
        spk_id: spk_id,
        phn: phonemes,
        wrd: words,
        ground_truth_phn_ends:
        '''
        datapoint = self.data[sorted(self.data.keys())[item]]
        signal = read_audio(datapoint["wav"])
        return [datapoint["wav"], signal, datapoint["duration"], datapoint["spk_id"], datapoint["phn"], datapoint["wrd"], datapoint["ground_truth_phn_ends"]]

