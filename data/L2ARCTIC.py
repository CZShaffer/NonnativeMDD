# created following guidelines of following tutorial: https://www.youtube.com/watch?v=88FFnqt5MNI
import os
import json
from typing import Tuple, Union
import torchaudio
import torch
from torch.utils.data import Dataset
from speechbrain.dataio.dataio import read_audio
import textgrid

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

def load_L2ARCTIC(file: str):
    data_path = os.path.splitext(file)[0]
    split_path =  data_path.split("\\")
    annotation_path = split_path.copy()
    annotation_path[-2] = "annotation"
    textgrid_path = split_path.copy()
    textgrid_path[-2] = "textgrid"
    transcript_path = split_path.copy()
    transcript_path[-2] = "transcript"
    #Manual annotation files does not have all corresponding entries
    manual_phonemes = None
    manual_words = None
    manual_phone_ends = None
    if os.path.exists("\\".join(annotation_path) + '.TextGrid'):
        tg = textgrid.TextGrid()
        # annotation_file = tg.read("\\".join(annotation_path) + '.TextGrid')#, round_digits=15
        annotation_file = textgrid.TextGrid.fromFile("\\".join(annotation_path) + '.TextGrid')
        manual_phonemes = []
        manual_words = []
        manual_phone_ends = []
        for item in annotation_file[0]:
            manual_words.append(item.mark)
        for item in annotation_file[1]:
            # if a mispronunciation, is in form (correct phoneme, Percieved (Incorrect) phoneme, change type)
            #TODO: verify this is how we want to handle it
            phones = item.mark.split(",")
            if len(phones) > 1:
                manual_phonemes.append((phones[1].strip("*"), phones[0], phones[2]))
            else:
                manual_phonemes.append((item.mark, item.mark, None))
            manual_phone_ends.append(item.maxTime)
    auto_phonemes = []
    auto_words = []
    auto_phone_ends = []
    textgrid_file = textgrid.TextGrid.fromFile("\\".join(textgrid_path) + '.TextGrid')
    for item in textgrid_file[0]:
        auto_words.append(item.mark)
    for item in textgrid_file[1]:
        auto_phonemes.append(item.mark)
        auto_phone_ends.append(item.maxTime)
    with open("\\".join(transcript_path) + '.txt', 'r') as transcript_file:
        transcript = next(iter(transcript_file)).strip()
    signal, sr = torchaudio.load(data_path + '.wav')
    base_path = "\\".join(split_path)
    duration = torch.numel(signal)/sr
    return base_path, transcript, (manual_words, auto_words), (manual_phonemes, auto_phonemes), (manual_phone_ends, auto_phone_ends), duration

class L2ARCTIC(Dataset):

    def __init__(self, root: str, download=False):
        self.root = root
        self._walker = sorted(list(walk_files(root, suffix='.wav', prefix=True)), reverse=True)
        #modified L2_ARCTIC\L2_ARCTIC\YDCK\YDCK\wav\arctic_a0209.wav  to correct interval data
        #modified L2_ARCTIC\L2_ARCTIC\YDCK\YDCK\wav\arctic_a0272.wav  to correct interval data
        # self._walker = ["L2_ARCTIC\\L2_ARCTIC\\YDCK\\YDCK\\wav\\arctic_a0209.wav"]
        self._data = {}
        if not os.path.exists("L2_ARCTIC.json"):
            for item in self._walker:
                if "suitcase_corpus" in item:
                    continue
                datapoint = load_L2ARCTIC(item)
                spk_id = datapoint[0].split("\\")[-4]
                snt_id = datapoint[0].split("\\")[-3].replace(".wav", "")
                snt_id = spk_id + "_" + snt_id
                self._data[snt_id] = {
                    "base_path":datapoint[0],
                    "transcript":datapoint[1],
                    "words":datapoint[2],
                    "phonemes":datapoint[3],
                    "phone_ends":datapoint[4],
                    "duration":datapoint[5]
                }
            with open("L2_ARCTIC.json", mode="w") as json_file:
                json.dump(self._data, json_file, indent=2)
        else:
            with open("L2_ARCTIC.json") as json_file:
                self._data = json.load(json_file)

    def __len__(self):
        return len(self._walker)

    def __getitem__(self, item):
        '''
        base_path: base path corresponding to the files associated with a given .wav file,
        signal: tensor of raw audio data
        transcript: string of transcript for given wav file
        words: List of words in transcript, including pauses
        phonemes: list of manual and automatically aligned phonemes
            -manual: tuple of (perceived phone, correct phone, phone change type)
            -automatic: list of perceived phones
        phone_ends: list of end times for manual and automatically aligned phonemes
            -manual: list of end times for phones corresponding to manual alignment
            -automatic: list of end times for phones corresponding to automatic alignment
        duration: duration of the wav in seconds
        '''
        datapoint = self._data[sorted(self._data.keys())[item]]
        signal = read_audio(datapoint["base_path"]+".wav")
        return [datapoint["base_path"], signal, datapoint["transcript"],datapoint["words"],datapoint["phonemes"],datapoint["phone_ends"],datapoint["duration"]]

