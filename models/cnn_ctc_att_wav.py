import torch
import torch.nn as nn

class cnn_ctc_att_wav(nn.Module):
    def __init__(self):
        super(cnn_ctc_att_wav, self).__init__()
        self.full_stack = nn.Sequential()

    def forward(self, x):
        pass