import torch
import torch.nn as nn

class cnn_ctc_att_mfcc(nn.Module):
    def __init__(self):
        super(cnn_ctc_att_mfcc, self).__init__()
        self.full_stack = nn.Sequential()

    def forward(self, x):
        pass