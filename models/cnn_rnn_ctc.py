import torch
import torch.nn as nn

class cnn_rnn_ctc(nn.Module):
    def __init__(self):
        super(cnn_rnn_ctc, self).__init__()
        self.full_stack = nn.Sequential()

    def forward(self, x):
        pass