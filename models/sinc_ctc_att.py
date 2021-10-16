import torch
import torch.nn as nn
import speechbrain

class sinc_ctc_att(nn.Module):
    def __init__(self):
        super(sinc_ctc_att, self).__init__()
        self.full_stack = nn.Sequential()
        # https://speechbrain.readthedocs.io/en/latest/API/speechbrain.nnet.CNN.html?highlight=sincnet
        # conv = SincConv(input_shape=inp_tensor.shape, out_channels=25, kernel_size=11)

    def forward(self, x):
        pass