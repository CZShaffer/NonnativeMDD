import torch
import torch.nn as nn

class cnn_rnn_ctc(nn.Module):
    def __init__(self):
        super(cnn_rnn_ctc, self).__init__()
        # first part
        # input layer received the framewise acoustic features, followed by a batch normalization + zero padding

        # second part
        # convolution, 4 CNN layers + 2 maxpooling layer + batch normalization

        # third part
        # bi-directional RNN - GRU

        # fourth part
        # time distributed dense layer - MLP + softmax

        # last part
        # CTC output layer


    def forward(self, x):
        pass