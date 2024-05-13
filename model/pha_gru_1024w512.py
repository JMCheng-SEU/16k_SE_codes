import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class PHA_RNN_simple(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """
    def __init__(self):
        super(PHA_RNN_simple, self).__init__()
        # shared layers

        self.dense_share1 = nn.Linear(1026,256)
        self.GRU_share1 = nn.GRU(input_size=256, hidden_size=256, num_layers=1, batch_first=True)


        # Decoder for real and imag
        self.GRU_real = nn.GRU(input_size=256, hidden_size=256, num_layers=1, batch_first=True)
        self.dense_real = nn.Linear(256, 513)

        self.GRU_imag = nn.GRU(input_size=256, hidden_size=256, num_layers=1, batch_first=True)
        self.dense_imag = nn.Linear(256, 513)



    def forward(self, x):
        # conv
        # (B, in_c, T, F)

        real_mag = x[:,0,:,:]
        imag_mag = x[:,1,:,:]

        shared_in = torch.cat([real_mag, imag_mag], dim=2)
        shared_out = self.dense_share1(shared_in)

        shared_GRU_out, hn = self.GRU_share1(shared_out)



        ##### real part

        real_1, hn = self.GRU_real(shared_GRU_out)
        real_o = self.dense_real(real_1)


        ##### imag part

        imag_1, hn = self.GRU_imag(shared_GRU_out)
        imag_o = self.dense_imag(imag_1)

        ##### concat tensor

        real_o = torch.unsqueeze(real_o, dim=1)
        imag_o = torch.unsqueeze(imag_o, dim=1)



        return real_o, imag_o