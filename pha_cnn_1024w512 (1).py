# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# from torch.autograd import Variable
#
#
# class FCNN(nn.Module):
#     """
#     Input: [batch size, channels=1, T, n_fft]
#     Output: [batch size, T, n_fft]
#     """
#
#     def __init__(self):
#         super(FCNN, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
#         # self.bn1 = nn.BatchNorm2d(num_features=8)
#         self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
#         # self.bn2 = nn.BatchNorm2d(num_features=16)
#         self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
#         # self.bn3 = nn.BatchNorm2d(num_features=32)
#         self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
#         # self.bn4 = nn.BatchNorm2d(num_features=64)
#         self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
#         # self.bn5 = nn.BatchNorm2d(num_features=128)
#
#         self.conv_mid = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1)
#         # self.bn_mid = nn.BatchNorm2d(num_features=128)
#
#
#
#
#         # self.fcs_mid = nn.Sequential(
#         #     nn.Linear(2048, 2048),
#         #     nn.ELU()
#         # )
#
#         # Decoder for real
#
#
#
#         self.convT1 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
#         # self.bnT1 = nn.BatchNorm2d(num_features=64)
#         self.convT2 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
#         # self.bnT2 = nn.BatchNorm2d(num_features=32)
#         self.convT3 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
#         # self.bnT3 = nn.BatchNorm2d(num_features=16)
#         # output_padding为1，不然算出来是79
#         self.convT4 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
#         # self.bnT4 = nn.BatchNorm2d(num_features=8)
#         self.convT5 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
#         # self.bnT5 = nn.BatchNorm2d(num_features=1)
#
#         self.fcs = nn.Sequential(
#             nn.Linear(513, 513),
#             nn.Tanh(),
#         )
#
#         # Decoder for imag
#
#
#         self.phaconvT1 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
#         # self.phabnT1 = nn.BatchNorm2d(num_features=64)
#         self.phaconvT2 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
#         # self.phabnT2 = nn.BatchNorm2d(num_features=32)
#         self.phaconvT3 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
#         # self.phabnT3 = nn.BatchNorm2d(num_features=16)
#         # output_padding为1，不然算出来是79
#         self.phaconvT4 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
#         # self.phabnT4 = nn.BatchNorm2d(num_features=8)
#         self.phaconvT5 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
#         # self.phabnT5 = nn.BatchNorm2d(num_features=1)
#
#         self.phafcs = nn.Sequential(
#             nn.Linear(513, 513),
#             nn.Tanh(),
#         )
#
#     def forward(self, x):
#         # conv
#         # (B, in_c, T, F)
#
#
#         # x1 = F.relu(self.bn1(self.conv1(x)))
#         # x2 = F.relu(self.bn2(self.conv2(x1)))
#         # x3 = F.relu(self.bn3(self.conv3(x2)))
#         # x4 = F.relu(self.bn4(self.conv4(x3)))
#         # x5 = F.relu(self.bn5(self.conv5(x4)))
#
#         x1 = F.relu(self.conv1(x))
#         x2 = F.relu(self.conv2(x1))
#         x3 = F.relu(self.conv3(x2))
#         x4 = F.relu(self.conv4(x3))
#         x5 = F.relu(self.conv5(x4))
#
#
#         output = F.relu(self.conv_mid(x5))
#
#         # out5 = x5.permute(0, 2, 1, 3)
#         # out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
#         # # # # lstm
#         # # #
#         # # # lstm, (hn, cn) = self.LSTM1(out5)
#         # # # # # reshape
#         # # # output = lstm.reshape(lstm.size()[0], lstm.size()[1], 256, -1)
#         # # # output = output.permute(0, 2, 1, 3)
#         # #
#         # # ####FC
#         # output = self.fcs_mid(out5)
#         # output = output.reshape(x5.size()[0], x5.size()[2], 128, -1)
#         # output = output.permute(0, 2, 1, 3)
#
#
#         # ConvTrans for real
#         res = torch.cat((output, x5), 1)
#         res1 = F.relu(self.convT1(res))
#         res1 = torch.cat((res1[:,:,:,:32], x4[:,:,:,:32]), 1)
#         res2 = F.relu(self.convT2(res1))
#         res2 = torch.cat((res2[:,:,:,:64], x3[:,:,:,:64]), 1)
#         res3 = F.relu(self.convT3(res2))
#         res3 = torch.cat((res3[:,:,:,:128], x2[:,:,:,:128]), 1)
#         res4 = F.relu(self.convT4(res3))
#         res4 = torch.cat((res4[:,:,:,:256], x1[:,:,:,:256]), 1)
#         # (B, o_c, T. F)
#         res5 = F.relu(self.convT5(res4))
#         res5 = self.fcs(res5)
#
#         # ConvTrans for imag
#         phares1 = F.relu(self.phaconvT1(res))
#         phares1 = torch.cat((phares1[:,:,:,:32], x4[:,:,:,:32]), 1)
#         phares2 = F.relu(self.phaconvT2(phares1))
#         phares2 = torch.cat((phares2[:,:,:,:64], x3[:,:,:,:64]), 1)
#         phares3 = F.relu(self.phaconvT3(phares2))
#         phares3 = torch.cat((phares3[:,:,:,:128], x2[:,:,:,:128]), 1)
#         phares4 = F.relu(self.phaconvT4(phares3))
#         phares4 = torch.cat((phares4[:,:,:,:256], x1[:,:,:,:256]), 1)
#         # (B, o_c, T. F)
#         phares5 = F.relu(self.phaconvT5(phares4))
#         phares5 = self.phafcs(phares5)
#
#
#
#         # # ConvTrans for real
#         # res = torch.cat((output, x5), 1)
#         # res1 = F.relu(self.bnT1(self.convT1(res)))
#         # res1 = torch.cat((res1[:,:,:,:32], x4[:,:,:,:32]), 1)
#         # res2 = F.relu(self.bnT2(self.convT2(res1)))
#         # res2 = torch.cat((res2[:,:,:,:64], x3[:,:,:,:64]), 1)
#         # res3 = F.relu(self.bnT3(self.convT3(res2)))
#         # res3 = torch.cat((res3[:,:,:,:128], x2[:,:,:,:128]), 1)
#         # res4 = F.relu(self.bnT4(self.convT4(res3)))
#         # res4 = torch.cat((res4[:,:,:,:256], x1[:,:,:,:256]), 1)
#         # # (B, o_c, T. F)
#         # res5 = F.relu(self.bnT5(self.convT5(res4)))
#         # res5 = self.fcs(res5)
#         #
#         # # ConvTrans for imag
#         # phares1 = F.relu(self.phabnT1(self.phaconvT1(res)))
#         # phares1 = torch.cat((phares1[:,:,:,:32], x4[:,:,:,:32]), 1)
#         # phares2 = F.relu(self.phabnT2(self.phaconvT2(phares1)))
#         # phares2 = torch.cat((phares2[:,:,:,:64], x3[:,:,:,:64]), 1)
#         # phares3 = F.relu(self.phabnT3(self.phaconvT3(phares2)))
#         # phares3 = torch.cat((phares3[:,:,:,:128], x2[:,:,:,:128]), 1)
#         # phares4 = F.relu(self.phabnT4(self.phaconvT4(phares3)))
#         # phares4 = torch.cat((phares4[:,:,:,:256], x1[:,:,:,:256]), 1)
#         # # (B, o_c, T. F)
#         # phares5 = F.relu(self.phabnT5(self.phaconvT5(phares4)))
#         # phares5 = self.phafcs(phares5)
#
#
#         # enh_spec = torch.cat((res5, phares5), 1)
#         return res5, phares5
#         # return real_comu, imag_comu
#
#
#
# class FCNN_slim(nn.Module):
#     """
#     Input: [batch size, channels=1, T, n_fft]
#     Output: [batch size, T, n_fft]
#     """
#
#     def __init__(self):
#         super(FCNN_slim, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
#         # self.bn1 = nn.BatchNorm2d(num_features=8)
#         self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
#         # self.bn2 = nn.BatchNorm2d(num_features=16)
#         self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
#         # self.bn3 = nn.BatchNorm2d(num_features=32)
#         self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
#         # self.bn4 = nn.BatchNorm2d(num_features=64)
#         self.conv5 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
#         # self.bn5 = nn.BatchNorm2d(num_features=128)
#
#         self.conv_mid = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1)
#         # self.bn_mid = nn.BatchNorm2d(num_features=128)
#
#
#
#
#         # self.fcs_mid = nn.Sequential(
#         #     nn.Linear(2048, 2048),
#         #     nn.ELU()
#         # )
#
#         # Decoder for real
#
#
#
#         self.convT1 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
#         # self.bnT1 = nn.BatchNorm2d(num_features=64)
#         self.convT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
#         # self.bnT2 = nn.BatchNorm2d(num_features=32)
#         self.convT3 = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
#         # self.bnT3 = nn.BatchNorm2d(num_features=16)
#         # output_padding为1，不然算出来是79
#         self.convT4 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
#         # self.bnT4 = nn.BatchNorm2d(num_features=8)
#         self.convT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
#         # self.bnT5 = nn.BatchNorm2d(num_features=1)
#
#         self.fcs = nn.Sequential(
#             nn.Linear(513, 513),
#             nn.Tanh(),
#         )
#
#         # Decoder for imag
#
#
#         self.phaconvT1 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
#         # self.phabnT1 = nn.BatchNorm2d(num_features=64)
#         self.phaconvT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
#         # self.phabnT2 = nn.BatchNorm2d(num_features=32)
#         self.phaconvT3 = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
#         # self.phabnT3 = nn.BatchNorm2d(num_features=16)
#         # output_padding为1，不然算出来是79
#         self.phaconvT4 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
#         # self.phabnT4 = nn.BatchNorm2d(num_features=8)
#         self.phaconvT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
#         # self.phabnT5 = nn.BatchNorm2d(num_features=1)
#
#         self.phafcs = nn.Sequential(
#             nn.Linear(513, 513),
#             nn.Tanh(),
#         )
#
#     def forward(self, x):
#         # conv
#         # (B, in_c, T, F)
#
#
#         # x1 = F.relu(self.bn1(self.conv1(x)))
#         # x2 = F.relu(self.bn2(self.conv2(x1)))
#         # x3 = F.relu(self.bn3(self.conv3(x2)))
#         # x4 = F.relu(self.bn4(self.conv4(x3)))
#         # x5 = F.relu(self.bn5(self.conv5(x4)))
#
#         x1 = F.relu(self.conv1(x))
#         x2 = F.relu(self.conv2(x1))
#         x3 = F.relu(self.conv3(x2))
#         x4 = F.relu(self.conv4(x3))
#         x5 = F.relu(self.conv5(x4))
#
#
#         output = F.relu(self.conv_mid(x5))
#
#         # out5 = x5.permute(0, 2, 1, 3)
#         # out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
#         # # # # lstm
#         # # #
#         # # # lstm, (hn, cn) = self.LSTM1(out5)
#         # # # # # reshape
#         # # # output = lstm.reshape(lstm.size()[0], lstm.size()[1], 256, -1)
#         # # # output = output.permute(0, 2, 1, 3)
#         # #
#         # # ####FC
#         # output = self.fcs_mid(out5)
#         # output = output.reshape(x5.size()[0], x5.size()[2], 128, -1)
#         # output = output.permute(0, 2, 1, 3)
#
#
#         # ConvTrans for real
#         res = torch.cat((output, x5), 1)
#         res1 = F.relu(self.convT1(res))
#         res1 = torch.cat((res1[:,:,:,:32], x4[:,:,:,:32]), 1)
#         res2 = F.relu(self.convT2(res1))
#         res2 = torch.cat((res2[:,:,:,:64], x3[:,:,:,:64]), 1)
#         res3 = F.relu(self.convT3(res2))
#         res3 = torch.cat((res3[:,:,:,:128], x2[:,:,:,:128]), 1)
#         res4 = F.relu(self.convT4(res3))
#         res4 = torch.cat((res4[:,:,:,:256], x1[:,:,:,:256]), 1)
#         # (B, o_c, T. F)
#         res5 = F.relu(self.convT5(res4))
#         print(list(self.fcs.named_parameters()))
#         res5 = self.fcs(res5)
#
#         # ConvTrans for imag
#         phares1 = F.relu(self.phaconvT1(res))
#         phares1 = torch.cat((phares1[:,:,:,:32], x4[:,:,:,:32]), 1)
#         phares2 = F.relu(self.phaconvT2(phares1))
#         phares2 = torch.cat((phares2[:,:,:,:64], x3[:,:,:,:64]), 1)
#         phares3 = F.relu(self.phaconvT3(phares2))
#         phares3 = torch.cat((phares3[:,:,:,:128], x2[:,:,:,:128]), 1)
#         phares4 = F.relu(self.phaconvT4(phares3))
#         phares4 = torch.cat((phares4[:,:,:,:256], x1[:,:,:,:256]), 1)
#         # (B, o_c, T. F)
#         phares5 = F.relu(self.phaconvT5(phares4))
#         phares5 = self.phafcs(phares5)
#
#
#
#
#         # enh_spec = torch.cat((res5, phares5), 1)
#         return res5, phares5
#         # return real_comu, imag_comu

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

class GLayerNorm2d(nn.Module):

    def __init__(self, in_channel, eps=1e-12):
        super(GLayerNorm2d, self).__init__()
        self.eps = eps
        self.beta = nn.Parameter(torch.ones([1, in_channel, 1, 1]))
        self.gamma = nn.Parameter(torch.zeros([1, in_channel, 1, 1]))

    def forward(self, inputs):
        mean = torch.mean(inputs, [1, 2, 3], keepdim=True)
        var = torch.var(inputs, [1, 2, 3], keepdim=True)
        tmp1 = 1. / torch.sqrt(var + self.eps)
        outputs = (inputs - mean) / torch.sqrt(var + self.eps) * self.beta + self.gamma
        return outputs

class FCNN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(FCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        # self.bn1 = nn.BatchNorm2d(num_features=8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        # self.bn2 = nn.BatchNorm2d(num_features=16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        # self.bn3 = nn.BatchNorm2d(num_features=32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        # self.bn4 = nn.BatchNorm2d(num_features=64)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
        # self.bn5 = nn.BatchNorm2d(num_features=128)

        self.conv_mid = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1)
        # self.bn_mid = nn.BatchNorm2d(num_features=128)




        # self.fcs_mid = nn.Sequential(
        #     nn.Linear(2048, 2048),
        #     nn.ELU()
        # )

        # Decoder for real



        self.convT1 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        # self.bnT1 = nn.BatchNorm2d(num_features=64)
        self.convT2 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        # self.bnT2 = nn.BatchNorm2d(num_features=32)
        self.convT3 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        # self.bnT3 = nn.BatchNorm2d(num_features=16)
        # output_padding为1，不然算出来是79
        self.convT4 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        # self.bnT4 = nn.BatchNorm2d(num_features=8)
        self.convT5 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        # self.bnT5 = nn.BatchNorm2d(num_features=1)

        self.fcs = nn.Sequential(
            nn.Linear(513, 513),
            nn.Tanh(),
        )

        # Decoder for imag


        self.phaconvT1 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        # self.phabnT1 = nn.BatchNorm2d(num_features=64)
        self.phaconvT2 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        # self.phabnT2 = nn.BatchNorm2d(num_features=32)
        self.phaconvT3 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        # self.phabnT3 = nn.BatchNorm2d(num_features=16)
        # output_padding为1，不然算出来是79
        self.phaconvT4 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        # self.phabnT4 = nn.BatchNorm2d(num_features=8)
        self.phaconvT5 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        # self.phabnT5 = nn.BatchNorm2d(num_features=1)

        self.phafcs = nn.Sequential(
            nn.Linear(513, 513),
            nn.Tanh(),
        )

    def forward(self, x):
        # conv
        # (B, in_c, T, F)


        # x1 = F.relu(self.bn1(self.conv1(x)))
        # x2 = F.relu(self.bn2(self.conv2(x1)))
        # x3 = F.relu(self.bn3(self.conv3(x2)))
        # x4 = F.relu(self.bn4(self.conv4(x3)))
        # x5 = F.relu(self.bn5(self.conv5(x4)))

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))


        output = F.relu(self.conv_mid(x5))

        # out5 = x5.permute(0, 2, 1, 3)
        # out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
        # # # # lstm
        # # #
        # # # lstm, (hn, cn) = self.LSTM1(out5)
        # # # # # reshape
        # # # output = lstm.reshape(lstm.size()[0], lstm.size()[1], 256, -1)
        # # # output = output.permute(0, 2, 1, 3)
        # #
        # # ####FC
        # output = self.fcs_mid(out5)
        # output = output.reshape(x5.size()[0], x5.size()[2], 128, -1)
        # output = output.permute(0, 2, 1, 3)


        # ConvTrans for real
        res = torch.cat((output, x5), 1)
        res1 = F.relu(self.convT1(res))
        res1 = torch.cat((res1[:,:,:,:32], x4[:,:,:,:32]), 1)
        res2 = F.relu(self.convT2(res1))
        res2 = torch.cat((res2[:,:,:,:64], x3[:,:,:,:64]), 1)
        res3 = F.relu(self.convT3(res2))
        res3 = torch.cat((res3[:,:,:,:128], x2[:,:,:,:128]), 1)
        res4 = F.relu(self.convT4(res3))
        res4 = torch.cat((res4[:,:,:,:256], x1[:,:,:,:256]), 1)
        # (B, o_c, T. F)
        res5 = F.relu(self.convT5(res4))
        res5 = self.fcs(res5)

        # ConvTrans for imag
        phares1 = F.relu(self.phaconvT1(res))
        phares1 = torch.cat((phares1[:,:,:,:32], x4[:,:,:,:32]), 1)
        phares2 = F.relu(self.phaconvT2(phares1))
        phares2 = torch.cat((phares2[:,:,:,:64], x3[:,:,:,:64]), 1)
        phares3 = F.relu(self.phaconvT3(phares2))
        phares3 = torch.cat((phares3[:,:,:,:128], x2[:,:,:,:128]), 1)
        phares4 = F.relu(self.phaconvT4(phares3))
        phares4 = torch.cat((phares4[:,:,:,:256], x1[:,:,:,:256]), 1)
        # (B, o_c, T. F)
        phares5 = F.relu(self.phaconvT5(phares4))
        phares5 = self.phafcs(phares5)



        # # ConvTrans for real
        # res = torch.cat((output, x5), 1)
        # res1 = F.relu(self.bnT1(self.convT1(res)))
        # res1 = torch.cat((res1[:,:,:,:32], x4[:,:,:,:32]), 1)
        # res2 = F.relu(self.bnT2(self.convT2(res1)))
        # res2 = torch.cat((res2[:,:,:,:64], x3[:,:,:,:64]), 1)
        # res3 = F.relu(self.bnT3(self.convT3(res2)))
        # res3 = torch.cat((res3[:,:,:,:128], x2[:,:,:,:128]), 1)
        # res4 = F.relu(self.bnT4(self.convT4(res3)))
        # res4 = torch.cat((res4[:,:,:,:256], x1[:,:,:,:256]), 1)
        # # (B, o_c, T. F)
        # res5 = F.relu(self.bnT5(self.convT5(res4)))
        # res5 = self.fcs(res5)
        #
        # # ConvTrans for imag
        # phares1 = F.relu(self.phabnT1(self.phaconvT1(res)))
        # phares1 = torch.cat((phares1[:,:,:,:32], x4[:,:,:,:32]), 1)
        # phares2 = F.relu(self.phabnT2(self.phaconvT2(phares1)))
        # phares2 = torch.cat((phares2[:,:,:,:64], x3[:,:,:,:64]), 1)
        # phares3 = F.relu(self.phabnT3(self.phaconvT3(phares2)))
        # phares3 = torch.cat((phares3[:,:,:,:128], x2[:,:,:,:128]), 1)
        # phares4 = F.relu(self.phabnT4(self.phaconvT4(phares3)))
        # phares4 = torch.cat((phares4[:,:,:,:256], x1[:,:,:,:256]), 1)
        # # (B, o_c, T. F)
        # phares5 = F.relu(self.phabnT5(self.phaconvT5(phares4)))
        # phares5 = self.phafcs(phares5)


        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5
        # return real_comu, imag_comu

class FCNN_midGRU_withPRELU(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(FCNN_midGRU_withPRELU, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=4),
            nn.PReLU(),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=4),
            nn.PReLU(),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),

        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),

        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=16),
            nn.PReLU(),

        )

        self.conv_mid_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, stride=1),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),
        )

        self.GRU_mid = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True)

        self.conv_mid_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, stride=1),
            GLayerNorm2d(in_channel=16),
            nn.PReLU(),
        )

        # Decoder for real

        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),

        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),

        )
        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=4),
            nn.PReLU(),

        )
        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=1),
            nn.PReLU(),

        )


        self.GRU_real = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)

        self.fcs = nn.Sequential(
            nn.Linear(257, 513),
        )
        # Decoder for imag


        self.phaconvT1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),
        )
        self.phaconvT2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),
        )
        self.phaconvT3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=4),
            nn.PReLU(),
        )
        self.phaconvT4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=4),
            nn.PReLU(),
        )
        self.phaconvT5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=1),
            nn.Tanh(),
        )

        # self.GRU_imag = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)
        # self.phafcs = nn.Sequential(
        #     nn.Linear(257, 513),
        # )

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def forward(self, x):
        # conv
        # (B, in_c, T, F)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        mid_in = self.conv_mid_1(x5)

        mid_GRU_in = mid_in.permute(0, 2, 1, 3)
        mid_GRU_in = mid_GRU_in.reshape(mid_GRU_in.size()[0], mid_GRU_in.size()[1], -1)
        mid_GRU_out, _ = self.GRU_mid(mid_GRU_in)
        mid_GRU_out = mid_GRU_out.reshape(mid_GRU_in.size()[0], mid_GRU_in.size()[1], 8, -1)
        mid_GRU_out = mid_GRU_out.permute(0, 2, 1, 3)

        mid_out = self.conv_mid_2(mid_GRU_out)



        # ConvTrans for real
        res = torch.cat((mid_out, x5), 1)
        res1 = self.convT1(res)
        res1 = torch.cat((res1[:, :, :, :32], x4[:, :, :, :32]), 1)
        res2 = self.convT2(res1)
        res2 = torch.cat((res2[:, :, :, :64], x3[:, :, :, :64]), 1)
        res3 = self.convT3(res2)
        res3 = torch.cat((res3[:, :, :, :128], x2[:, :, :, :128]), 1)
        res4 = self.convT4(res3)

        res5 = res4.squeeze(1)
        res5, _ = self.GRU_real(res5)
        res5 = self.fcs(res5)
        res5 = res5.unsqueeze(1)

        # ConvTrans for imag
        phares1 = self.phaconvT1(res)
        phares1 = torch.cat((phares1[:, :, :, :32], x4[:, :, :, :32]), 1)
        phares2 = self.phaconvT2(phares1)
        phares2 = torch.cat((phares2[:, :, :, :64], x3[:, :, :, :64]), 1)
        phares3 = self.phaconvT3(phares2)
        phares3 = torch.cat((phares3[:, :, :, :128], x2[:, :, :, :128]), 1)
        phares4 = self.phaconvT4(phares3)
        phares4 = torch.cat((phares4[:, :, :, :256], x1[:, :, :, :256]), 1)
        phares5 = self.phaconvT5(phares4)

        # phares5 = phares4.squeeze(1)
        #
        # phares5, _ = self.GRU_imag(phares5)
        # phares5 = self.phafcs(phares5)
        # phares5 = phares5.unsqueeze(1)

        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5


class FCNN_phafullconv(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(FCNN_phafullconv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn1 = GLayerNorm2d(in_channel=4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn2 = GLayerNorm2d(in_channel=4)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn3 = GLayerNorm2d(in_channel=8)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn4 = GLayerNorm2d(in_channel=8)
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.bn5 = GLayerNorm2d(in_channel=16)

        self.conv_mid = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1)

        # Decoder for real

        self.convT1 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.bnT1 = GLayerNorm2d(in_channel=8)
        self.convT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.bnT2 = GLayerNorm2d(in_channel=8)
        self.convT3 = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
        self.bnT3 = GLayerNorm2d(in_channel=4)
        self.convT4 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        self.bnT4 = GLayerNorm2d(in_channel=1)
        # self.convT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        self.GRU_real = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)

        self.fcs = nn.Sequential(
            nn.Linear(257, 513),
        )
        # Decoder for imag

        self.phaconvT1 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT1 = GLayerNorm2d(in_channel=8)
        self.phaconvT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT2 = GLayerNorm2d(in_channel=8)
        self.phaconvT3 = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT3 = GLayerNorm2d(in_channel=4)
        self.phaconvT4 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT4 = GLayerNorm2d(in_channel=4)
        self.phaconvT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT5 = GLayerNorm2d(in_channel=1)
        # self.GRU_imag = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)
        # self.phafcs = nn.Sequential(
        #     nn.Linear(257, 513),
        # )

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def forward(self, x):
        # conv
        # (B, in_c, T, F)
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x5 = F.relu(self.bn5(self.conv5(x4)))

        output = F.relu(self.conv_mid(x5))

        # ConvTrans for real
        res = torch.cat((output, x5), 1)
        res1 = F.relu(self.bnT1(self.convT1(res)))
        res1 = torch.cat((res1[:, :, :, :32], x4[:, :, :, :32]), 1)
        res2 = F.relu(self.bnT2(self.convT2(res1)))
        res2 = torch.cat((res2[:, :, :, :64], x3[:, :, :, :64]), 1)
        res3 = F.relu(self.bnT3(self.convT3(res2)))
        res3 = torch.cat((res3[:, :, :, :128], x2[:, :, :, :128]), 1)
        res4 = F.relu(self.bnT4(self.convT4(res3)))

        res5 = res4.squeeze(1)
        res5, _ = self.GRU_real(res5)
        res5 = self.fcs(res5)
        res5 = res5.unsqueeze(1)

        # ConvTrans for imag
        phares1 = self.phabnT1(self.phaconvT1(res))
        phares1 = torch.cat((phares1[:, :, :, :32], x4[:, :, :, :32]), 1)
        phares2 = self.phabnT2(self.phaconvT2(phares1))
        phares2 = torch.cat((phares2[:, :, :, :64], x3[:, :, :, :64]), 1)
        phares3 = self.phabnT3(self.phaconvT3(phares2))
        phares3 = torch.cat((phares3[:, :, :, :128], x2[:, :, :, :128]), 1)
        phares4 = self.phabnT4(self.phaconvT4(phares3))
        phares4 = torch.cat((phares4[:, :, :, :256], x1[:, :, :, :256]), 1)
        tmp = self.phaconvT5(phares4)
        phares5 = self.phabnT5(self.phaconvT5(phares4))

        # phares5 = phares4.squeeze(1)
        #
        # phares5, _ = self.GRU_imag(phares5)
        # phares5 = self.phafcs(phares5)
        # phares5 = phares5.unsqueeze(1)

        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5

class FCNN_slim_V2(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(FCNN_slim_V2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn1 = GLayerNorm2d(in_channel=4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn2 = GLayerNorm2d(in_channel=4)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn3 = GLayerNorm2d(in_channel=8)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn4 = GLayerNorm2d(in_channel=8)
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.bn5 = GLayerNorm2d(in_channel=16)

        self.conv_mid = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1)

        # Decoder for real

        self.convT1 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.bnT1 = GLayerNorm2d(in_channel=8)
        self.convT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.bnT2 = GLayerNorm2d(in_channel=8)
        self.convT3 = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
        self.bnT3 = GLayerNorm2d(in_channel=4)
        self.convT4 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        self.bnT4 = GLayerNorm2d(in_channel=1)
        # self.convT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        self.GRU_real = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)

        self.fcs = nn.Sequential(
            nn.Linear(257, 513),
        )
        # Decoder for imag

        self.phaconvT1 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT1 = GLayerNorm2d(in_channel=8)
        self.phaconvT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT2 = GLayerNorm2d(in_channel=8)
        self.phaconvT3 = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT3 = GLayerNorm2d(in_channel=4)
        self.phaconvT4 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        self.phabnT4 = GLayerNorm2d(in_channel=1)
        # self.phaconvT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        self.GRU_imag = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)
        self.phafcs = nn.Sequential(
            nn.Linear(257, 513),
        )

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def forward(self, x):
        # conv
        # (B, in_c, T, F)
        tmp = self.conv1(x)
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x5 = F.relu(self.bn5(self.conv5(x4)))

        output = F.relu(self.conv_mid(x5))

        # ConvTrans for real
        res = torch.cat((output, x5), 1)
        res1 = F.relu(self.bnT1(self.convT1(res)))



        res1 = torch.cat((res1[:, :, :, :32], x4[:, :, :, :32]), 1)
        res2 = F.relu(self.bnT2(self.convT2(res1)))
        res2 = torch.cat((res2[:, :, :, :64], x3[:, :, :, :64]), 1)
        res3 = F.relu(self.bnT3(self.convT3(res2)))
        res3 = torch.cat((res3[:, :, :, :128], x2[:, :, :, :128]), 1)
        res4 = F.relu(self.bnT4(self.convT4(res3)))

        res5 = res4.squeeze(1)
        res5, _ = self.GRU_real(res5)
        res5 = self.fcs(res5)
        res5 = res5.unsqueeze(1)

        # ConvTrans for imag
        phares1 = F.relu(self.phabnT1(self.phaconvT1(res)))
        phares1 = torch.cat((phares1[:, :, :, :32], x4[:, :, :, :32]), 1)
        phares2 = F.relu(self.phabnT2(self.phaconvT2(phares1)))
        phares2 = torch.cat((phares2[:, :, :, :64], x3[:, :, :, :64]), 1)
        phares3 = F.relu(self.phabnT3(self.phaconvT3(phares2)))
        phares3 = torch.cat((phares3[:, :, :, :128], x2[:, :, :, :128]), 1)
        phares4 = F.relu(self.phabnT4(self.phaconvT4(phares3)))

        phares5 = phares4.squeeze(1)

        phares5, _ = self.GRU_imag(phares5)
        phares5 = self.phafcs(phares5)
        phares5 = phares5.unsqueeze(1)

        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5


class FCNN_slim(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(FCNN_slim, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2))

        self.conv_mid = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1)

        # Decoder for real

        self.convT1 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.convT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.convT3 = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
        self.convT4 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        # self.convT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        self.GRU_real = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)

        self.fcs = nn.Sequential(
            nn.Linear(257, 513),
        )
        # Decoder for imag

        self.phaconvT1 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.phaconvT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.phaconvT3 = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(1, 3), stride=(1, 2))
        self.phaconvT4 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        # self.phaconvT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        self.GRU_imag = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)
        self.phafcs = nn.Sequential(
            nn.Linear(257, 513),
        )


    def forward(self, x):
        # conv
        # (B, in_c, T, F)

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))

        output = F.relu(self.conv_mid(x5))

        # ConvTrans for real
        res = torch.cat((output, x5), 1)
        res1 = F.relu(self.convT1(res))
        res1 = torch.cat((res1[:, :, :, :32], x4[:, :, :, :32]), 1)
        res2 = F.relu(self.convT2(res1))
        res2 = torch.cat((res2[:, :, :, :64], x3[:, :, :, :64]), 1)
        res3 = F.relu(self.convT3(res2))
        res3 = torch.cat((res3[:, :, :, :128], x2[:, :, :, :128]), 1)
        res4 = F.relu(self.convT4(res3))

        res5 = res4.squeeze(1)
        res5, _ = self.GRU_real(res5)
        res5 = self.fcs(res5)
        res5 = res5.unsqueeze(1)

        # ConvTrans for imag
        phares1 = F.relu(self.phaconvT1(res))
        phares1 = torch.cat((phares1[:, :, :, :32], x4[:, :, :, :32]), 1)
        phares2 = F.relu(self.phaconvT2(phares1))
        phares2 = torch.cat((phares2[:, :, :, :64], x3[:, :, :, :64]), 1)
        phares3 = F.relu(self.phaconvT3(phares2))
        phares3 = torch.cat((phares3[:, :, :, :128], x2[:, :, :, :128]), 1)
        phares4 = F.relu(self.phaconvT4(phares3))

        phares5 = phares4.squeeze(1)

        phares5, _ = self.GRU_imag(phares5)
        phares5 = self.phafcs(phares5)
        phares5 = phares5.unsqueeze(1)

        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5
        # return real_comu, imag_comu

class FCNN_slim_V3(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(FCNN_slim_V3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 2))

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1)

        # Decoder for real

        self.convT1 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.convT2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.convT3 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.convT4 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        # self.convT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        self.GRU_real = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)

        self.fcs = nn.Sequential(
            nn.Linear(257, 513),
        )
        # Decoder for imag

        self.phaconvT1 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.phaconvT2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.phaconvT3 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.phaconvT4 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        # self.phaconvT5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        self.GRU_imag = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)
        self.phafcs = nn.Sequential(
            nn.Linear(257, 513),
        )


    def forward(self, x):
        # conv
        # (B, in_c, T, F)

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))

        output = F.relu(self.conv_mid(x5))

        # ConvTrans for real
        res = torch.cat((output, x5), 1)
        res1 = F.relu(self.convT1(res))
        res1 = torch.cat((res1[:, :, :, :32], x4[:, :, :, :32]), 1)
        res2 = F.relu(self.convT2(res1))
        res2 = torch.cat((res2[:, :, :, :64], x3[:, :, :, :64]), 1)
        res3 = F.relu(self.convT3(res2))
        res3 = torch.cat((res3[:, :, :, :128], x2[:, :, :, :128]), 1)
        res4 = F.relu(self.convT4(res3))

        res5 = res4.squeeze(1)
        res5, _ = self.GRU_real(res5)
        res5 = self.fcs(res5)
        res5 = res5.unsqueeze(1)

        # ConvTrans for imag
        phares1 = F.relu(self.phaconvT1(res))
        phares1 = torch.cat((phares1[:, :, :, :32], x4[:, :, :, :32]), 1)
        phares2 = F.relu(self.phaconvT2(phares1))
        phares2 = torch.cat((phares2[:, :, :, :64], x3[:, :, :, :64]), 1)
        phares3 = F.relu(self.phaconvT3(phares2))
        phares3 = torch.cat((phares3[:, :, :, :128], x2[:, :, :, :128]), 1)
        phares4 = F.relu(self.phaconvT4(phares3))

        phares5 = phares4.squeeze(1)

        phares5, _ = self.GRU_imag(phares5)
        phares5 = self.phafcs(phares5)
        phares5 = phares5.unsqueeze(1)

        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5

class FCNN_mid_GRU_new(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(FCNN_mid_GRU_new, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=16),
            nn.PReLU(),

        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=16),
            nn.PReLU(),

        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=32),
            nn.PReLU(),

        )

        self.GRU_mid = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True)


        # Decoder for real

        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=16),
            nn.PReLU(),

        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=16),
            nn.PReLU(),

        )
        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),

        )
        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),

        )

        self.convT5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=1),
            nn.PReLU(),
        )


        self.GRU_real = nn.GRU(input_size=129, hidden_size=129, num_layers=1, batch_first=True)

        # self.fcs = nn.Sequential(
        #     nn.Linear(257, 513),
        # )
        # Decoder for imag


        self.phaconvT1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=16),
            nn.PReLU(),
        )
        self.phaconvT2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=16),
            nn.PReLU(),
        )
        self.phaconvT3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),
        )
        self.phaconvT4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.PReLU(),
        )
        self.phaconvT5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=1),
            nn.PReLU(),
        )

        self.GRU_imag = nn.GRU(input_size=129, hidden_size=129, num_layers=1, batch_first=True)
        # self.GRU_imag = nn.GRU(input_size=257, hidden_size=257, num_layers=1, batch_first=True)
        # self.phafcs = nn.Sequential(
        #     nn.Linear(257, 513),
        # )

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def forward(self, x):
        # conv
        # (B, in_c, T, F)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        mid_in = x5

        mid_GRU_in = mid_in.permute(0, 2, 1, 3)
        mid_GRU_in = mid_GRU_in.reshape(mid_GRU_in.size()[0], mid_GRU_in.size()[1], -1)
        mid_GRU_out, _ = self.GRU_mid(mid_GRU_in)
        mid_GRU_out = mid_GRU_out.reshape(mid_GRU_in.size()[0], mid_GRU_in.size()[1], 32, -1)
        mid_GRU_out = mid_GRU_out.permute(0, 2, 1, 3)

        mid_out = mid_GRU_out

        # ConvTrans for real
        res = torch.cat((mid_out, x5), 1)
        res1 = self.convT1(res)
        res1 = torch.cat((res1[:, :, :, :8], x4[:, :, :, :8]), 1)
        res2 = self.convT2(res1)
        res2 = torch.cat((res2[:, :, :, :16], x3[:, :, :, :16]), 1)
        res3 = self.convT3(res2)
        res3 = torch.cat((res3[:, :, :, :32], x2[:, :, :, :32]), 1)
        res4 = self.convT4(res3)
        res4 = torch.cat((res4[:, :, :, :64], x1[:, :, :, :64]), 1)
        res5 = self.convT5(res4)

        res5 = res5.squeeze(1)
        res5, _ = self.GRU_real(res5)
        # # res5 = self.fcs(res5)
        res5 = res5.unsqueeze(1)

        # ConvTrans for imag
        phares1 = self.phaconvT1(res)
        phares1 = torch.cat((phares1[:, :, :, :8], x4[:, :, :, :8]), 1)
        phares2 = self.phaconvT2(phares1)
        phares2 = torch.cat((phares2[:, :, :, :16], x3[:, :, :, :16]), 1)
        phares3 = self.phaconvT3(phares2)
        phares3 = torch.cat((phares3[:, :, :, :32], x2[:, :, :, :32]), 1)
        phares4 = self.phaconvT4(phares3)
        phares4 = torch.cat((phares4[:, :, :, :64], x1[:, :, :, :64]), 1)
        phares5 = self.phaconvT5(phares4)

        phares5 = phares5.squeeze(1)
        phares5, _ = self.GRU_imag(phares5)
        # # phares5 = self.phafcs(phares5)
        phares5 = phares5.unsqueeze(1)

        # enh_spec = torch.cat((res5, phares5), 1)
        return res5, phares5

class CRN_for_IRM(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(CRN_for_IRM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=8),
            nn.ReLU(),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=8),
            nn.ReLU(),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=16),
            nn.ReLU(),

        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            GLayerNorm2d(in_channel=16),
            nn.ReLU(),

        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=32),
            nn.ReLU(),

        )

        self.GRU_mid = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True)


        # Decoder for real

        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=16),
            nn.ReLU(),

        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=16),
            nn.ReLU(),

        )
        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.ReLU(),

        )
        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=8),
            nn.ReLU(),

        )

        self.convT5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(1, 3), stride=(1, 2)),
            GLayerNorm2d(in_channel=1),
            nn.ReLU(),
        )


        self.GRU_out = nn.GRU(input_size=129, hidden_size=129, num_layers=1, batch_first=True)

        self.fcs = nn.Sequential(
            nn.Linear(129, 129),
            nn.Sigmoid(),
        )

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def forward(self, x):
        # conv
        # (B, in_c, T, F)
        x = x.unsqueeze(1)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        mid_in = x5

        mid_GRU_in = mid_in.permute(0, 2, 1, 3)
        mid_GRU_in = mid_GRU_in.reshape(mid_GRU_in.size()[0], mid_GRU_in.size()[1], -1)
        mid_GRU_out, _ = self.GRU_mid(mid_GRU_in)
        mid_GRU_out = mid_GRU_out.reshape(mid_GRU_in.size()[0], mid_GRU_in.size()[1], 32, -1)
        mid_GRU_out = mid_GRU_out.permute(0, 2, 1, 3)

        mid_out = mid_GRU_out

        # ConvTrans for real
        res = torch.cat((mid_out, x5), 1)
        res1 = self.convT1(res)
        res1 = torch.cat((res1[:, :, :, :8], x4[:, :, :, :8]), 1)
        res2 = self.convT2(res1)
        res2 = torch.cat((res2[:, :, :, :16], x3[:, :, :, :16]), 1)
        res3 = self.convT3(res2)
        res3 = torch.cat((res3[:, :, :, :32], x2[:, :, :, :32]), 1)
        res4 = self.convT4(res3)
        res4 = torch.cat((res4[:, :, :, :64], x1[:, :, :, :64]), 1)
        res5 = self.convT5(res4)

        res5 = res5.squeeze(1)
        res5, _ = self.GRU_out(res5)
        res5 = self.fcs(res5)
        # res5 = res5.unsqueeze(1)



        return res5
