import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """
    def __init__(self):
        super(CRNN, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0))
        self.bn5 = nn.BatchNorm2d(num_features=64)


        # LSTM
        self.LSTM1 = nn.LSTM(input_size=512, hidden_size=512, num_layers=2, batch_first=True)


        # Decoder
        self.convT1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.bnT1 = nn.BatchNorm2d(num_features=64)
        self.convT2 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.bnT2 = nn.BatchNorm2d(num_features=32)
        self.convT3 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.bnT3 = nn.BatchNorm2d(num_features=16)
        self.convT4 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2))
        self.bnT4 = nn.BatchNorm2d(num_features=8)
        self.convT5 = nn.ConvTranspose2d(in_channels=16, out_channels=2, kernel_size=(1, 3), stride=(1, 2))


    def forward(self, x):
        # conv
        # (B, in_c, T, F)

        x1 = F.elu(self.bn1(self.conv1(x)))
        x2 = F.elu(self.bn2(self.conv2(x1)))
        x3 = F.elu(self.bn3(self.conv3(x2)))
        x4 = F.elu(self.bn4(self.conv4(x3)))
        x5 = F.elu(self.bn5(self.conv5(x4)))

        # reshape
        out5 = x5.permute(0, 2, 1, 3)
        out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
        # lstm

        lstm, _ = self.LSTM1(out5)
        # reshape
        output = lstm.reshape(lstm.size()[0], lstm.size()[1], 64, -1)
        output = output.permute(0, 2, 1, 3)

        # ConvTrans
        res = torch.cat((output, x5), 1)
        res1 = F.elu(self.bnT1(self.convT1(res)))
        res1 = torch.cat((res1[:, :, :, :-1], x4[:, :, :, :-1]), 1)
        res2 = F.elu(self.bnT2(self.convT2(res1)))
        res2 = torch.cat((res2[:, :, :, :-1], x3[:, :, :, :-1]), 1)
        res3 = F.elu(self.bnT3(self.convT3(res2)))
        res3 = torch.cat((res3[:, :, :, :-1], x2[:, :, :, :-1]), 1)
        res4 = F.elu(self.bnT4(self.convT4(res3)))
        res4 = torch.cat((res4[:, :, :, :-1], x1[:, :, :, :-1]), 1)
        # (B, o_c, T. F)
        res5 = self.convT5(res4)

        enh_real = res5[:, 0, :]
        enh_imag = res5[:, 1, :]
        return enh_real, enh_imag

if __name__ == '__main__':

    inputs = torch.rand(1, 2, 100, 257)

    model = CRNN()

    enh_real, enh_imag = model(inputs)

    print(enh_real.shape)