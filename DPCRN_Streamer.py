import torch
import numpy as np

from model.oneC_DPRNN_Streamer import DPCRN_Model_Streamer, Simple_Streamer

model1 = DPCRN_Model_Streamer()
model2 = DPCRN_Model_Streamer()

path = "F:\\JMCheng\\real_time_exp\\16k_exp_new\\DPRNN_cmp_newwindow_PMSQE_PFPL_APCSNR\\checkpoints\\model_0035.pth"

# torch.save(model.state_dict(), path)
state_dict1 = torch.load(path)
state_dict2 = torch.load(path)

inputs = torch.ones([1, 2, 100, 257])
# mixture_mag = torch.sqrt(inputs[:, 0, :, :] ** 2 + inputs[:, 1, :, :] ** 2 + 1e-8)
# LPS_fea = torch.log10(mixture_mag ** 2)
# print(LPS_fea)
model1.load_state_dict(state_dict1)
model1.eval()

model2.load_state_dict(state_dict2)
model2.eval()

out1, _ = model1(inputs)
# out1 = model(inputs)

####### split frames
mix_tuple = torch.split(inputs, 1, dim=2)  # 按照split_frame这个维度去分
index_num = 0
for item in mix_tuple:

    out2, _ = model2(item)

    # out2 = model(item)

    if index_num == 0:
        out2_est = out2
    else:
        out2_est = torch.cat([out2_est, out2], dim=1)

    index_num = index_num + 1

print(((out1 - out2_est)** 2).mean())

