import torch
import numpy as np
from model.pha_cnn_256_160 import CRN_for_IRM
import os


model = CRN_for_IRM()
state_dict = torch.load('E:\\real_time_exp\\CRN_16k_DNS_2\\PHA-CRN\\checkpoints\\model_0007.pth')
print(len(state_dict))
for i in state_dict:
    param = state_dict[i].numpy()
    # print(i + ': ' + np.max(param))
    # if i.split('.')[-2] == '0':
    #     # print(i)
    #     param = param.T
    # if i.split('.')[0][-2] == 'T':
    #     print(i)
    if i.split('.')[1] == '1' or i.split('.')[1] == '0':
        param = param.flatten()
        name = i.split('.')[0] + '_' + i.split('.')[-1]
        np.savetxt('E:\\real_time_exp\\CRN_16k_DNS_2\\PHA-CRN\\checkpoints\\parameters\\{}.txt'.format(name), param)

# for key, value in state_dict.items():
#     value = value.numpy()
#     # value = value.T
#     save_space = "E:\\GRUCNN_gln_0908"
#     key_list = key.split('.')
#     # if key_list[0].split('_')[0] == "dense":
#     #     value = value.T
#     if key.split('_')[0] == 'GRU':
#         value = value.T
#         key_new = key_list[0] + '_' + key_list[1]
#         para_name = key_new + '.txt'
#         save_path = os.path.join(save_space, para_name)
#         np.savetxt(save_path, value)
#     elif key.split('.')[-2] == '0':
#         value = value.T
#         key_new = key_list[0] + '_' + key_list[2]
#         para_name = key_new + '.txt'
#         save_path = os.path.join(save_space, para_name)
#         np.savetxt(save_path, value)


for key, value in state_dict.items():
    value = value.numpy()
    # value = value.T
    save_space = "E:\\real_time_exp\\CRN_16k_DNS_2\\PHA-CRN\\checkpoints\\parameters"
    key_list = key.split('.')
    # if key_list[0].split('_')[0] == "dense":
    #     value = value.T
    if key.split('_')[0] == 'GRU':
        value = value.T
        key_new = key_list[0] + '_' + key_list[1]
        para_name = key_new + '.txt'
        save_path = os.path.join(save_space, para_name)
        np.savetxt(save_path, value)
    elif key.split('.')[0] == 'fcs':
        value = value.T
        key_new = key_list[0] + '_' + key_list[2]
        para_name = key_new + '.txt'
        save_path = os.path.join(save_space, para_name)
        np.savetxt(save_path, value)
    # elif key.split('.')[1] == '0':
    #     key_new = key_list[0] + '_' + key_list[2]
    #     para_name = key_new + '.txt'
    #     save_path = os.path.join(save_space, para_name)
    #     np.savetxt(save_path, value)
    # elif key.split('.')[1] == '1':
    #     key_new = key_list[0] + '_' + key_list[2]
    #     para_name = key_new + '.txt'
    #     save_path = os.path.join(save_space, para_name)
    #     np.savetxt(save_path, value)
    elif key.split('.')[1] == '2':
        key_new = key_list[0] + '_' + 'alpha'
        para_name = key_new + '.txt'
        save_path = os.path.join(save_space, para_name)
        np.savetxt(save_path, value)

inputs = torch.ones([1, 2, 1, 129])
mixture_mag = torch.sqrt(inputs[:, 0, :, :] ** 2 + inputs[:, 1, :, :] ** 2 + 1e-8)
LPS_fea = torch.log10(mixture_mag ** 2)
print(LPS_fea)
model.load_state_dict(state_dict)
model.eval()
with torch.no_grad():
    out = model(LPS_fea)
enhanced_mag = out * mixture_mag
enhanced_real = enhanced_mag * inputs[:, 0, :, :] / mixture_mag
enhanced_imag = enhanced_mag * inputs[:, 1, :, :] / mixture_mag
print(out)
