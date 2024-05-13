import torch
import numpy as np
import os

from model.CGRNN_256 import CGRNN_FB_256

model = CGRNN_FB_256()
path = "D:\\JMCheng\\model_0100.pth"
# torch.save(model.state_dict(), path)

state_dict = torch.load(path)

# for i in state_dict:
#     param = state_dict[i].numpy()
#     if i.split('.')[0][0:4] == 'conv':
#         param = param.flatten()
#         name = i.split('.')[0] + '_' + i.split('.')[-2] + '_' + i.split('.')[-1]
#         np.savetxt('F:\\JMCheng\\real_time_exp\\16k_DNS_64_exp\\PHA-CRN\\epoch021\\params\\{}.txt'.format(name), param)
#
#     # print(i + ': ' + np.max(param))
#     # if i.split('.')[-2] == '0':
#     #     # print(i)
#     #     param = param.T
#     # if i.split('.')[0][-2] == 'T':
#     #     print(i)
#     # if i.split('.')[1] == '1' or i.split('.')[1] == '0':
#     #     param = param.flatten()
#     #     name = i.split('.')[0] + '_' + i.split('.')[-1]
#     #     np.savetxt('E:\\JMCheng\\Torch2VS_model\\Complex_CRN_BN\\params\\{}.txt'.format(name), param)
#
# for key, value in state_dict.items():
#     value = value.numpy()
#     # value = value.T
#     save_space = "F:\\JMCheng\\real_time_exp\\16k_DNS_64_exp\\PHA-CRN\\epoch021\\GRU_params"
#     key_list = key.split('.')
#     key_basename = key.split('.')[0]
#     # if key_list[0].split('_')[0] == "dense":
#     #     value = value.T
#     if key_basename.split('_')[0] == 'GGRU':
#         value = value.T
#         key_new = key_list[0] + '_' + key_list[1] + '_' + key_list[2] + '_' + key_list[3]
#         para_name = key_new + '.txt'
#         save_path = os.path.join(save_space, para_name)
#         np.savetxt(save_path, value)
#     elif key_basename.split('_')[0] == 'fc':
#         value = value.T
#         key_new = key_list[0] + '_' + key_list[1] + '_' + key_list[2] + '_' + key_list[3]
#         para_name = key_new + '.txt'
#         save_path = os.path.join(save_space, para_name)
#         np.savetxt(save_path, value)

inputs = torch.ones([1, 2, 100, 129])
model.load_state_dict(state_dict)
model.eval()
with torch.no_grad():
    out_real, out_imag= model(inputs)

print(out_real.shape)
