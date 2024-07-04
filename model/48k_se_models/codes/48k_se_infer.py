import json
import os
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.stft import STFT
from utils.utils import initialize_config

from model.DB_AIAT_model.aia_trans import aia_complex_trans_ri
from model.AIA_DPCRN.AIA_CRN import AIA_CRN_SIG_oneL, DF_DIL_AIA_DCRN_merge_new, DIL_AIA_DCRN_merge_new

device = torch.device("cuda")
se_model = DF_DIL_AIA_DCRN_merge_new()
# model_static_dict = torch.load("D:\\JMCheng\\SIG_Challenge\\codes\\SIG_MGAN_SIG_48k\\model\\sig_final\\pre_trained\\model_G_0005.pth")
model_static_dict = torch.load("D:\\JMCheng\\SIG_Challenge\\SIG_exp\\100h_DF_DIL_AIA_DCRN_merge_new_MRSTFT_PMSQE\\checkpoints\\model_G_0010.pth")
se_model.load_state_dict(model_static_dict)
se_model.to(device)
se_model.eval()


stft = STFT(
    filter_length=960,
    hop_length=480
).to("cuda")

speech_dir = "D:\\JMCheng\\SIG_Challenge\\DATASET\\testset_new\\nearend"

out_dir = "D:\\JMCheng\\SIG_Challenge\\codes\\SIG_MGAN_SIG_48k\\model\\sig_final\\enhanced_wavs_new"

speech_names = []
for dirpath, dirnames, filenames in os.walk(speech_dir):
    for filename in filenames:
         if filename.lower().endswith(".wav"):
            # print(os.path.join(dirpath,filename))
            speech_names.append(os.path.join(dirpath, filename))

for speech_na in speech_names:
    # Read speech
    speech_na_basename = os.path.basename(speech_na)

    print(speech_na_basename)

    mixture, fs = sf.read(speech_na)

    mixture = torch.from_numpy(mixture).float()
    mixture = mixture.unsqueeze(0).to("cuda")

    mixture_D = stft.transform(mixture)
    mixture_real = mixture_D[:, :, :, 0]
    mixture_imag = mixture_D[:, :, :, 1]
    spec_complex = torch.stack([mixture_real, mixture_imag], 1)
    with torch.no_grad():
        est_real, est_imag = se_model(spec_complex)

    enhanced_D = torch.stack([est_real, est_imag], 3)
    est_wav = stft.inverse(enhanced_D).squeeze()

    est_wav = est_wav.detach().cpu().numpy()

    del mixture, mixture_D, mixture_real, mixture_imag, spec_complex, est_real, est_imag, enhanced_D
    torch.cuda.empty_cache()

    out_path = os.path.join(out_dir, speech_na_basename)
    sf.write(out_path, est_wav, 48000)

