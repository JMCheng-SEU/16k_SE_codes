import argparse
import json
import os
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.stft import STFT
from utils.utils import initialize_config
import numpy as np

from model.oneC_DIL_CRN import DPCRN_Model
from model.CGRNN_FB_64 import CGRNN_FB_64_Lite_extra_spec_oneG

def main(config, epoch):
    # root_dir = Path(config["experiments_dir"]) / config["name"]
    root_dir = Path(config["experiments_dir"])
    enhancement_dir = root_dir / "enhancements"
    checkpoints_dir = root_dir / "checkpoints"


    """============== 加载模型断点（"best"，"latest"，通过数字指定） =============="""
    # model = initialize_config(config["model"])
    model = CGRNN_FB_64_Lite_extra_spec_oneG()

    stft = STFT(
        filter_length=128,
        hop_length=64
    ).to('cuda')

    path = "F:\\JMCheng\\real_time_exp\\16k_DNS_64_exp\\PHA-CRN\\checkpoints\\model_0050.pth"

    state_dict = torch.load(path)

    model.load_state_dict(state_dict)
    model.to('cuda')
    model.eval()


    # if epoch == "best":
    #     model_path = checkpoints_dir / "best_model.tar"
    #     model_checkpoint = torch.load(model_path.as_posix())
    #     model_static_dict = model_checkpoint["model"]
    #     checkpoint_epoch = model_checkpoint['epoch']
    # elif epoch == "latest":
    #     model_path = checkpoints_dir / "latest_model.tar"
    #     model_checkpoint = torch.load(model_path.as_posix())
    #     model_static_dict = model_checkpoint["model"]
    #     checkpoint_epoch = model_checkpoint['epoch']
    # else:
    #     model_path = checkpoints_dir / f"model_{str(epoch).zfill(4)}.pth"
    #     model_checkpoint = torch.load(model_path.as_posix())
    #     model_static_dict = model_checkpoint
    #     checkpoint_epoch = epoch
    #
    # print(f"Loading model checkpoint, epoch = {checkpoint_epoch}")
    # model.load_state_dict(model_static_dict)
    # model.to('cuda')
    # model.eval()

    """============== 增强语音 =============="""
    if epoch == "best" or epoch == "latest":
        results_dir = enhancement_dir / f"checkpoint_{epoch}_epoch"
    else:
        results_dir = enhancement_dir / f"checkpoint_{epoch}_epoch"

    results_dir.mkdir(parents=True, exist_ok=True)

    ################## visualize real recording wavs
    real_record_path = "F:\\JMCheng\\DATASET\\100h_16k_DNS_noreverb_-5-15\\test_mix_new"
    speech_names = []
    for dirpath, dirnames, filenames in os.walk(real_record_path):
        for filename in filenames:
            if filename.lower().endswith(".wav"):
                speech_names.append(os.path.join(dirpath, filename))

    for speech_na in speech_names:

        name = os.path.basename(speech_na)
        print(name)
        mixture, sr = sf.read(speech_na, dtype="float32")
        real_len = len(mixture)
        assert sr == 16000
        mixture = torch.from_numpy(mixture).to('cuda')
        mixture = mixture.unsqueeze(0)


        mixture_D = stft.transform(mixture)
        mixture_real = mixture_D[:, :, :, 0]
        mixture_imag = mixture_D[:, :, :, 1]

        spec_complex = torch.stack([mixture_real, mixture_imag], 1)



        est_real, est_imag = model(spec_complex)

        enhanced_D = torch.stack([est_real, est_imag], 3)
        enhanced = stft.inverse(enhanced_D)

        enhanced_audio = enhanced.detach().cpu().squeeze().numpy()


        sf.write(f"{results_dir}/{name}", enhanced_audio, 16000)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser("Spectrogram mapping: Speech Enhancement")
    # parser.add_argument("-C", "--config", type=str, required=True,
    #                     help="Specify the configuration file for enhancement (*.json).")
    # parser.add_argument("-E", "--epoch", default="best",
    #                     help="Model checkpoint for speech enhancement, can be set to 'best', 'latest' and specific epoch. (default: 'best')")
    # args = parser.parse_args()
    config_path = "D:\\JMCheng\\RT_16k\\codes\\CGRNN_FB_512_16k\\config\\enhancement\\Non-Stationary-Datasets.json"
    config = json.load(open(config_path))
    config["name"] = os.path.splitext(os.path.basename(config_path))[0]
    main(config, "best")
    # main(config, 7)
