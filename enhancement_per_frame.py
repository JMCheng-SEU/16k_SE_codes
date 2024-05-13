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
# from test_for_PSD import *
# from model.Simple_DCCRN import *
# from model.DCCRN_origin import *
from model.oneC_DPRNN_Streamer import DPCRN_Model_Streamer, Simple_Streamer
from model.DPCRN_new_FB_Streamer import DPCRN_Model_new

def main(config, epoch):
    # root_dir = Path(config["experiments_dir"]) / config["name"]
    root_dir = Path(config["experiments_dir"])
    enhancement_dir = root_dir / "enhancements"
    checkpoints_dir = root_dir / "checkpoints"

    """============== 加载数据集 =============="""
    dataset = initialize_config(config["dataset"])
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
    )

    """============== 加载模型断点（"best"，"latest"，通过数字指定） =============="""
    # model = initialize_config(config["model"])
    model = DPCRN_Model_new()
    # device = torch.device("cuda:0")
    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    stft = STFT(
        filter_length=512,
        hop_length=256
    ).to('cpu')



    if epoch == "best":
        model_path = checkpoints_dir / "best_model.tar"
        model_checkpoint = torch.load(model_path.as_posix())
        model_static_dict = model_checkpoint["model"]
        checkpoint_epoch = model_checkpoint['epoch']
    elif epoch == "latest":
        model_path = checkpoints_dir / "latest_model.tar"
        model_checkpoint = torch.load(model_path.as_posix())
        model_static_dict = model_checkpoint["model"]
        checkpoint_epoch = model_checkpoint['epoch']
    else:
        model_path = checkpoints_dir / f"model_{str(epoch).zfill(4)}.pth"
        model_checkpoint = torch.load(model_path.as_posix())
        model_static_dict = model_checkpoint
        checkpoint_epoch = epoch

    print(f"Loading model checkpoint, epoch = {checkpoint_epoch}")
    model.load_state_dict(model_static_dict)
    model.to('cpu')
    model.eval()

    """============== 增强语音 =============="""
    if epoch == "best" or epoch == "latest":
        results_dir = enhancement_dir / f"{epoch}_checkpoint_{checkpoint_epoch}_epoch"
    else:
        results_dir = enhancement_dir / f"checkpoint_{epoch}_epoch"

    results_dir.mkdir(parents=True, exist_ok=True)

    for i, (mixture, _, _, names) in enumerate(dataloader):
        print(f"Enhance {i + 1}th speech")
        name = names[0]
        print(name)

        mixture = mixture.to('cpu')

        mixture_D = stft.transform(mixture)
        mixture_real = mixture_D[:, :, :, 0]
        mixture_imag = mixture_D[:, :, :, 1]

        spec_complex = torch.stack([mixture_real, mixture_imag], 1)

        ####### split frames
        mix_tuple = torch.split(spec_complex, 1, dim=2)  # 按照split_frame这个维度去分
        index_num = 0
        for item in mix_tuple:

            item = item.to('cpu')

            out_real, out_imag = model(item)


            if index_num == 0:
                est_real = out_real
                est_imag = out_imag
            else:
                est_real = torch.cat([est_real, out_real], dim=1)
                est_imag = torch.cat([est_imag, out_imag], dim=1)

            index_num = index_num + 1

        enhanced_D = torch.stack([est_real, est_imag], 3)

        enhanced_D = enhanced_D.detach().cpu()
        enhanced = stft.inverse(enhanced_D)

        if enhanced.shape[1] < mixture.shape[1]:
            pad = torch.zeros([enhanced.shape[0], mixture.shape[1] - enhanced.shape[1]], dtype=torch.float).to(
                'cpu')
            enhanced = torch.cat((enhanced, pad), dim=1)
        enhanced_audio = enhanced.squeeze().numpy()


        sf.write(f"{results_dir}/{name}.wav", enhanced_audio, 16000)



if __name__ == '__main__':
    # parser = argparse.ArgumentParser("Spectrogram mapping: Speech Enhancement")
    # parser.add_argument("-C", "--config", type=str, required=True,
    #                     help="Specify the configuration file for enhancement (*.json).")
    # parser.add_argument("-E", "--epoch", default="best",
    #                     help="Model checkpoint for speech enhancement, can be set to 'best', 'latest' and specific epoch. (default: 'best')")
    # args = parser.parse_args()
    config_path = "F:\\JMCheng\\codes\\DPCRN_16k\\config\\enhancement\\Non-Stationary-Datasets.json"
    config = json.load(open(config_path))
    config["name"] = os.path.splitext(os.path.basename(config_path))[0]
    main(config, "best")
    # main(config, 7)
