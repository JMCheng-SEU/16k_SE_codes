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
    model = initialize_config(config["model"])
    # device = torch.device("cuda:0")
    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    stft = STFT(
        filter_length=320,
        hop_length=160
    ).to("cpu")

    if epoch == "best":
        model_path = checkpoints_dir / "best_model.tar"
        model_checkpoint = torch.load(model_path.as_posix(), map_location=device)
        model_static_dict = model_checkpoint["model"]
        checkpoint_epoch = model_checkpoint['epoch']
    elif epoch == "latest":
        model_path = checkpoints_dir / "latest_model.tar"
        model_checkpoint = torch.load(model_path.as_posix(), map_location=device)
        model_static_dict = model_checkpoint["model"]
        checkpoint_epoch = model_checkpoint['epoch']
    else:
        model_path = checkpoints_dir / f"model_{str(epoch).zfill(4)}.pth"
        model_checkpoint = torch.load(model_path.as_posix(), map_location=device)
        model_static_dict = model_checkpoint
        checkpoint_epoch = epoch

    print(f"Loading model checkpoint, epoch = {checkpoint_epoch}")
    model.load_state_dict(model_static_dict)
    model.to(device)
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

        # Mixture mag and Clean mag
        print("\tSTFT...")
        mixture_D = stft.transform(mixture)
        mixture_real = mixture_D[:, :, :, 0]
        mixture_imag = mixture_D[:, :, :, 1]
        mixture_mag = torch.sqrt(mixture_real ** 2 + mixture_imag ** 2)  # [1, T, F]
        mix_spec = mixture_D.permute(0, 3, 1, 2)
        print("\tEnhancement...")

        # mixture_mag_chunks = torch.split(mixture_mag, mixture_mag.size()[1] // 5, dim=1)
        # mixture_mag_chunks = mixture_mag_chunks[:-1]
        # enhanced_mag_chunks = []
        # for mixture_mag_chunk in tqdm(mixture_mag_chunks):
        #     mixture_mag_chunk = mixture_mag_chunk.to(device)
        #     enhanced_mag_chunks.append(model(mixture_mag_chunk).detach().cpu())  # [T, F]
        #
        #
        # enhanced_mag = torch.cat(enhanced_mag_chunks, dim=0).unsqueeze(0)  # [1, T, F]

        #####CRN
        # enhanced_mag = (model(mixture_mag).detach().cpu()).unsqueeze(0)
        # mixture_mag = mixture_mag.squeeze(0)
        # # enhanced_mag = enhanced_mag.unsqueeze(0)
        # # enhanced_mag = enhanced_mag.detach().cpu().data.numpy()
        # # mixture_mag = mixture_mag.cpu()
        #
        # enhanced_real = enhanced_mag * mixture_real[:, :enhanced_mag.size(1), :] / mixture_mag[:, :enhanced_mag.size(1),
        #                                                                            :]
        # enhanced_imag = enhanced_mag * mixture_imag[:, :enhanced_mag.size(1), :] / mixture_mag[:, :enhanced_mag.size(1),:]

        ####PHA_CRN
        # enhanced_spec = (model(mix_spec).detach().cpu()).unsqueeze(0)
        # enhanced_D = enhanced_spec.permute(0, 2, 3, 1)
        # enhanced_real = torch.squeeze(enhanced_real, 1)
        # enhanced_imag = torch.squeeze(enhanced_imag, 1)

        ####cIRM_CRN


        enhanced_real, enhanced_imag = model(mix_spec)
        enhanced_real = torch.squeeze(enhanced_real, 0)
        enhanced_imag = torch.squeeze(enhanced_imag, 0)

        est_Mr = -10. * torch.log((10 - enhanced_real) / ((10 + enhanced_real)) + 1e-8)
        est_Mi = -10. * torch.log((10 - enhanced_imag) / ((10 + enhanced_imag)) + 1e-8)

        recons_real = est_Mr * mixture_real - est_Mi * mixture_imag
        recons_imag = est_Mr * mixture_imag - est_Mi * mixture_real
        enhanced_D = torch.stack([recons_real, recons_imag], 3)

        #### avoid nan data
        zero = torch.zeros_like(enhanced_D)
        enhanced_D = torch.where(torch.isnan(enhanced_D), zero, enhanced_D)

        # enhanced_D = torch.stack([enhanced_real, enhanced_imag], 3)
        enhanced = stft.inverse(enhanced_D)

        enhanced = enhanced.detach().cpu().squeeze().numpy()

        sf.write(f"{results_dir}/{name}.wav", enhanced, 16000)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser("Spectrogram mapping: Speech Enhancement")
    # parser.add_argument("-C", "--config", type=str, required=True,
    #                     help="Specify the configuration file for enhancement (*.json).")
    # parser.add_argument("-E", "--epoch", default="best",
    #                     help="Model checkpoint for speech enhancement, can be set to 'best', 'latest' and specific epoch. (default: 'best')")
    # args = parser.parse_args()
    config_path = "E:\JMCheng\A-Convolutional-Recurrent-Neural-Network-for-Real-Time-Speech-Enhancement-master\config\enhancement\\Non-Stationary-Datasets.json"
    config = json.load(open(config_path))
    config["name"] = os.path.splitext(os.path.basename(config_path))[0]
    main(config, "best")
