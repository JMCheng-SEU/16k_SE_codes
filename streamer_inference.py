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


model = DPCRN_Model_Streamer()

stft = STFT(
    filter_length=512,
    hop_length=256
).to('cuda')

path = "F:\\JMCheng\\real_time_exp\\16k_exp_new\\PHA-CRN\\checkpoints\\model_0020.pth"

model_static_dict = torch.load(path)

model.load_state_dict(model_static_dict)
model.to('cuda')
model.eval()


