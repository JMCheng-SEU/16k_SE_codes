import librosa
import numpy as np
# from hyperparameters import Hyperparams as hp
import scipy.io
from scipy import signal
import scipy.io.wavfile as wav
from utils.stft import STFT

import torch
import torch.nn as nn
import torch.nn.functional as F

filename = 'F:\\real_time_codes\\andriod_codes\\NEON_TEST_1010\\InnoTalkDNNNS_4816_withRIR_1029\\test_wavs_104\\window256w512.txt'
new_window = np.zeros(512)
count = 0

with open(filename, 'r') as file_to_read:
    line = file_to_read.readline()
    for i in line.split(','):
        try:
            new_window[count] = float(i)
            count += 1
        except:
            flag = 0




wav_path ="F:\\real_time_codes\\andriod_codes\\NEON_TEST_1010\\InnoTalkDNNNS_4816_withRIR_1029\\test_wavs_104\\fileid_0.wav"
sr1, y1 = wav.read(wav_path)
y, sr = librosa.load(wav_path, sr=16000)
if sr != 16000:
    raise ValueError('Sampling rate is expected to be 16kHz!')
if y.dtype == 'int16':
    y = np.float32(y / 32767.)
elif y.dtype != 'float32':
    y = np.float32(y)


zeros = np.zeros(256)
audio = y[0:256]
audio = np.concatenate((zeros,audio),axis=0)
D = librosa.stft(audio,center=False, n_fft=512, hop_length=256, win_length=512, window=new_window)

stft_block = STFT(
            filter_length=512,
            hop_length=256
        )

x_torch = torch.from_numpy(np.array(audio, dtype=np.float32)).unsqueeze(0)
mixture_D = stft_block.transform(x_torch)


real_part = D.real
imag_part = D.imag

D_recons = real_part + 1j * imag_part
result = librosa.istft(D_recons,
                       center=False,
                       hop_length=256,
                       win_length=512,
                       window=scipy.signal.hamming)

utt_len = D.shape[-1]


enh_wav_path ="F:\\real_time_exp\\DSDATASET_16k\\p232_001_enh.wav"
wav.write(enh_wav_path, 16000, np.int16(result * 32767))
