import wave
import numpy as np
import soundfile
import librosa
import os
import torch
import torch as th
from torch import nn
from torch.nn import functional as F
import random


def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)

    return audio, fs

def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

speech_dir = "F:\\JMCheng\\DATASET\\VAD_DATASET_new\\final_vad_rewrite"

workspace = "F:\\JMCheng\\DATASET\\VAD_DATASET_new"

speech_names = []

noise_id = 1

cur_noise = []



for dirpath, dirnames, filenames in os.walk(speech_dir):
    for filename in filenames:
         if filename.lower().endswith(".wav"):
            # print(os.path.join(dirpath,filename))
            speech_names.append(os.path.join(dirpath, filename))

for speech_na in speech_names:
    # Read speech
    speech_na_basename = os.path.basename(speech_na)
    (speech_audio, fs) = read_audio(speech_na, target_fs=16000)

    noisy_audio = speech_audio[:, 0]
    audio_label = speech_audio[:, 1]

    for index in range(len(noisy_audio)):
        if audio_label[index] == 0:
            cur_noise.append(noisy_audio[index])
        if len(cur_noise) >= 16000 * 10:
            out_audio_path = os.path.join(workspace, "split_noises", "noise_%d.wav" % noise_id)
            create_folder(os.path.dirname(out_audio_path))
            write_audio(out_audio_path, np.array(cur_noise), fs)

            noise_id += 1
            cur_noise = []

            if noise_id % 1000 == 0:
                print(noise_id)





