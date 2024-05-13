from pyvad import vad, trim
from librosa import load
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import soundfile
import librosa
import os

def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs

def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


speech_dir = "F:\\JMCheng\\DATASET\\100h_16k_DNS_noreverb_-5-15\\clean"
noisy_dir = "F:\\JMCheng\\DATASET\\100h_16k_DNS_noreverb_-5-15\\noisy"
cnt = 0
workspace = "F:\\JMCheng\\DATASET\\100h_16k_DNS_noreverb_-5-15"

speech_names = []
for dirpath, dirnames, filenames in os.walk(speech_dir):
    for filename in filenames:
         if filename.lower().endswith(".wav"):
            # print(os.path.join(dirpath,filename))
            speech_names.append(os.path.join(dirpath, filename))

for speech_na in speech_names:
    # Read speech
    speech_na_basename = os.path.basename(speech_na)
    speech_fpart = os.path.splitext(speech_na_basename)[0]
    index = speech_fpart.find(".", 0)  # 找到点号的位置
    if index == -1:
        (speech_audio, fs) = read_audio(speech_na, target_fs=16000)

        noisy_audio_path = os.path.join(noisy_dir, "%s" % speech_na_basename)

        (noisy_audio, fs1) = read_audio(noisy_audio_path, target_fs=16000)

        speech_na_basename = os.path.basename(speech_na)

        vact = vad(speech_audio, fs, fs_vad=16000, hoplength=30, vad_mode=3) * 0.5

        labeled_speech = np.stack((noisy_audio, vact), axis=1)

        out_speech_audio_path = os.path.join(workspace, "noisy_vadnew", "%s" % speech_na_basename)
        create_folder(os.path.dirname(out_speech_audio_path))
        write_audio(out_speech_audio_path, labeled_speech, fs)

        cnt += 1

        if cnt % 1000 == 0:
            print(cnt)
