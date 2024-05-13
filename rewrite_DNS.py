import wave
import matplotlib.pyplot as plt
import numpy as np
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



speech_dir = "E:\\JMCheng\\DNS_NEW\\DNS_Challenge\\dataset\\noise"

workspace = "D:\\JMCheng\\RT_16k\\DATASET\\MC_DATASET\\DNS_DATASET"

speech_names = []
cnt = 1
for dirpath, dirnames, filenames in os.walk(speech_dir):
    for filename in filenames:
         if filename.lower().endswith(".wav"):
            # print(os.path.join(dirpath,filename))
            speech_names.append(os.path.join(dirpath, filename))
#
for speech_na in speech_names:
    # Read speech
    speech_na_basename = os.path.basename(speech_na)
    (speech_audio, fs) = read_audio(speech_na, target_fs=16000)

    if len(speech_audio) > 16000 * 16:

        speech_audio = speech_audio - np.mean(speech_audio)  # 消除直流分量


        out_audio_path = os.path.join(workspace, "noise", "%s" % speech_na_basename)
        create_folder(os.path.dirname(out_audio_path))
        write_audio(out_audio_path, speech_audio, fs)

        if cnt % 1000 == 0:
            print(cnt)
        cnt += 1


