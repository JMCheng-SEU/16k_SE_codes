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



speech_dir="E:\JMCheng\Microsoft_dataset\DSDATASET\clean_testset_wav_16k"
enh_dir = "E:\JMCheng\Microsoft_dataset\DSDATASET\PHASEN_model\\rec_wav_pre"
workspace="E:\JMCheng\Microsoft_dataset\DSDATASET\PHASEN_model"

speech_names = []
for dirpath, dirnames, filenames in os.walk(speech_dir):
    for filename in filenames:
         if filename.lower().endswith(".wav"):
            # print(os.path.join(dirpath,filename))
            speech_names.append(os.path.join(dirpath, filename))
#
for speech_na in speech_names:
    # Read speech
    speech_na_basename = os.path.basename(speech_na)
    speech_fpart = os.path.splitext(speech_na_basename)[0]
    index = speech_fpart.find(".", 0)  # 找到点号的位置
    if index == -1:
        (speech_audio, fs) = read_audio(speech_na, target_fs=16000)
        speech_na_basename = os.path.basename(speech_na)
        enh_na = os.path.join(enh_dir, speech_na_basename)
        (enh_audio, _) = read_audio(enh_na, target_fs=16000)
        speech_audio = speech_audio - np.mean(speech_audio)  # 消除直流分量
        speech_audio = speech_audio / np.max(np.abs(speech_audio))  # 幅值归一化
        enh_audio = enh_audio - np.mean(enh_audio)  # 消除直流分量
        enh_audio = enh_audio / np.max(np.abs(enh_audio))  # 幅值归一化


        out_audio_path = os.path.join(workspace, "enh_new", "%s.wav" % speech_na_basename)
        create_folder(os.path.dirname(out_audio_path))
        write_audio(out_audio_path, enh_audio, fs)
        out_speech_audio_path = os.path.join(workspace, "clean_new", "%s.wav" % speech_na_basename)
        create_folder(os.path.dirname(out_speech_audio_path))
        write_audio(out_speech_audio_path, speech_audio, fs)