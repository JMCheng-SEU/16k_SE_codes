import wave
import numpy as np
import soundfile
import librosa
import os

def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs

def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

# speech_names = []
#
# speech_dir="D:\博士\A-DNN文章资料\9.2实验\\test-mini"
# workspace="D:\博士\A-DNN文章资料\9.2实验"
#
speech_dir="E:\\JMCheng\\Microsoft_dataset\\DSDATASET\\CRN_48k\\PHA-CRN\\enhancements\\best_checkpoint_27_epoch"

workspace="E:\\JMCheng\\Microsoft_dataset\\DSDATASET\\CRN_48k\\PHA-CRN\\enhancements"

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
        speech_audio_test=speech_audio.shape
        speech_na_basename = os.path.basename(speech_na)
        # speech_na_basename = speech_na_basename[0:8]

        out_audio_path = os.path.join(workspace, "enh_16k_320w160", "%s" % speech_na_basename)
        create_folder(os.path.dirname(out_audio_path))
        write_audio(out_audio_path, speech_audio, fs)
