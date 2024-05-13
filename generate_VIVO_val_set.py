import os
import soundfile
import numpy as np

import librosa


EPSILON = np.finfo(np.float32).eps

import sklearn.model_selection


workspace = "D:\\JMCheng\\DATASET\\VIVO_DATASET"
speech_dir = "D:\\JMCheng\\DATASET\\VIVO_DATASET\\final_vad_1"
speech_names = []

# def test

def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)

    return audio, fs

def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


for dirpath, dirnames, filenames in os.walk(speech_dir):
    for filename in filenames:
         if filename.lower().endswith(".wav"):
            # print(os.path.join(dirpath,filename))
            speech_names.append(os.path.join(dirpath, filename))

train, test = sklearn.model_selection.train_test_split(speech_names, test_size=100/2559)

print(test)


# for speech_na in train:
#     (speech_audio, fs) = read_audio(speech_na)
#     speech_na_basename = os.path.basename(speech_na)
#     out_audio_path = os.path.join(workspace, "trainset_clean", "%s" % speech_na_basename)
#     create_folder(os.path.dirname(out_audio_path))
#     write_audio(out_audio_path, speech_audio, fs)

for speech_na in test:
    (speech_audio, fs) = read_audio(speech_na)
    speech_na_basename = os.path.basename(speech_na)
    out_audio_path = os.path.join(workspace, "valset_vivo", "%s" % speech_na_basename)
    create_folder(os.path.dirname(out_audio_path))
    write_audio(out_audio_path, speech_audio, fs)
