import wave
import numpy as np
import soundfile
import librosa
import os



def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)

    return audio, fs

def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


workspace = "D:\\JMCheng\\RT_16k\\DATASET"

speech_names = []

noise_id = 1


all_noises_wav = "D:\\JMCheng\\RT_16k\\DATASET\\T41.wav"

all_noise, fs = read_audio(all_noises_wav, target_fs=48000)

cur_pos = 0

while (cur_pos + 48000 * 10 < len(all_noise) - 1):
    temp_noise = all_noise[cur_pos:cur_pos + 48000 * 10]
    out_audio_path = os.path.join(workspace, "new423_liang_bg_noise", "noise_%d.wav" % noise_id)
    write_audio(out_audio_path, np.array(temp_noise), fs)
    noise_id += 1
    cur_pos = cur_pos + 48000 * 10
    if noise_id % 500 == 0:
        print(noise_id)




