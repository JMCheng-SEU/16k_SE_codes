import os
import librosa
import soundfile
import numpy as np

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


speech_dir = "D:\\JMCheng\\DATASET\\to_liang"
workspace = "D:\\JMCheng\\DATASET\\SIG_100h_noisy60_wo_norm_16k"


speech_names = []

cnt = 0
for dirpath, dirnames, filenames in os.walk(speech_dir):
    for filename in filenames:
         if filename.lower().endswith(".wav"):
            # print(os.path.join(dirpath,filename))
            speech_names.append(os.path.join(dirpath, filename))

for speech_na in speech_names:

    speech_basename = os.path.basename(speech_na)
    speech_fpart = os.path.splitext(speech_basename)[0]
    name_list = speech_fpart.split("_")
    if name_list[1] == "nearend":
        nearend, fs = read_audio(speech_na, target_fs=16000)
        target_name = name_list[0] + "_" + "target" + ".wav"
        # mic_name = name_list[0] + "_" + "mic" + ".wav"
        out_name = name_list[0] + ".wav"

        target_path = os.path.join(speech_dir, target_name)
        target, fs1 = read_audio(target_path, target_fs=16000)

        # mic_path = os.path.join(speech_dir, mic_name)
        # mic, fs2 = read_audio(mic_path, target_fs=16000)



        out_nearend_path = os.path.join(workspace, "nearend", "%s" % out_name)
        create_folder(os.path.dirname(out_nearend_path))
        write_audio(out_nearend_path, nearend, fs)

        out_target_path = os.path.join(workspace, "target", "%s" % out_name)
        create_folder(os.path.dirname(out_target_path))
        write_audio(out_target_path, target, fs)

        # out_mic_path = os.path.join(workspace, "mic", "%s" % out_name)
        # create_folder(os.path.dirname(out_mic_path))
        # write_audio(out_mic_path, mic, fs)

        cnt += 1

        if cnt % 1000 == 0:
            print(cnt)