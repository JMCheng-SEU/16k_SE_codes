from pesq import pesq
import soundfile
import librosa
import os
import numpy as np
from utils.utils import compute_STOI
# speech_path = "E:\JMCheng\Microsoft_dataset\DSDATASET\Val_clean_new\\p226_017.wav"
# enh_path = "E:\JMCheng\Microsoft_dataset\DSDATASET\Val_noisy_new\\p226_017.wav"
#
# clean, sr1 = sf.read(speech_path)
# noisy, sr2 = sf.read(enh_path)
#
# score = pypesq(sr1, clean, noisy, 'wb')
# print(score)

speech_dir = "F:\\JMCheng\\real_time_exp\\16k_exp_new\\DPRNN_cmp_newwindow_PMSQE_PFPL_APCSNR\\enhancements\\PCS_post_35_epoch"
clean_dir = "F:\\JMCheng\\DATASET\\100h_16k_DNS_noreverb_-5-15\\test_clean_new"

speech_names = []
PESQ_avg = 0
STOI_avg  = 0
index = 0

def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs


for dirpath, dirnames, filenames in os.walk(speech_dir):
    for filename in filenames:
         if filename.lower().endswith(".wav"):
            speech_names.append(os.path.join(dirpath, filename))

for speech_na in speech_names:
    (speech_audio, fs1) = read_audio(speech_na, target_fs=16000)
    speech_na_basename = os.path.basename(speech_na)

    speech_name_list = speech_na_basename.split('_')
    # clean_basename = 'clean_' + speech_name_list[1] + '_' + speech_name_list[2]
    clean_basename = speech_na_basename

    clean_path = os.path.join(clean_dir, clean_basename)
    (clean_audio, fs2) = read_audio(clean_path, target_fs=16000)
    PESQ_avg += pesq(fs1, clean_audio, speech_audio, 'wb')
    if len(clean_audio) > len(speech_audio):
        clean_audio = clean_audio[0:len(speech_audio)]
    STOI_avg += compute_STOI(clean_audio, speech_audio, sr=16000)
    index +=1

PESQ_avg = PESQ_avg / index
STOI_avg = STOI_avg / index

print(PESQ_avg)
print(STOI_avg)