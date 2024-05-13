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


speech_dir = "F:\\JMCheng\\DATASET\\VAD_DATASET_new\\test_0718\\enh"

workspace = "F:\\JMCheng\\DATASET\\VAD_DATASET_new\\test_0718"

speech_names = []

ori_path = "D:\\JMCheng\\DATASET\\VIVO_DATASET\\valset_vivo"
enh_path = "F:\\JMCheng\\real_time_exp\\16k_exp_new\\add_vivo_CGRNN_FB_LitenoSC_cmp_newwindow_PMSQE_PASE_APCSNR\\enhancements\\checkpoint_37_epoch"
# ori_path = "D:\\JMCheng\\DATASET\\VIVO_DATASET\\test1\\noisy"
# enh_path = "D:\\JMCheng\\DATASET\\VIVO_DATASET\\test1\\enh"


for dirpath, dirnames, filenames in os.walk(ori_path):
    for filename in filenames:
         if filename.lower().endswith(".wav"):
            # print(os.path.join(dirpath,filename))
            speech_names.append(os.path.join(dirpath, filename))

def compute_indicators(ori_audio, enh_audio):
    speech_audio = ori_audio
    enhanced_wav = enh_audio

    framesize = 160
    GSNR = 0
    NSR = 0
    SSR = 0

    # (speech_audio, fs) = read_audio(ori_path, target_fs=16000)
    #
    # (enhanced_wav, fs1) = read_audio(enh_path, target_fs=16000)


    noisy_wav = speech_audio[:, 0]
    vad_results = speech_audio[:, 1]

    for index in range(len(vad_results)):
        if vad_results[index] == 0:
            vad_results[index] = 1
        else:
            vad_results[index] = 0


    wav_len = min(len(noisy_wav), len(enhanced_wav))

    frame_num = int(wav_len/framesize)

    noise_frame_num = 0
    speech_frame_num = 0

    p_isspeech = 0

    def_frame_noise_energy_cur = 0
    enh_frame_noise_energy_cur = 0

    p_before = 0
    noise_segment_num = 0
    speech_segment_num = 0

    enh_snr_cur = 0
    def_snr_cur = 0
    GSNR_cur = 0
    NSR_cur = 0

    ### 音频语音抑制比
    SSR_cur_wav = 0

    ### 音频噪声抑制比
    NSR_cur_wav = 0

    ### 音频信噪比增益
    GSNR_cur_wav = 0

    noise_suppression_segment = 0
    speech_suppression_segment = 0


    for j in range(0, (frame_num*framesize), framesize):
        is_speech_num = 0

        def_frame_energy_sqrt = 0
        enh_frame_energy_sqrt = 0


        for k in range(j, j + framesize):
            is_speech_num = is_speech_num + vad_results[k]
            def_frame_energy_sqrt = def_frame_energy_sqrt + noisy_wav[k] * noisy_wav[k]
            enh_frame_energy_sqrt = enh_frame_energy_sqrt + enhanced_wav[k] * enhanced_wav[k]

        def_frame_energy_sqrt = np.sqrt(def_frame_energy_sqrt / framesize)
        enh_frame_energy_sqrt = np.sqrt(enh_frame_energy_sqrt / framesize)

        if (is_speech_num > (framesize/2)):
            p_isspeech = 1
        else:
            p_isspeech = 0

        if (j == 0):
            p_isspeech = 0

        if (p_isspeech == 1 and p_before == 0):
            ## 对噪声段进行结算
            ## def_segment_noise_energy_cur是当前段的噪声能量
            def_segment_noise_energy_cur = def_frame_noise_energy_cur / noise_frame_num
            enh_segment_noise_energy_cur = enh_frame_noise_energy_cur / noise_frame_num
            def_frame_noise_energy_cur = 0
            enh_frame_noise_energy_cur = 0

            noise_suppression_segment_cur = -noise_suppression_segment / noise_frame_num
            noise_suppression_segment = 0
            NSR_cur_wav = NSR_cur_wav + noise_suppression_segment_cur

            noise_frame_num = 0
            p_before = 1
            noise_segment_num = noise_segment_num + 1

        if (p_isspeech == 0 and p_before == 1):
            ## 对语音段进行结算

            enh_snr_cur_segment = enh_snr_cur / speech_frame_num
            def_snr_cur_segment = def_snr_cur / speech_frame_num
            ## GSNR_cur_segment = enh_snr_cur_segment - def_snr_cur_segment
            ## GSNR_cur_wav = GSNR_cur_wav + GSNR_cur_segment

            speech_suppression_segment_cur = -speech_suppression_segment / speech_frame_num
            speech_suppression_segment = 0
            SSR_cur_wav = SSR_cur_wav + speech_suppression_segment_cur

            GSNR_cur_segment = noise_suppression_segment_cur - speech_suppression_segment_cur
            GSNR_cur_wav = GSNR_cur_wav + GSNR_cur_segment

            enh_snr_cur = 0
            def_snr_cur = 0
            speech_frame_num = 0
            p_before = 0
            speech_segment_num = speech_segment_num + 1

        if (p_isspeech == 1):
            enh_frame_snr_cur = 20 * np.log10(max(enh_frame_energy_sqrt, 1e-15) / max(enh_segment_noise_energy_cur, 1e-15))
            def_frame_snr_cur = 20 * np.log10(max(def_frame_energy_sqrt, 1e-15) / max(def_segment_noise_energy_cur, 1e-15))
            enh_snr_cur = enh_snr_cur + enh_frame_snr_cur
            def_snr_cur = def_snr_cur + def_frame_snr_cur

            speech_suppression_frame = 20 * np.log10(max(enh_frame_energy_sqrt, 1e-15) / max(def_frame_energy_sqrt, 1e-15))

            speech_suppression_segment = speech_suppression_segment + speech_suppression_frame

            speech_frame_num = speech_frame_num + 1
        else:
            def_frame_noise_energy_cur = def_frame_noise_energy_cur + def_frame_energy_sqrt

            enh_frame_noise_energy_cur = enh_frame_noise_energy_cur + enh_frame_energy_sqrt

            noise_suppression_frame = 20 * np.log10(max(enh_frame_energy_sqrt, 1e-5) / max(def_frame_energy_sqrt, 1e-5))

            noise_suppression_segment = noise_suppression_segment + noise_suppression_frame

            noise_frame_num = noise_frame_num + 1

    GSNR_cur_wav = GSNR_cur_wav / speech_segment_num
    GSNR = GSNR + GSNR_cur_wav

    NSR_cur_wav = NSR_cur_wav / speech_segment_num
    NSR = NSR + NSR_cur_wav

    SSR_cur_wav = SSR_cur_wav / speech_segment_num
    SSR = SSR + SSR_cur_wav

    return GSNR, NSR, SSR


total_GSNR = 0
total_NSR = 0
total_SSR = 0
audio_num = 0
for speech_na in speech_names:
    # Read speech
    speech_na_basename = os.path.basename(speech_na)
    enh_na = os.path.join(enh_path, speech_na_basename)
    (speech_audio, fs) = read_audio(speech_na, target_fs=16000)
    (enh_audio, fs1) = read_audio(enh_na, target_fs=16000)
    try:
        GSNR, NSR, SSR = compute_indicators(speech_audio, enh_audio)
    except ZeroDivisionError as e:
        print(speech_na_basename)
    else:
        total_GSNR += GSNR
        total_NSR += NSR
        total_SSR += SSR
        audio_num += 1

print(total_GSNR / audio_num)
print(total_NSR / audio_num)
print(total_SSR / audio_num)






#
# print(GSNR)
# print(NSR)
# print(SSR)