import os
import librosa
import soundfile as sf
path1 = 'D:\\JMCheng\\DATASET\\Near_dns_p075_100h_target_norm_16k\\nearend\\'
path2 = 'D:\\JMCheng\\DATASET\\Near_dns_p075_100h_target_norm_16k\\target\\'

wav_names1 = os.listdir(path1)
wav_names2 = os.listdir(path2)
f = open('D:\\JMCheng\\RT_16k_DNS_exp_new\\Near_dns_p075_100h_target_norm_16k.lst', 'w')
for i in range(len(wav_names1)):
    noisy, fs = sf.read(path1 + wav_names1[i])
    duration = len(noisy) / fs
    f.write(path1 + wav_names1[i] + ' ' + path2 + wav_names2[i] + ' ' + '{}'.format(duration) + '\n')
    print("No.{} ".format(i) + wav_names1[i])
f.close()